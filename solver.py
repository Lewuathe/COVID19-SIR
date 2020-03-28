#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import argparse
import sys
import json
import ssl
import urllib.request


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--countries',
        action='store',
        dest='countries',
        help='Countries on CSV format. ' +
        'It must exact match the data names or you will get out of bonds error.',
        metavar='COUNTRY_CSV',
        type=str,
        default="")
    
    parser.add_argument(
        '--download-data',
        action='store_true',
        dest='download_data',
        help='Download fresh data and then run',
        default=False
    )

    parser.add_argument(
        '--start-date',
        required=False,
        action='store',
        dest='start_date',
        help='Start date on MM/DD/YY format ... I know ...' +
        'It defaults to first data available 1/22/20',
        metavar='START_DATE',
        type=str,
        default="1/22/20")

    parser.add_argument(
        '--prediction-days',
        required=False,
        dest='predict_range',
        help='Days to predict with the model. Defaults to 150',
        metavar='PREDICT_RANGE',
        type=int,
        default=150)

    parser.add_argument(
        '--S_0',
        required=False,
        dest='s_0',
        help='S_0. Defaults to 100000',
        metavar='S_0',
        type=int,
        default=100000)

    parser.add_argument(
        '--I_0',
        required=False,
        dest='i_0',
        help='I_0. Defaults to 2',
        metavar='I_0',
        type=int,
        default=2)

    parser.add_argument(
        '--R_0',
        required=False,
        dest='r_0',
        help='R_0. Defaults to 0',
        metavar='R_0',
        type=int,
        default=10)

    args = parser.parse_args()

    country_list = []
    if args.countries != "":
        try:
            countries_raw = args.countries
            country_list = countries_raw.split(",")
        except Exception:
            sys.exit("QUIT: countries parameter is not on CSV format")
    else:
        sys.exit("QUIT: You must pass a country list on CSV format.")

    return (country_list, args.download_data, args.start_date, args.predict_range, args.s_0, args.i_0, args.r_0)


def remove_province(input_file, output_file):
    input = open(input_file, "r")
    output = open(output_file, "w")
    output.write(input.readline())
    for line in input:
        if line.lstrip().startswith(","):
            output.write(line)
    input.close()
    output.close()


def download_data(url_dictionary):
    #Lets download the files
    for url_title in url_dictionary.keys():
        urllib.request.urlretrieve(url_dictionary[url_title], "./data/" + url_title)


def load_json(json_file_str):
    # Loads  JSON into a dictionary or quits the program if it cannot.
    try:
        with open(json_file_str, "r") as json_file:
            json_variable = json.load(json_file)
            return json_variable
    except Exception:
        sys.exit("Cannot open JSON file: " + json_file_str)


class Learner(object):
    def __init__(self, country, loss, start_date, predict_range,s_0, i_0, r_0, d_0):
        self.country = country
        self.loss = loss
        self.start_date = start_date
        self.predict_range = predict_range
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.d_0 = d_0


    def load_confirmed(self, country):
        df = pd.read_csv('data/time_series_19-covid-Confirmed-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]


    def load_recovered(self, country):
        df = pd.read_csv('data/time_series_19-covid-Recovered-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]


    def load_dead(self, country):
        df = pd.read_csv('data/time_series_19-covid-Deaths-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]
    

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, a, b, data, recovered, death, country, s_0, i_0, r_0, d_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            D = y[3]
            return [-beta*S*I, beta*S*I-(a+b)*I, a*I, b*I]
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        ivp = solve_ivp(SIR, [0, size], [s_0,i_0,r_0,d_0], t_eval=np.arange(0, size, 1))
        return new_index, extended_actual, extended_recovered, extended_death, ivp


    def train(self):
        recovered = self.load_recovered(self.country)
        death = self.load_dead(self.country)
        data = (self.load_confirmed(self.country) - recovered - death)

        optimal = minimize(loss,
            [0.001, 0.001, 0.001],
            args=(data, recovered, death, self.s_0, self.i_0, self.r_0, self.d_0),
            method='L-BFGS-B',
            bounds=[(0.00000001, 0.4), (0.00000001, 0.4), (0.00000001, 0.4)])

        print(optimal)
        beta, a, b = optimal.x
        new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, a, b, data, recovered, death, self.country, self.s_0, self.i_0, self.r_0, self.d_0)

        df = pd.DataFrame({
            'Infected data': extended_actual,
            'Recovered data': extended_recovered,
            'Death data': extended_death,
            'Susceptible': prediction.y[0],
            'Infected': prediction.y[1],
            'Recovered': prediction.y[2],
            'Extimated Deaths': prediction.y[3]},
            index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        df.plot(ax=ax)
        print(f"country={self.country}, beta={beta:.8f}, a={a:.8f}, b={b:.8f},  gamma={(a+b):.8f}, r_0:{(beta/(a+b)):.8f}")
        fig.savefig(f"{self.country}.png")


def loss(point, data, recovered, death, s_0, i_0, r_0, d_0):
    size = len(data)
    beta, a, b = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        D = y[3]
        return [-beta*S*I, beta*S*I-(a+b)*I, a*I, b*I]
    solution = solve_ivp(SIR, [0, size], [s_0,i_0,r_0,d_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered)**2))
    l3 = np.sqrt(np.mean((solution.y[3] - death)**2))
    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2 + 0.9*l3


def main():

    countries, download, startdate, predict_range , s_0, i_0, r_0 = parse_arguments()

    if download:
        data_d = load_json("./data_url.json")
        download_data(data_d)

    remove_province('data/time_series_19-covid-Confirmed.csv', 'data/time_series_19-covid-Confirmed-country.csv')
    remove_province('data/time_series_19-covid-Recovered.csv', 'data/time_series_19-covid-Recovered-country.csv')
    remove_province('data/time_series_19-covid-Deaths.csv', 'data/time_series_19-covid-Deaths-country.csv')

    for country in countries:
        learner = Learner(country, loss, startdate, predict_range, s_0, i_0, r_0, 0)
        #try:
        learner.train()
        #except BaseException:
        #    print('WARNING: Problem processing ' + str(country) +
        #        '. Be sure it exists in the data exactly as you entry it.' +
        #        ' Also check date format if you passed it as parameter.')
           

if __name__ == '__main__':
    main()
