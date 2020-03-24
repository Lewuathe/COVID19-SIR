#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import argparse
import sys

#SIR
S_0 = 15000
I_0 = 2
R_0 = 0


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
        help='Days to predict with the model. Defaults to 120',
        metavar='PREDICT_RANGE',
        type=int,
        default=120)

    parser.add_argument(
        '--S_0',
        required=False,
        dest='S_0',
        help='NOT USED YET. Susceptible. Defaults to 15000',
        metavar='S_0',
        type=int,
        default=15000)

    parser.add_argument(
        '--I_0',
        required=False,
        dest='I_0',
        help='NOT USED YET. Infected. Defaults to 2',
        metavar='I_0',
        type=int,
        default=2)

    parser.add_argument(
        '--R_0',
        required=False,
        dest='R_0',
        help='NOT USED YET. Recovered. Defaults to 0',
        metavar='R_0',
        type=int,
        default=0)    

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

    return (country_list, args.start_date, args.predict_range)

class Learner(object):
    def __init__(self, country, loss, start_date, predict_range):
        self.country = country
        self.loss = loss
        self.start_date = start_date
        self.predict_range = predict_range

    def load_confirmed(self, country):
      df = pd.read_csv('data/time_series_2019-ncov-Confirmed.csv')
      country_df = df[df['Country/Region'] == country]
      return country_df.iloc[0].loc[self.start_date:]

    def load_recovered(self, country):
      df = pd.read_csv('data/time_series_2019-ncov-Recovered.csv')
      country_df = df[df['Country/Region'] == country]
      return country_df.iloc[0].loc[self.start_date:]

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, data, recovered, country):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        return new_index, extended_actual, extended_recovered, solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1))

    def train(self):
        data = self.load_confirmed(self.country)
        recovered = self.load_recovered(self.country)
        optimal = minimize(loss, [0.001, 0.001], args=(data, recovered), method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
        print(optimal)
        beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, prediction = self.predict(beta, gamma, data, recovered, self.country)
        df = pd.DataFrame({'Infected real': extended_actual, 'Recovered real': extended_recovered, 'Susceptible': prediction.y[0], 'Infected modeled': prediction.y[1], 'Recovered modeled': prediction.y[2]}, index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        df.plot(ax=ax)
        print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
        fig.savefig(f"{self.country}.png")


def loss(point, data, recovered):
    size = len(data)
    beta, gamma = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
    solution = solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered)**2))
    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2



def main():

    countries, startdate, predict_range = parse_arguments()

    for country in countries:
        learner = Learner(country, loss, startdate, predict_range)
        #try:
        learner.train()
        #except BaseException:
        #    print('WARNING: Problem processing above country. ' +
        #        'Be sure it exists in the data exactly as you entry it.')
           

if __name__ == '__main__':
    main()
