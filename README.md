# COVID-19 SIR Model Estimation
SIR model estimation on COVID-19 cases dataset. There is a blog post describing the detail of the SIR model and COVID-19 cases dataset.

- [https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html](https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html)

## Usage

All dependencies are resolved by [Pipenv](https://pipenv.kennethreitz.org/en/latest/)

```
$ pipenv shell
$ python solver.py
```

## Data Source

The data used by this simulation is available in HDX site. 

- [HDX](https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases)

You need to put time_series_2019-ncov-Confirmed.csv and time_series_2019-ncov-Recovered.csv on data dir. Will look into auto pull from there later on.

```
usage: solver.py [-h] [--countries COUNTRY_CSV] [--start-date START_DATE]
                 [--prediction-days PREDICT_RANGE] [--S_0 S_0] [--I_0 I_0]
                 [--R_0 R_0]

optional arguments:
  -h, --help            show this help message and exit
  --countries COUNTRY_CSV
                        Countries on CSV format. It must exact match the data
                        names or you will get out of bonds error.
  --start-date START_DATE
                        Start date on MM/DD/YY format ... I know ...It
                        defaults to first data available 1/22/20
  --prediction-days PREDICT_RANGE
                        Days to predict with the model. Defaults to 120
  --S_0 S_0             NOT USED YET. Susceptible. Defaults to 15000
  --I_0 I_0             NOT USED YET. Infected. Defaults to 2
  --R_0 R_0             NOT USED YET. Recovered. Defaults to 0
```