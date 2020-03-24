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
