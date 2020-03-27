# COVID-19 SIR Model Estimation
SIR model estimation on COVID-19 cases dataset. There is a blog post describing the detail of the SIR model and COVID-19 cases dataset.

- [https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html](https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html)

![japan](/Japan.png)

## Usage

All dependencies are resolved by [Pipenv](https://pipenv.kennethreitz.org/en/latest/)

```
$ pipenv shell
$ python solver.py
```

Option to run
```
usage: solver.py [-h] [--countries COUNTRY_CSV] [--download-data]
                 [--start-date START_DATE] [--prediction-days PREDICT_RANGE]
                 [--S_0 S_0] [--I_0 I_0] [--R_0 R_0]

optional arguments:
  -h, --help            show this help message and exit
  --countries COUNTRY_CSV
                        Countries on CSV format. It must exact match the data
                        names or you will get out of bonds error.
  --download-data       Download fresh data and then run
  --start-date START_DATE
                        Start date on MM/DD/YY format ... I know ...It
                        defaults to first data available 1/22/20
  --prediction-days PREDICT_RANGE
                        Days to predict with the model. Defaults to 150
  --S_0 S_0             S_0. Defaults to 100000
  --I_0 I_0             I_0. Defaults to 2
  --R_0 R_0             R_0. Defaults to 0
```


## Data Sources

The data used by this simulation is available in:

- [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19)

