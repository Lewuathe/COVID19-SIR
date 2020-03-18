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

Four images will be generated.

- [Japan.png](/Japan.png)
- [Republic of Korea.png](/Republic%20of%20Korea.png)
- [Italy.png](/Italy.png)
- [Iran (Islamic Republic of).png](/Iran%20(Islamic%20Republic%20of).png)

## Data Source

The data used by this simulation is available in HDX site.

- [HDX](https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases)
