---
title: Customize your synthetic time series data by timeseries-generator
classes: wide
categories:
  - Machine Learning
tags:
  - Python
  - Time Series Data Generation
  - Time Series Forecasting
---
**_A Tunable Synthetic Time Series Data Generator in Python_**

### TL;DR
When I worked in Nike, we open sourced a tunable synthetic time series data generator python package: [timeseries-generator](https://github.com/Nike-Inc/timeseries-generator){:target="_blank"}. You can use it to customize synthetic time series data by compositing different configurable factors such as trend, seasonality and noise. This package is easy to use and configure. We also provide a [Streamlit](https://streamlit.io/){:target="_blank"}-based web UI to generate synthetic time series.

### Why You Need This Package
In Nike, time series forecasting is one of the most impactful areas of Machine Learning (ML), which facilitates data-driven decision-making in many fields of retail supply chain management. One mission of my team in Nike is building reusable time series forecasting artefacts such as template, examples and slides, so that other ML teams can leverage our artefacts and make their ML journey smoother. Since we need to demo and share our models and solutions cross teams and organisations, using the real Nike datasets is not feasible. Synthetic time-series data is needed for prototyping, demoing and sharing our time series forecasting solutions. In this way, we can 1) get rid of security and legal risk, 2) share data between different teams and enjoy multiple demonstrations / presentations built on the same story.

The package should meet these criteria:
- Simulate time series data in real-life retail sale scenarios
- Easy to use and configurable, since a user may have limited programming experience
- The package should be extendable so that the inner enterprise developer community and the open-source community can improve it and introduce new features.

Therefore, we developed this Python package timeseries-generator and open source it [here](https://github.com/Nike-Inc/timeseries-generator){:target="_blank"}. Next, I will introduce the core concept and how to use this package in the following sections.

### How This Package Works
<figure class="align-center">
    <img class="align-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/posts/time-series-generator/diagram.png" alt="Diagram">
    <figcaption>ts = base_value * factor1 * factor2 * … * factorN + Noiser (image by author)</figcaption>
</figure>     

As illustrated by the diagram and the formula, our time series generator works in a composition way. It first starts from a constant basic value. Then, various factors which simulate the time series components such as trend, seasonality are applied on top of the basic value. In the end, a customized noiser is also appended.

To implement it in a generic and easy extendable way, Python classes [Generator](https://github.com/Nike-Inc/timeseries-generator/blob/master/timeseries_generator/generator.py){:target="_blank"} and [Factor](https://blog.devgenius.io/customize-your-synthetic-time-series-data-by-timeseries-generator-9a6669e393bc#:~:text=/timeseries_generator/*_factor.py){:target="_blank"} are introduced:
- **Generator**: a class to generate the time series. A generator contains a list of factors and noisers. By overlaying the factors and noisers, the generator can produce a customized time series
- **Factors**: an abstract class “base_factor” defined the common methods. A customized factor is a concrete class derived from base_factor. Any time series component can be defined as a factor. These factors can be anything, from random noise, linear trends, to seasonality. Factors take effect by multiplying on the base value of the generator. Your eventual generated time series is decided by the selected factors.

In order to customize your time series, select factors you want and input to the **Generator** class. By calling **Generator.generate** method, a Pandas dataframe is created, which contains the base value, all the different factor values and the final time series value in columns along with the timestamps index.

#### Built-in Factors
We have built several factors as below. You can follow these examples to extend your own factors.

- **LinearTrend**: give a linear trend based on the input slope and intercept
- **CountryYearlyTrend**: give a yearly-based market cap factor based on the GDP per capita.
- **EUEcoTrendComponents**: give a monthly changed factor based on EU industry product public data
- **HolidayTrendComponents**: simulate the holiday sale peak. It adapts the holiday days differently in different countries
- **BlackFridaySaleComponents**: simulate the BlackFriday sale event
- **WeekendTrendComponents**: more sales at weekends than on weekdays
- **FeatureRandFactorComponents**: set up different sale amounts for different stores and different product
- **ProductSeasonTrendComponents**: simulate season-sensitive product sales. In this example -code, we have 3 different types of products:
    - winter jacket: inverse-proportional to the temperature, more sales in winter
    - basketball top: proportional to the temperature, more sales in summer
    - Yoga Mat: temperature insensitive
- **White noisier**: generate random white noise

## Installation and Usage
You can install the package by `pip install timeseries-generator`.

Here is a code snippet to generate a simple time series with linear trends and white noise. We also have 2 example notebooks available in [example folder](https://github.com/Nike-Inc/timeseries-generator/tree/master/examples){:target="_blank"}:
- **generate_stationary_process**: Good for introducing the basics of the timeseries_generator. Shows how to apply simple linear trends and how to introduce features and labels, as well as random noise.
- **use_external_factors**: Goes more into detail and shows how to use the external_factors submodule. Shows how to create seasonal trends.

``` python
from timeseries_generator import LinearTrend, Generator, WhiteNoise, RandomFeatureFactor
import pandas as pd

# setting up a linear tren
lt = LinearTrend(coef=2.0, offset=1., col_name="my_linear_trend")
g = Generator(factors={lt}, features=None, date_range=pd.date_range(start="01-01-2020", end="01-20-2020"))
g.generate()
g.plot()

# update by adding some white noise to the generator
wn = WhiteNoise(stdev_factor=0.05)
g.update_factor(wn)
g.generate()
g.plot()
```

### Web-based prototyping UI
We also used [Streamlit](https://streamlit.io/){:target="_blank"} to build a web-based UI to demonstrate how to use this package to generate synthesis time series data in an interactive web UI. You can run `streamlit run examples/streamlit/app.py` to startup the streamlit application.
<figure class="align-center">
    <img class="align-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/posts/time-series-generator/streamlit.png" alt="Diagram">
    <figcaption>Streamlit dashboard (image by author)</figcaption>
</figure>  

### Summary
Timeseries-generator is a tunable synthetic time series data generator. It can simulate real-life retail sale scenarios, is easy to use and configurable, and easy to extend. We open sourced it [here](https://github.com/Nike-Inc/timeseries-generator){:target="_blank"}. Welcome to use and improve it.

#### **Special thanks to the contribution from Jaap Langemeijer to this package.**