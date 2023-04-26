import panel as pn
import altair as alt
from altair import datum
import pandas as pd
from vega_datasets import data
import datetime as dt

alt.renderers.enable('default')
pn.extension('vega')

source = data.stocks()
source = pd.DataFrame(source)
source.head()
# https://github.com/bendoesdata/panel-altair-dashboard/blob/master/panel-altair-demo.ipynb
# create list of company names (tickers) to use as options
tickers = ['AAPL', 'GOOG', 'IBM', 'MSFT']
# this creates the dropdown widget
ticker = pn.widgets.Select(name= 'Company', options= tickers)

# this creates the date range slider
date_range_slider = pn.widgets.DateRangeSlider(
name = 'Date Range Slider',
start=dt.datetime(2001, 1, 1), end=dt.datetime(2010, 1, 1),
value=(dt.datetime(2001, 1, 1), dt.datetime(2010, 1, 1))
)

title = '### Stock Price Dashboard'
subtitle = 'This dashboard allows you to select a company and date range to see stock prices.'


@pn.depends(ticker.param.value, date_range_slider.param.value)
def get_plot(ticker, date_range):
     df = source # define dfconda install -c bokeh jupyter_bokeh
     df['date'] = pd.to_datetime(df['date'])
     # create date filter using values from the range slider
     # store the first and last date range slider value in a var
     start_date = date_range_slider.value[0] 
     end_date = date_range_slider.value[1] 
     # create filter mask for the dataframe
     mask = (df['date'] > start_date) & (df['date'] <= end_date)
     df = df.loc[mask] # filter the dataframe
     # create the Altair chart object
     chart = alt.Chart(df).mark_line().encode(x='date', y='price',      tooltip=alt.Tooltip(['date','price'])).transform_filter(
(datum.symbol == ticker) # this ties in the filter 
)
     return chart

dashboard = pn.Row(pn.Column(title, subtitle, ticker, date_range_slider),
get_plot # our draw chart function!
)

dashboard.servable()
