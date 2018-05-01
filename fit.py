# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:08:38 2017

@author: Mark
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select
from bokeh.plotting import figure
#import matplotlib.pyplot as plt

def callback(attr, old, new):
    if menu.value == 'normal':
        (loc, scale) = stats.norm.fit(db['data'])
        norm_fit = stats.norm(loc=loc, scale=scale)
        demo_domain = norm_fit.ppf(demo_range)
        cdf_source.data['x'] = demo_domain
        data_source.data['perc_mle'] = norm_fit.cdf(db['data'])
        data_source.data['quant_mle'] = norm_fit.ppf(db['perc_emp'])
    elif menu.value == 'lognormal':
        (shape, loc, scale) = stats.lognorm.fit(db['data'])
        lognorm_fit = stats.lognorm(s=shape, loc=loc, scale=scale)
        demo_domain = lognorm_fit.ppf(demo_range)
        cdf_source.data['x'] = demo_domain
        data_source.data['perc_mle'] = lognorm_fit.cdf(db['data'])
        data_source.data['quant_mle'] = lognorm_fit.ppf(db['perc_emp'])
    elif menu.value == 'gamma':
        (shape, loc, scale) = stats.gamma.fit(db['data'])
        gamma_fit = stats.gamma(a=shape, loc=loc, scale=scale)
        demo_domain = gamma_fit.ppf(demo_range)
        cdf_source.data['x'] = demo_domain
        data_source.data['perc_mle'] = gamma_fit.cdf(db['data'])
        data_source.data['quant_mle'] = gamma_fit.ppf(db['perc_emp'])


# %% Get raw data
db = pd.read_csv('depth.csv')
db.columns = ['post', 'data']
n = db['data'].count()

# %% Create new dataframe with data values and corresponding cdf probabilities
db.sort_values(by='data', inplace=True)
db.reset_index(drop=True, inplace=True)

db['perc_emp'] = ((db.index+1)-0.3175)/(n+0.365)
db['perc_emp'].iloc[-1] = 0.5**(1/n)
db['perc_emp'].iloc[0] = 1 - db['perc_emp'].iloc[-1]

# %% Calculate distribution parameters
loc_mle, scale_mle = stats.norm.fit(db['data'])
norm_mle = stats.norm(loc=loc_mle, scale=scale_mle)

#loc_ls, scale ls = 
#norm_ls = 

# %% Calculate percentiles and quantiles
demo_range = np.linspace(0.000001, 0.999999, 1000)
demo_domain = norm_mle.ppf(demo_range)
cdf_source = ColumnDataSource(data={
        'x': demo_domain,
        'y': demo_range
        })

db['perc_mle'] = norm_mle.cdf(db['data'])
db['quant_mle'] = norm_mle.ppf(db['perc_emp'])
data_source = ColumnDataSource(db)

# %% Define Bokeh plots
cdf = figure(plot_width=400, tools='pan,box_zoom')
cdf.circle(db['data'], db['perc_emp'])

pp = figure(plot_width=400, tools='pan,box_zoom')
pp.circle('perc_mle', 'perc_emp', source=data_source)

qq = figure(plot_width=400, tools='pan,box_zoom')
qq.circle('data', 'quant_mle', source=data_source)

# %% Plot fitted line
cdf.line('x', 'y', source=cdf_source)

menu = Select(options=['normal', 'lognormal', 'gamma'],
              value='normal', title='Distribution')
menu.on_change('value', callback)

layout = row(menu, cdf, column(pp, qq))
curdoc().add_root(layout)
