import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select
from bokeh.plotting import figure


# TODO: handle cases where least squares fails
# TODO: put populating mle and least squares data in its own function
# TODO: figure out how to handle distributions with >1 shape parameter
# TODO: figure out how to make least squares work with arbitrary number of shape variables

def myFun(x, data, perc_emp):
    return perc_emp - stats.norm.cdf(data, loc=x[0], scale=x[1])


def freeze_dist(dist, loc, scale, shape=None):
    if dist.shapes is None:
        return dist(scale=scale, loc=loc)
    else:
        # TODO: code to count number of shape parameters for dist to put in dist func call
        return dist(shape, scale=scale, loc=loc)


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


# %% Get raw data.
db = pd.read_csv('data.csv')
db.columns = ['post', 'data']
n = db['data'].count()

# %% Calculate empirical percentiles for ordered (ranked) data.
db.sort_values(by='data', inplace=True)
db.reset_index(drop=True, inplace=True)
db['perc_emp'] = ((db.index + 1) - 0.3175) / (n + 0.365)
db['perc_emp'].iloc[-1] = 0.5 ** (1 / n)
db['perc_emp'].iloc[0] = 1 - db['perc_emp'].iloc[-1]

# %% Calculate distribution parameters for default (Normal) distribution.
dist_type = stats.norm
params_mle = dist_type.fit(db['data'])
dist_mle = freeze_dist(dist_type, params_mle[0], params_mle[1])

# %% Calculate percentiles and quantiles using fitted distribution and data as inputs.
db['perc_mle'] = dist_mle.cdf(db['data'])
db['quant_mle'] = dist_mle.ppf(db['perc_emp'])

res = optimize.least_squares(myFun, params_mle, args=(db['data'], db['perc_emp']), method='lm')
x = res.x
dist_ls = freeze_dist(dist_type, loc=x[0], scale=x[1])
db['perc_ls'] = dist_ls.cdf(db['data'])
db['quant_ls'] = dist_ls.ppf(db['perc_emp'])

db['mle_cdf'] = np.linspace(0.000001, 0.999999, len(db))
db['mle_x'] = dist_mle.ppf(db['mle_cdf'])
db['mle_pdf'] = dist_mle.pdf(db['mle_x'])

db['ls_cdf'] = np.linspace(0.000001, 0.999999, len(db))
db['ls_x'] = dist_ls.ppf(db['ls_cdf'])
db['ls_pdf'] = dist_ls.pdf(db['ls_x'])

db.plot(x='mle_x', y='mle_pdf')
db.plot(x='ls_x', y='ls_pdf', ax=plt.gca())
plt.hist(db['data'], density=True)

db.plot(x='mle_x', y='mle_cdf')
db.plot(x='ls_x', y='ls_cdf', ax=plt.gca())
plt.scatter(x=db['data'], y=db['perc_emp'], )

plt.show()

data_source = ColumnDataSource(db)

# %% Calculate datapoints to represent the assumed distribution.
demo_range = np.linspace(0.000001, 0.999999, 1000)
demo_domain = dist_mle.ppf(demo_range)
cdf_source = ColumnDataSource(data={
    'x': demo_domain,
    'y': demo_range
})


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
