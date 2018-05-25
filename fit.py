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
# TODO: is it a problem to return db from calculate_fitted_data? Changing db inside the function changes it outside too.
# TODO: add histograms, more graphs.
# TODO: allow for fixed location parameter.
def min_fun(x, data, dist_str):
    res = stats.probplot(data, x, dist=dist_str)
    return res[0][0] - res[0][1]


def freeze_dist(dist_str, params):
    dist = getattr(stats, dist_str)
    return dist(*params)


def calculate_fitted_data(db, dist_type):
    # Calculate distribution parameters using mle, and calculate corresponding percentiles and quantiles
    params_mle = getattr(stats, dist_type).fit(db['data'])
    dist_mle = freeze_dist(dist_type, params_mle)
    db['perc_mle'] = dist_mle.cdf(db['data'])
    db['quant_mle'] = dist_mle.ppf(db['perc_emp'])

    # Calculate distribution parameters using ls, and calculate corresponding percentiles and quantiles
    ls_results = optimize.least_squares(min_fun, params_mle, args=(db['data'], dist_type), method='lm')
    params_ls = ls_results.x
    dist_ls = freeze_dist(dist_type, params_ls)
    db['perc_ls'] = dist_ls.cdf(db['data'])
    db['quant_ls'] = dist_ls.ppf(db['perc_emp'])

    return db, dist_mle, dist_ls


def callback(attr, old, new):
    dist_type = menu.value
    _, dist_mle, dist_ls = calculate_fitted_data(db, dist_type)
    demo_domain_mle = dist_mle.ppf(demo_range)
    demo_domain_ls = dist_ls.ppf(demo_range)
    cdf_source.data['x'] = demo_domain_mle
    data_source.data['perc_mle'] = db['perc_mle']
    data_source.data['quant_mle'] = db['quant_mle']

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
dist_type = 'norm'
db, dist_mle, dist_ls = calculate_fitted_data(db, dist_type)

# TODO: make theoretical distributions not dependent on length of db.
db['mle_cdf'] = np.linspace(0.000001, 0.999999, len(db))
db['mle_x'] = dist_mle.ppf(db['mle_cdf'])
db['mle_pdf'] = dist_mle.pdf(db['mle_x'])

db['ls_cdf'] = np.linspace(0.000001, 0.999999, len(db))
db['ls_x'] = dist_ls.ppf(db['ls_cdf'])
db['ls_pdf'] = dist_ls.pdf(db['ls_x'])

# db.plot(x='mle_x', y='mle_pdf')
# db.plot(x='ls_x', y='ls_pdf', ax=plt.gca())
# plt.hist(db['data'], density=True)
#
# db.plot(x='mle_x', y='mle_cdf')
# db.plot(x='ls_x', y='ls_cdf', ax=plt.gca())
# plt.scatter(x=db['data'], y=db['perc_emp'])
#
# plt.show()

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

# options = ['norm', 'lognorm', 'gamma']
options = [x for x in dir(stats) if isinstance(getattr(stats, x), stats.rv_continuous)]
menu = Select(options=options, value='norm', title='Distribution')
menu.on_change('value', callback)

layout = row(menu, cdf, column(pp, qq))
curdoc().add_root(layout)
