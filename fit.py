import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
from bokeh.io import curdoc
from bokeh.layouts import column, row, widgetbox, gridplot
from bokeh.models import ColumnDataSource, Select, Title
from bokeh.plotting import figure


# TODO: handle cases where least squares fails
# TODO: is it a problem to return db from calculate_fitted_data? Changing db inside the function changes it outside too.
# TODO: add histograms, more graphs.
# TODO: allow for fixed location parameter.
def perc_emp_filliben(indices):
    n_values = len(indices)
    perc_emp = ((indices + 1) - 0.3175) / (n_values + 0.365)
    perc_emp[-1] = 0.5 ** (1 / n_values)
    perc_emp[0] = 1 - perc_emp[-1]
    return np.array(perc_emp)


def min_fun(x, data, dist_str):
    perc_emp = perc_emp_filliben(data.index.get_values())
    dist_ls = freeze_dist(dist_str, x)
    quant_emp = dist_ls.ppf(perc_emp)
    return np.array(data - quant_emp)


def freeze_dist(dist_str, params):
    dist = getattr(stats, dist_str)
    return dist(*params)


def calculate_fitted_data(df, dist_type):
    # Calculate distribution parameters using mle, and calculate corresponding percentiles and quantiles
    params_mle = getattr(stats, dist_type).fit(df['data'])
    dist_mle = freeze_dist(dist_type, params_mle)
    df['perc_mle'] = dist_mle.cdf(df['data'])
    df['quant_mle'] = dist_mle.ppf(df['perc_emp'])

    # Calculate distribution parameters using ls, and calculate corresponding percentiles and quantiles
    ls_results = optimize.least_squares(min_fun, params_mle, args=(df['data'], dist_type), method='lm')
    params_ls = ls_results.x
    dist_ls = freeze_dist(dist_type, params_ls)
    df['perc_ls'] = dist_ls.cdf(df['data'])
    df['quant_ls'] = dist_ls.ppf(df['perc_emp'])

    return df, dist_mle, dist_ls


def callback(attr, old, new):
    dist_type = menu.value
    _, dist_mle, dist_ls = calculate_fitted_data(df, dist_type)

    data_source.data['x_mle'] = dist_mle.ppf(data_source.data['cdf_y'])
    data_source.data['pdf_mle'] = dist_mle.pdf(data_source.data['x_mle'])
    data_source.data['perc_mle'] = dist_mle.cdf(data_source.data['data'])
    data_source.data['quant_mle'] = dist_mle.ppf(data_source.data['perc_emp'])

    data_source.data['x_ls'] = dist_ls.ppf(data_source.data['cdf_y'])
    data_source.data['pdf_ls'] = dist_ls.pdf(data_source.data['x_ls'])
    data_source.data['perc_ls'] = dist_ls.cdf(data_source.data['data'])
    data_source.data['quant_ls'] = dist_ls.ppf(data_source.data['perc_emp'])

    quantile_unity.data['x'] = (0, max(max(data_source.data['quant_ls']), max(data_source.data['quant_mle'])))
    quantile_unity.data['y'] = (0, max(max(data_source.data['quant_ls']), max(data_source.data['quant_mle'])))
    # quantile_unity.data['y'] = (0, max(max(data_source.data['quant_ls']), max(data_source.data['quant_mle'])))

# %% Get raw data.
df = pd.read_csv('data.csv').dropna()
df.columns = ['post', 'data']
df = df.sort_values(by='data').reset_index(drop=True)
n = df['data'].count()

# %% Calculate empirical percentiles for ordered (ranked) data.
df['perc_emp'] = perc_emp_filliben(df.index.get_values())

# %% Calculate distribution parameters for default (Normal) distribution.
dist_type = 'norm'
df, dist_mle, dist_ls = calculate_fitted_data(df, dist_type)

# TODO: make theoretical distributions not dependent on length of db.
df['cdf_y'] = np.linspace(0.000001, 0.999999, len(df))

df['x_mle'] = dist_mle.ppf(df['cdf_y'])
df['pdf_mle'] = dist_mle.pdf(df['x_mle'])

df['x_ls'] = dist_ls.ppf(df['cdf_y'])
df['pdf_ls'] = dist_ls.pdf(df['x_ls'])

data_source = ColumnDataSource(df)
quantile_unity = ColumnDataSource(dict(x=(0, max(max(df['quant_ls']), max(df['quant_mle']))),
                                       y=(0, max(max(df['quant_ls']), max(df['quant_mle'])))))
# quantile_unity = ColumnDataSource(dict(x=(0, max(df['data'])), y=(0, max(df['data']))))

# %% Calculate datapoints to represent the assumed distribution.
demo_range = np.linspace(0.000001, 0.999999, 1000)
demo_domain = dist_mle.ppf(demo_range)
cdf_source = ColumnDataSource(data={
    'x': demo_domain,
    'y': demo_range
})

# Define histogram
bin_heights, bin_edges = np.histogram(df['data'], normed=True, bins='auto')
hist_df = pd.DataFrame({'bin_heights': bin_heights})
hist_df['bin_mids'] = pd.Series(bin_edges).rolling(window=2).mean().dropna().reset_index(drop=True)
hist_df['bin_widths'] = pd.Series(bin_edges).diff().dropna().reset_index(drop=True)
hist_source = ColumnDataSource(hist_df)

# %% Define Bokeh plots
bin_range = max(bin_edges) - min(bin_edges)
hist = figure(plot_width=400, tools='pan,box_zoom,reset', title='Histogram',
              x_range=[min(bin_edges) - 0.1 * bin_range, max(bin_edges) + 0.1 * bin_range],
              y_range=[0, max(bin_heights) * 1.1])
hist.yaxis.axis_label = 'Probability Density'
hist.yaxis.axis_label_text_font_style = 'bold'
hist.vbar(x='bin_mids', width='bin_widths', top='bin_heights', source=hist_source, color='red')
hist.line(x='x_mle', y='pdf_mle', color='green', source=data_source, line_width=3)
hist.line(x='x_ls', y='pdf_ls', color='blue', source=data_source, line_width=3)

cdf = figure(plot_width=400, tools='pan,box_zoom,reset', title='CDF')
cdf.circle('data', 'perc_emp', color='gray', source=data_source, alpha=0.5)
cdf.line('x_mle', 'cdf_y', color='green', source=data_source, line_width=3)
cdf.line('x_ls', 'cdf_y', color='blue', source=data_source, line_width=3)

pp = figure(plot_width=400, tools='pan,box_zoom,reset', title='pp')
pp.xaxis.axis_label = 'Fitted Probabilities'
pp.yaxis.axis_label = 'Empirical Probabilities'
pp.xaxis.axis_label_text_font_style = 'bold'
pp.yaxis.axis_label_text_font_style = 'bold'
pp.circle('perc_mle', 'perc_emp', color='green', source=data_source)
pp.circle('perc_ls', 'perc_emp', color='blue', source=data_source)
pp.line(x=[0, 1], y=[0, 1], color='gray')

qq = figure(plot_width=400, tools='pan,box_zoom,reset', title='qq')
qq.xaxis.axis_label = 'Fitted Quantiles'
qq.yaxis.axis_label = 'Empirical Quantiles'
qq.xaxis.axis_label_text_font_style = 'bold'
qq.yaxis.axis_label_text_font_style = 'bold'
qq.circle('quant_mle', 'data', color='green', source=data_source)
qq.circle('quant_ls', 'data', color='blue', source=data_source)
qq.line('x', y='y', color='gray', source=quantile_unity)

options = [x for x in dir(stats) if isinstance(getattr(stats, x), stats.rv_continuous)]
menu = Select(options=options, value='norm', title='Distribution')
widgets = widgetbox(menu, width=400)
menu.on_change('value', callback)

grid = gridplot([hist, cdf, widgets],
                [pp, qq, None])
curdoc().add_root(grid)
