#import os
#import pathlib
import pandas as pd
import numpy as np
import scipy.stats as stats
import fit
from bokeh.io import curdoc
from bokeh.layouts import widgetbox, gridplot
from bokeh.models import ColumnDataSource, Select, DataTable, TableColumn, NumberFormatter, TextInput
from bokeh.plotting import figure


# TODO: handle cases where least squares fails.
# TODO: when using manual input of loc parameter, handle case where dataset is empty because outside of ub/lb
# TODO: add option to save plot.
def load_data(data_source_menu_value, dist_type, loc):
    #data_path = pathlib.Path('data\\' + data_source_menu_value)
    data_path = data_source_menu_value
    df = pd.read_csv(data_path).dropna()
    df.columns = ['data']

    # Calculate empirical percentiles for ordered (ranked) data.
    df['data'] = df.sort_values(by='data').reset_index(drop=True)

    # Calculate distribution parameters for default (Normal) distribution.
    dist_mle, _ = fit.calc_fit_from_data(df['data'], dist_type, loc, 'mle')
    dist_ls, _ = fit.calc_fit_from_data(df['data'], dist_type, loc, 'ls')

    return df['data'], dist_mle, dist_ls


def update_data_source(data, dist_mle, dist_ls):
    perc_emp = fit.perc_emp_filliben(np.linspace(1, len(data), len(data)))
    return dict(
        data=data,
        perc_emp=perc_emp,
        perc_mle=dist_mle.cdf(data),
        quant_mle=dist_mle.ppf(perc_emp),
        perc_ls=dist_ls.cdf(data),
        quant_ls=dist_ls.ppf(perc_emp),
    )


def update_data_fit_source(cdf_y, dist_mle, dist_ls):
    x_mle = dist_mle.ppf(cdf_y)
    pdf_mle = dist_mle.pdf(x_mle)
    x_ls = dist_ls.ppf(cdf_y)
    pdf_ls = dist_ls.pdf(x_ls)
    out_dict = dict(
        cdf_y=cdf_y,
        x_mle=x_mle,
        pdf_mle=pdf_mle,
        x_ls=x_ls,
        pdf_ls=pdf_ls,
    )
    return out_dict


def update_metrics_source(method, data_source, dist_mle, dist_ls, loc_val):
    means = (data_source.data['data'].mean(), dist_mle.mean(), dist_ls.mean())
    sds = (data_source.data['data'].std(), dist_mle.std(), dist_ls.std())
    scales = (np.nan, dist_mle.args[-1], dist_ls.args[-1])
    locs = (np.nan, dist_mle.args[-2], dist_ls.args[-2])
    fixed_loc = loc_val != ''
    k = fit.calc_k(getattr(stats, dist_mle.dist.name), fixed_loc)
    if fixed_loc:
        mle_likelihoods = dist_mle.pdf(data_source.data['data'][data_source.data['data'] > float(loc_val)])
        ls_likelihoods = dist_ls.pdf(data_source.data['data'][data_source.data['data'] > float(loc_val)])
    else:
        mle_likelihoods = dist_mle.pdf(data_source.data['data'])
        ls_likelihoods = dist_ls.pdf(data_source.data['data'])
    aics = (np.nan, fit.calc_aic(mle_likelihoods, k), fit.calc_aic(ls_likelihoods, k))
    if dist_mle.args[:-2]:
        shapes = (np.nan, dist_mle.args[:-2], dist_ls.args[:-2])
    else:
        shapes = (np.nan, np.nan, np.nan)
    return dict(
        method=method,
        mean=means,
        sd=sds,
        scale=scales,
        loc=locs,
        shape=shapes,
        aic=aics
    )


def on_change_data_source(attr, old, new):
    # Load data.
    data, dist_mle, dist_ls = load_data(data_source_menu.value, dist_menu.value, loc_val_input.value)

    # Updated data_source.
    data_source.data = update_data_source(data, dist_mle, dist_ls)

    # Update data_fit.
    data_fit_source.data = update_data_fit_source(data_fit_source.data['cdf_y'], dist_mle, dist_ls)

    # Update metrics_source.
    metrics_source.data = update_metrics_source(metrics_source.data['method'], data_source,
                                                dist_mle, dist_ls, loc_val_input.value)

    bin_heights, bin_edges = np.histogram(data, normed=True, bins='auto')
    hist.x_range.start = min(bin_edges) - 0.1 * bin_range
    hist.x_range.end = max(bin_edges) + 0.1 * bin_range
    hist.y_range.end = max(bin_heights) * 1.1
    hist_source.data['bin_heights'] = bin_heights
    hist_source.data['bin_mids'] = pd.Series(bin_edges).rolling(window=2).mean().dropna().reset_index(drop=True)
    hist_source.data['bin_widths'] = pd.Series(bin_edges).diff().dropna().reset_index(drop=True)

    qq_line_source.data['x'] = (0, max(data))
    qq_line_source.data['y'] = (0, max(data))


def on_dist_change(attr, old, new):
    dist_type = dist_menu.value
    loc = loc_val_input.value
    dist_mle, _ = fit.calc_fit_from_data(data_source.data['data'], dist_type, loc, 'mle')
    dist_ls, _ = fit.calc_fit_from_data(data_source.data['data'], dist_type, loc, 'ls')

    # Updated data_source.
    data_source.data = update_data_source(data_source.data['data'], dist_mle, dist_ls)

    # Update data_fit.
    data_fit_source.data = update_data_fit_source(data_fit_source.data['cdf_y'], dist_mle, dist_ls)

    # Update metrics_source.
    metrics_source.data = update_metrics_source(metrics_source.data['method'], data_source,
                                                dist_mle, dist_ls, loc_val_input.value)


#if ('bk_script' in __name__) or (__name__ == '__main__'):
default_data_file = 'data.csv'
default_dist_type = 'norm'
data, dist_mle, dist_ls = load_data(default_data_file, default_dist_type, '')

# Updated data_source.
data_source = ColumnDataSource(update_data_source(data, dist_mle, dist_ls))

# Populate data_fit ColumnDataSource.
cdf_y = np.linspace(0.000001, 0.999999, 100)
data_fit_source = ColumnDataSource(update_data_fit_source(cdf_y, dist_mle, dist_ls))

# Populate metrics_source ColumnDataSource.
metrics_source = ColumnDataSource(update_metrics_source(('Source Data',
                                                         'Maximum Likelihood',
                                                         'Least Squares (Quantiles)'),
                                                        data_source, dist_mle, dist_ls, ''))

# Define tools for Bokeh plots.
tools = 'pan,box_zoom,reset,save'

# Histogram
bin_heights, bin_edges = np.histogram(data, normed=True, bins='auto')
hist_df = pd.DataFrame({'bin_heights': bin_heights})
hist_df['bin_mids'] = pd.Series(bin_edges).rolling(window=2).mean().dropna().reset_index(drop=True)
hist_df['bin_widths'] = pd.Series(bin_edges).diff().dropna().reset_index(drop=True)
hist_source = ColumnDataSource(hist_df)
bin_range = max(bin_edges) - min(bin_edges)
hist = figure(plot_width=400, plot_height=300, tools=tools, title='Histogram',
              x_range=[min(bin_edges) - 0.1 * bin_range, max(bin_edges) + 0.1 * bin_range],
              y_range=[0, max(bin_heights) * 1.1])
hist.yaxis.axis_label = 'Probability Density'
hist.yaxis.axis_label_text_font_style = 'bold'
hist.vbar(x='bin_mids', width='bin_widths', top='bin_heights', source=hist_source, color='red', legend='Data')
hist.line(x='x_mle', y='pdf_mle', color='green', source=data_fit_source, line_width=3, legend='MLE')
hist.line(x='x_ls', y='pdf_ls', color='blue', source=data_fit_source, line_width=3, legend='LS')

cdf = figure(plot_width=400, plot_height=300, tools=tools, title='CDF',
             x_range=[min(bin_edges) - 0.1 * bin_range, max(bin_edges) + 0.1 * bin_range])
cdf.circle('data', 'perc_emp', color='gray', source=data_source, alpha=0.5)
cdf.line('x_mle', 'cdf_y', color='green', source=data_fit_source, line_width=3)
cdf.line('x_ls', 'cdf_y', color='blue', source=data_fit_source, line_width=3)

# Probability plot
pp = figure(plot_width=400, plot_height=300, tools=tools, title='pp')
pp.xaxis.axis_label = 'Theoretical Probabilities'
pp.yaxis.axis_label = 'Empirical Probabilities'
pp.xaxis.axis_label_text_font_style = 'bold'
pp.yaxis.axis_label_text_font_style = 'bold'
pp.circle('perc_mle', 'perc_emp', color='green', source=data_source)
pp.circle('perc_ls', 'perc_emp', color='blue', source=data_source)
pp.line(x=[0, 1], y=[0, 1], color='gray')

# Quantile plot
qq = figure(plot_width=400, plot_height=300, tools=tools, title='qq')
qq.xaxis.axis_label = 'Theoretical Quantiles'
qq.yaxis.axis_label = 'Empirical Quantiles'
qq.xaxis.axis_label_text_font_style = 'bold'
qq.yaxis.axis_label_text_font_style = 'bold'
qq.circle('quant_mle', 'data', color='green', source=data_source)
qq.circle('quant_ls', 'data', color='blue', source=data_source)
qq_line_source = ColumnDataSource(dict(x=(0, max(data)), y=(0, max(data))))
qq.line(x='x', y='y', color='gray', source=qq_line_source)

# Distribution dropdown widget
options = [x for x in dir(stats) if isinstance(getattr(stats, x), stats.rv_continuous)]
dist_menu = Select(options=options, value='norm', title='Distribution:')
dist_menu.on_change('value', on_dist_change)

# Data source dropdown widget
#files = [x for x in os.listdir('data') if x.split('.')[-1] == 'csv']
files = ['data.csv', 'data2.csv']
data_source_menu = Select(options=files, value='data.csv', title='Source from \'data\' directory:')
data_source_menu.on_change('value', on_change_data_source)

# Table widget
num_format = NumberFormatter()
num_format.format = '0.00000'
metrics_columns = [
    TableColumn(field='method', title='Source'),
    TableColumn(field='mean', title='Mean', formatter=num_format),
    TableColumn(field='sd', title='Std Dev', formatter=num_format),
    TableColumn(field='loc', title='Loc Param', formatter=num_format),
    TableColumn(field='scale', title='Scale Param', formatter=num_format),
    TableColumn(field='shape', title='Shape Param', formatter=num_format),
    TableColumn(field='aic', title='AIC', formatter=num_format),
]
metrics_table = DataTable(source=metrics_source, columns=metrics_columns, height=100)

# Text input widget
loc_val_input = TextInput(title='Specify loc value:', placeholder='none', value='')
loc_val_input.on_change('value', on_dist_change)

# Format app layout
widgets = widgetbox(metrics_table, dist_menu, data_source_menu, loc_val_input, width=400)
grid = gridplot([hist, cdf],
                [pp, qq],
                [widgets, None])
curdoc().add_root(grid)
