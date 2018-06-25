import os
import pathlib
import pandas as pd
import numpy as np
import scipy.stats as stats
import fit
from bokeh.io import curdoc
from bokeh.layouts import widgetbox, gridplot
from bokeh.models import ColumnDataSource, Select, DataTable, TableColumn, NumberFormatter, TextInput
from bokeh.plotting import figure


# TODO: handle cases where least squares fails.
# TODO: is it a problem to return df from calculate_fitted_data? Changing df inside the function changes it outside too.
# TODO: when using manual input of loc parameter, handle case where dataset is empty because outside of ub/lb
# TODO: add option to save plot.
# TODO: allow for selecting data source
def load_data(data_source):
    data_path = pathlib.Path(os.getcwd()) / pathlib.Path('data\\' + data_source)
    df = pd.read_csv(data_path).dropna()
    df.columns = ['data']
    # df.columns = ['post', 'data']

    # Calculate empirical percentiles for ordered (ranked) data.
    df = df.sort_values(by='data').reset_index(drop=True)
    df['perc_emp'] = fit.perc_emp_filliben(df.index.get_values())

    # Calculate distribution parameters for default (Normal) distribution.
    dist_type = dist_menu.value
    loc = loc_val_input.value
    dist_mle = fit.calc_fit_from_data(df['data'], dist_type, loc, 'mle')
    dist_ls = fit.calc_fit_from_data(df['data'], dist_type, loc, 'ls')

    return df, dist_mle, dist_ls


def on_change_datasource(attr, old, new):
    df, dist_mle, dist_ls = load_data(data_source_menu.value)
    data_source.data['data'] = df['data']
    data_source.data['perc_emp'] = df['perc_emp']
    data_source.data['perc_mle'] = dist_mle.cdf(df['data'])
    data_source.data['quant_mle'] = dist_mle.ppf(df['perc_emp'])
    data_source.data['perc_ls'] = dist_ls.cdf(df['data'])
    data_source.data['quant_ls'] = dist_ls.ppf(df['perc_emp'])

    data_fit.data['x_mle'] = dist_mle.ppf(data_fit.data['cdf_y'])
    data_fit.data['pdf_mle'] = dist_mle.pdf(data_fit.data['x_mle'])
    data_source.data['perc_mle'] = dist_mle.cdf(data_source.data['data'])
    data_source.data['quant_mle'] = dist_mle.ppf(data_source.data['perc_emp'])

    data_fit.data['x_ls'] = dist_ls.ppf(data_fit.data['cdf_y'])
    data_fit.data['pdf_ls'] = dist_ls.pdf(data_fit.data['x_ls'])
    data_source.data['perc_ls'] = dist_ls.cdf(data_source.data['data'])
    data_source.data['quant_ls'] = dist_ls.ppf(data_source.data['perc_emp'])

    metrics_source.data['mean'] = (df['data'].mean(), dist_mle.mean(), dist_ls.mean())
    metrics_source.data['sd'] = (df['data'].std(), dist_mle.std(), dist_ls.std())
    metrics_source.data['scale'] = (np.nan, dist_mle.args[-1], dist_ls.args[-1])
    metrics_source.data['loc'] = (np.nan, dist_mle.args[-2], dist_ls.args[-2])
    fixed_loc = loc_val_input.value != ''
    k = fit.calc_k(getattr(stats, dist_type), fixed_loc)
    if fixed_loc:
        mle_likelihoods = dist_mle.pdf(data_source.data['data'][data_source.data['data'] > float(loc_val_input.value)])
        ls_likelihoods = dist_ls.pdf(data_source.data['data'][data_source.data['data'] > float(loc_val_input.value)])
    else:
        mle_likelihoods = dist_mle.pdf(data_source.data['data'])
        ls_likelihoods = dist_ls.pdf(data_source.data['data'])
    metrics_source.data['aic'] = (np.nan, fit.calc_aic(mle_likelihoods, k), fit.calc_aic(ls_likelihoods, k))
    if dist_mle.args[:-2]:
        metrics_source.data['shape'] = (np.nan, dist_mle.args[:-2], dist_ls.args[:-2])
    else:
        metrics_source.data['shape'] = (np.nan, np.nan, np.nan)

    bin_heights, bin_edges = np.histogram(df['data'], normed=True, bins='auto')
    hist.x_range.start = min(bin_edges) - 0.1 * bin_range
    hist.x_range.end = max(bin_edges) + 0.1 * bin_range
    hist.y_range.end = max(bin_heights) * 1.1
    hist_source.data['bin_heights'] = bin_heights
    hist_source.data['bin_mids'] = pd.Series(bin_edges).rolling(window=2).mean().dropna().reset_index(drop=True)
    hist_source.data['bin_widths'] = pd.Series(bin_edges).diff().dropna().reset_index(drop=True)

    qq_line_source.data['x'] = (0, max(df['data']))
    qq_line_source.data['y'] = (0, max(df['data']))


def on_dist_change(attr, old, new):
    dist_type = dist_menu.value
    loc = loc_val_input.value
    if loc == '':
        fixed_loc = False
    else:
        fixed_loc = True

    dist_mle = fit.calc_fit_from_data(data_source.data['data'], dist_type, loc, 'mle')
    dist_ls = fit.calc_fit_from_data(data_source.data['data'], dist_type, loc, 'ls')

    data_fit.data['x_mle'] = dist_mle.ppf(data_fit.data['cdf_y'])
    data_fit.data['pdf_mle'] = dist_mle.pdf(data_fit.data['x_mle'])
    data_source.data['perc_mle'] = dist_mle.cdf(data_source.data['data'])
    data_source.data['quant_mle'] = dist_mle.ppf(data_source.data['perc_emp'])

    data_fit.data['x_ls'] = dist_ls.ppf(data_fit.data['cdf_y'])
    data_fit.data['pdf_ls'] = dist_ls.pdf(data_fit.data['x_ls'])
    data_source.data['perc_ls'] = dist_ls.cdf(data_source.data['data'])
    data_source.data['quant_ls'] = dist_ls.ppf(data_source.data['perc_emp'])

    metrics_source.data['mean'] = (df['data'].mean(), dist_mle.mean(), dist_ls.mean())
    metrics_source.data['sd'] = (df['data'].std(), dist_mle.std(), dist_ls.std())
    metrics_source.data['scale'] = (np.nan, dist_mle.args[-1], dist_ls.args[-1])
    metrics_source.data['loc'] = (np.nan, dist_mle.args[-2], dist_ls.args[-2])
    k = fit.calc_k(getattr(stats, dist_type), fixed_loc)
    if fixed_loc:
        mle_likelihoods = dist_mle.pdf(data_source.data['data'][data_source.data['data'] > float(loc)])
        ls_likelihoods = dist_ls.pdf(data_source.data['data'][data_source.data['data'] > float(loc)])
    else:
        mle_likelihoods = dist_mle.pdf(data_source.data['data'])
        ls_likelihoods = dist_ls.pdf(data_source.data['data'])
    metrics_source.data['aic'] = (np.nan, fit.calc_aic(mle_likelihoods, k), fit.calc_aic(ls_likelihoods, k))
    if dist_mle.args[:-2]:
        metrics_source.data['shape'] = (np.nan, dist_mle.args[:-2], dist_ls.args[:-2])
    else:
        metrics_source.data['shape'] = (np.nan, np.nan, np.nan)


if ('bk_script' in __name__) or (__name__ == '__main__'):
    # Get raw data.
    fit_dir = pathlib.Path(os.getcwd())
    data_dir = fit_dir / pathlib.Path('data')
    df = pd.read_csv(data_dir / pathlib.Path('data.csv')).dropna()
    # df.columns = ['post', 'data']
    df.columns = ['data']

    # Calculate empirical percentiles for ordered (ranked) data.
    df = df.sort_values(by='data').reset_index(drop=True)
    df['perc_emp'] = fit.perc_emp_filliben(df.index.get_values())

    # Calculate distribution parameters for default (Normal) distribution.
    dist_type = 'norm'
    dist_mle = fit.calc_fit_from_data(df['data'], dist_type, '', 'mle')
    df['perc_mle'] = dist_mle.cdf(df['data'])
    df['quant_mle'] = dist_mle.ppf(df['perc_emp'])

    dist_ls = fit.calc_fit_from_data(df['data'], dist_type, '', 'ls')
    df['perc_ls'] = dist_ls.cdf(df['data'])
    df['quant_ls'] = dist_ls.ppf(df['perc_emp'])

    data_source = ColumnDataSource(df)

    df_fit = pd.DataFrame(data=np.linspace(0.000001, 0.999999, 1000), columns=['cdf_y'])
    df_fit['x_mle'] = dist_mle.ppf(df_fit['cdf_y'])
    df_fit['pdf_mle'] = dist_mle.pdf(df_fit['x_mle'])
    df_fit['x_ls'] = dist_ls.ppf(df_fit['cdf_y'])
    df_fit['pdf_ls'] = dist_ls.pdf(df_fit['x_ls'])
    data_fit = ColumnDataSource(df_fit)

    metrics_source = ColumnDataSource(
        dict(
            method=('Source Data', 'Maximum Likelihood', 'Least Squares (Quantiles)'),
            mean=(df['data'].mean(), dist_mle.mean(), dist_ls.mean()),
            sd=(df['data'].std(), dist_mle.std(), dist_ls.std()),
            scale=(np.nan, dist_mle.args[-1], dist_ls.args[-1]),
            loc=(np.nan, dist_mle.args[-2], dist_ls.args[-2]),
            shape=(np.nan, np.nan, np.nan),  # This is because default dist, norm, has no shape values
            aic=(np.nan, fit.calc_aic(dist_mle.pdf(df['data']), 2), fit.calc_aic(dist_ls.pdf(df['data']), 2)),
        )
    )

    # %% Calculate datapoints to represent the assumed distribution.
    demo_range = np.linspace(0.000001, 0.999999, 1000)
    demo_domain = dist_mle.ppf(demo_range)
    cdf_source = ColumnDataSource(data={
        'x': demo_domain,
        'y': demo_range
    })

    # Define tools for Bokeh plots
    tools = 'pan,box_zoom,reset,save'

    # Histogram
    bin_heights, bin_edges = np.histogram(df['data'], normed=True, bins='auto')
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
    hist.line(x='x_mle', y='pdf_mle', color='green', source=data_fit, line_width=3, legend='MLE')
    hist.line(x='x_ls', y='pdf_ls', color='blue', source=data_fit, line_width=3, legend='LS')

    cdf = figure(plot_width=400, plot_height=300, tools=tools, title='CDF',
                 x_range=[min(bin_edges) - 0.1 * bin_range, max(bin_edges) + 0.1 * bin_range])
    cdf.circle('data', 'perc_emp', color='gray', source=data_source, alpha=0.5)
    cdf.line('x_mle', 'cdf_y', color='green', source=data_fit, line_width=3)
    cdf.line('x_ls', 'cdf_y', color='blue', source=data_fit, line_width=3)

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
    qq_line_source = ColumnDataSource(dict(x=(0, max(df['data'])), y=(0, max(df['data']))))
    qq.line(x='x', y='y', color='gray', source=qq_line_source)
    # qq.line(x=(0, max(df['data'])), y=(0, max(df['data'])), color='gray')

    # Distribution dropdown widget
    options = [x for x in dir(stats) if isinstance(getattr(stats, x), stats.rv_continuous)]
    dist_menu = Select(options=options, value='norm', title='Distribution:')
    dist_menu.on_change('value', on_dist_change)

    # Data source dropdown widget
    files = [x for x in os.listdir('data') if x.split('.')[-1] == 'csv']
    data_source_menu = Select(options=files, value='data.csv', title='Source from \'data\' directory:')
    data_source_menu.on_change('value', on_change_datasource)

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
