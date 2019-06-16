import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt


def perc_emp_filliben(indices):
    n_values = len(indices)
    perc_emp = ((indices) - 0.3175) / (n_values + 0.365)
    perc_emp[-1] = 0.5 ** (1 / n_values)
    perc_emp[0] = 1 - perc_emp[-1]
    return np.array(perc_emp)


def calc_k(dist, fixed_loc):
    k = 2  # Count loc and scale parameters
    if dist.shapes:
        k += len(dist.shapes.split(','))  # Add in shape parameters if they exist.
    if fixed_loc:
        k -= 1  # Remove the loc parameter if it is fixed.
    return k


def calc_aic(likelihoods, k):
    likelihoods = likelihoods[likelihoods>0]
    return 2*k - 2*sum(np.log(likelihoods))
    pass


def min_fun(x, data, perc_emp, dist_str, loc):
    if loc == '':
        dist_ls = freeze_dist(dist_str, x)
    else:
        x_with_loc = np.append(x, x[-1])
        x_with_loc[-2] = float(loc)
        dist_ls = freeze_dist(dist_str, x_with_loc)

    quant_emp = dist_ls.ppf(perc_emp)

    return np.array(data - quant_emp)


def freeze_dist(dist_str, params):
    dist = getattr(stats, dist_str)
    return dist(*params)


def calc_fit_from_data(data, dist_type, loc='', alg='ls'):
    # Calculate distribution parameters using mle, and calculate corresponding percentiles and quantiles

    if alg not in ['mle', 'ls']:
        print('Invalid algorithm parameter \'alg\' submitted to calc_fit_from_data()')  #TODO: make into try/catch

    if loc == '':
        params_mle = getattr(stats, dist_type).fit(data)
        fixed_loc = False
    else:
        fixed_loc = True
        general_dist = getattr(stats, dist_type)
        shapes = general_dist.shapes
        if shapes is None:
            n_shapes = 0
        else:
            n_shapes = len(shapes.split(','))

        # Remove data points outside of the lower/upper bounds of dist_type, a requirement for scipy's fit method.
        general_params = n_shapes*[1] + [float(loc), 1]
        lb, ub = general_dist(*general_params).ppf(0), general_dist(*general_params).ppf(1)
        data_subset = data[(data > lb) & (data < ub)]

        # Use scipy's fit method to get params_mle
        params_mle = getattr(stats, dist_type).fit(data_subset, floc=float(loc))

    dist_mle = freeze_dist(dist_type, params_mle)

    if alg == 'mle':
        k = calc_k(dist_mle.dist, loc)
        aic = calc_aic(dist_mle.pdf(data), k)
        return dist_mle, aic

    # Calculate distribution parameters using ls, and calculate corresponding percentiles and quantiles
    perc_emp = perc_emp_filliben(np.linspace(1, len(data), len(data)))
    data = np.sort(data)  # Data must be sorted to correspond to correct Filliben percentiles.
    if not fixed_loc:
        ls_results = optimize.least_squares(min_fun, params_mle, args=(data, perc_emp, dist_type, loc), method='lm')
    else:
        params_no_loc = [x for x in params_mle if params_mle.index(x) != (len(params_mle) - 2)]
        ls_results = optimize.least_squares(min_fun, params_no_loc, args=(data, perc_emp, dist_type, loc), method='lm')

    if not fixed_loc:
        params_ls = ls_results.x
    else:
        params_ls = ls_results.x
        params_ls = np.append(params_ls, params_ls[-1])
        params_ls[-2] = float(loc)

    dist_ls = freeze_dist(dist_type, params_ls)

    k = calc_k(dist_ls.dist, loc)
    aic = calc_aic(dist_ls.pdf(data), k)
    return dist_ls, aic


def make_fourplot(data, dist, title='Title goes here', fig_save_path=None):
    data = np.array(np.sort(data))
    perc_emp = perc_emp_filliben(np.linspace(1, len(data), len(data)))
    x_fit = dist.ppf(np.linspace(1e-3, (1 - 1e-3), 500))

    fourplot, ((hist, cdf), (pp, qq)) = plt.subplots(2, 2)
    fourplot.suptitle(title)

    hist.plot(x_fit, dist.pdf(x_fit), color='green', linewidth=2.)
    hist_heights, bins = np.histogram(data)
    unrep_heights, _, _ = hist.hist(data, color='gray', bins=bins, density=True, label='Data', alpha=0.7)
    hist.set_ylim([0, 1.1 * unrep_heights.max()])
    hist.set_xlabel('Data')
    hist.set_ylabel('Probability Density')

    cdf.scatter(data, perc_emp, color='gray', s=1., alpha=0.7)
    cdf.plot(x_fit, dist.cdf(x_fit), color='green', linewidth=2.)
    cdf.set_xlabel('Data')
    cdf.set_ylabel('CDF')

    pp.scatter(dist.cdf(data), perc_emp, color='gray', s=1., alpha=0.7)
    pp.plot((0, 1), (0,1), color='black', linewidth=1.)
    pp.set_xlabel('Theoretical Probability')
    pp.set_ylabel('Empirical Probability')

    qq.scatter(dist.ppf(perc_emp), data, color='gray', s=1., alpha=0.7)
    qq.plot((min(data),max(data)), (min(data),max(data)), color='black', linewidth=1.)
    qq.set_xlabel('Theoretical Quantile')
    qq.set_ylabel('Empirical Quantile')

    plt.tight_layout(rect=[0,0,1,0.95])

    if fig_save_path:
        plt.savefig(fig_save_path + '\\' + title + '.png')

    return fourplot
