import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize


# TODO: split up calculate fitted so it calculates ls or mle, not both.
def perc_emp_filliben(indices):
    n_values = len(indices)
    # perc_emp = ((indices + 1) - 0.3175) / (n_values + 0.365)
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


def calc_fit_from_data(df, dist_type, loc, alg):
    # Calculate distribution parameters using mle, and calculate corresponding percentiles and quantiles

    if alg not in ['mle', 'ls']:
        print('Invalid algorithm parameter \'alg\' submitted to calc_fit_from_data()')  #TODO: make into try/catch

    if loc == '':
        params_mle = getattr(stats, dist_type).fit(df)
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
        data_subset = df[(df > lb) & (df < ub)]

        # Use scipy's fit method to get params_mle
        params_mle = getattr(stats, dist_type).fit(data_subset, floc=float(loc))

    dist_mle = freeze_dist(dist_type, params_mle)

    if alg == 'mle':
        return dist_mle

    # Calculate distribution parameters using ls, and calculate corresponding percentiles and quantiles
    perc_emp = perc_emp_filliben(np.linspace(1, len(df), len(df)))
    if not fixed_loc:
        ls_results = optimize.least_squares(min_fun, params_mle, args=(df, perc_emp, dist_type, loc), method='lm')
    else:
        params_no_loc = [x for x in params_mle if params_mle.index(x) != (len(params_mle) - 2)]
        ls_results = optimize.least_squares(min_fun, params_no_loc, args=(df, perc_emp, dist_type, loc), method='lm')

    if not fixed_loc:
        params_ls = ls_results.x
    else:
        params_ls = ls_results.x
        params_ls = np.append(params_ls, params_ls[-1])
        params_ls[-2] = float(loc)

    dist_ls = freeze_dist(dist_type, params_ls)

    return dist_ls
