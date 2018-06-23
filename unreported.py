import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fit


def sample_w_replacement(obs, prob, n_sim=10000):
    """
        Draw a set from obs with replacement
        prob is the probabilty that each entry could occur
    """
    ind = np.linspace(1, len(obs), len(obs), dtype=int) - 1  # all indices of obs (index starts at 0)
    ind_rand = np.random.choice(ind, size=n_sim, replace=True, p=prob)  # Randomly from ind using probabilities in prob.
    sample = obs[ind_rand]  # retrieve the actual values from obs

    return sample


def det_func_const(d_th, pod_thj):
    """ detection function constant """
    q = -np.log(1 - pod_thj) / d_th

    return q


def pod(reported_depth, det_threshold, pod_threshold):
    """ The probability of detection given the reported depth """
    q = det_func_const(det_threshold, pod_threshold)
    pod = 1 - np.exp(-q * reported_depth)

    return pod


def gen_unrep_sample(reported_size, rel_freq):
    """
        return a randomly generated sample of a representative population of non-detected defects
    """
    non_reported_sample = sample_w_replacement(reported_size, rel_freq)

    return non_reported_sample


def non_reported_params(reported_depth, det_threshold, pod_threshold, pofc, poi, use_const_pod=False):
    """
        get the likelihood of non-reported defects for each reported defect; the relative frequency and the total
        expected number of non reported defects
    """
    if use_const_pod:
        reported_pod = np.array([pod_threshold] * len(reported_depth))
    else:
        reported_pod = pod(reported_depth, det_threshold, pod_threshold)  # POD is a function of depth only
    likelihood = (1 - pofc) * (1 / (poi * reported_pod) - 1)  # (1 - POD) / POD
    n_unreported = np.sum(likelihood)
    rel_freq = likelihood / n_unreported  # normalise the likelihoods to get at relative probabilities

    return [n_unreported, likelihood, rel_freq]


if __name__ == '__main__':
    fit_dir = pathlib.Path(os.getcwd())
    data_dir = fit_dir / pathlib.Path('data')
    df = pd.read_csv(data_dir / pathlib.Path('data.csv')).dropna()
    df.columns = ['post', 'data']
    test_data = df['data']

    det_threshold = 1.
    pod_threshold = 0.9
    pofc = 0.
    poi = 1.
    n_unreported, likelihood, rel_freq = non_reported_params(df['data'], det_threshold, pod_threshold, pofc, poi)

    unrep_sample = gen_unrep_sample(df['data'], rel_freq)

    unrep_dist = fit.calc_fit_from_data(unrep_sample, 'lognorm')
    rep_dist = fit.calc_fit_from_data(df['data'], 'lognorm')

    sns.set()
    fig = plt.Figure()
    ax = plt.gca()

    n_bins = max(int(np.rint(np.sqrt(len(df['data'])))), 10)
    bins = np.linspace(min(df['data']), max(df['data']), n_bins + 1)
    data_heights, _, _ = plt.hist(df['data'], color='gray', alpha=0.3, bins=bins, density=True, label='Reported')
    unrep_heights, _, _ = plt.hist(unrep_sample, color='green', alpha=0.3, bins=bins, density=True, label='Unreported')
    ax.set_ylim([0, 1.1 * max(max(data_heights), max(unrep_heights))])

    x_line = np.linspace(df['data'].min() * .9, df['data'].max() * 1.1, 500)
    plt.plot(x_line, unrep_dist.pdf(x_line), color='green', label='Unreported Fit')

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 2, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    plt.show()
