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
    ind = np.linspace(1, len(obs), len(obs), dtype=int) - 1  # All indices of obs (index starts at 0)
    np.random.seed(0)   # Fix random seed.
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
        return a randomly generated sample of a representative population of undetected defects
    """
    reported_size = np.array(reported_size)
    rel_freq = np.array(rel_freq)
    unreported_sample = sample_w_replacement(reported_size, rel_freq)

    return unreported_sample


def unreported_params(reported_depth, det_threshold, pod_threshold, pofc, poi, use_const_pod=False):
    """
        get the likelihood of unreported defects for each reported defect; the relative frequency and the total
        expected number of unreported defects
    """
    if use_const_pod:
        reported_pod = np.array([pod_threshold] * len(reported_depth))
    else:
        reported_pod = pod(reported_depth, det_threshold, pod_threshold)  # POD is a function of depth only
    likelihood = (1 - pofc) * (1 / (poi * reported_pod) - 1)  # (1 - POD) / POD
    n_unreported = np.sum(likelihood)
    rel_freq = likelihood / n_unreported  # normalise the likelihoods to get at relative probabilities

    return [n_unreported, likelihood, rel_freq]


def rep_unrep_plot(rep_sample, unrep_sample, rep_dist, unrep_dist, title='Title goes here', fig_save_path=None):
    """
        plots histograms of the reported and unreported data along with fitted distributions to each.
    """
    fig, ax = plt.subplots(1, 1)

    heights_data, bins_data = np.histogram(rep_sample, normed=True)
    heights_unrep, bins_unrep = np.histogram(unrep_sample, normed=True)
    ax.hist([rep_sample, unrep_sample], color=['gray', 'green'], bins=bins_data, density=True,
             label=['Reported', 'Unreported'], alpha=0.7)
    ax.set_ylim([0, 1.1 * max(heights_data.max(), heights_unrep.max())])

    x_line = np.linspace(min(rep_sample) * .95, max(rep_sample) * 1.05, 500)
    ax.plot(x_line, rep_dist.pdf(x_line), color='gray', label='Reported Fit', dashes=[3, 3])
    ax.plot(x_line, unrep_dist.pdf(x_line), color='green', label='Unreported Fit', dashes=[3, 3])

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 3, 1]
    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right',
               bbox_to_anchor=(0.9, 0.9))

    fig.suptitle(title)
    fig.tight_layout(rect=[0,0,1,0.95])

    if fig_save_path:
        fig.savefig(pl.Path(fig_save_path) / pl.Path(title + '.png'))

    return fig


if __name__ == '__main__':
    fit_dir = pathlib.Path(os.getcwd())
    data_dir = fit_dir / pathlib.Path('data')
    df = pd.read_csv(data_dir / pathlib.Path('data.csv')).dropna()
    df.columns = ['data']
    test_data = df['data']

    det_threshold = 1.
    pod_threshold = 0.9
    pofc = 0.
    poi = 1.
    n_unreported, likelihood, rel_freq = unreported_params(df['data'], det_threshold, pod_threshold, pofc, poi)

    unrep_sample = gen_unrep_sample(df['data'], rel_freq)

    unrep_dist = fit.calc_fit_from_data(unrep_sample, 'lognorm')
    rep_dist = fit.calc_fit_from_data(df['data'], 'lognorm')

    _ = rep_unrep_plot(df['data'], unrep_sample, rep_dist, unrep_dist)
    _ = fit.make_fourplot(unrep_sample, unrep_dist, title='Unreported dist')
    _ = fit.make_fourplot(df['data'], rep_dist, title='Reported dist')

    sns.set()
    plt.show()
