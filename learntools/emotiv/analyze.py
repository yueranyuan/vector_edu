import numpy as np
from scipy import stats

from learntools.libs.utils import mask_to_idx
from learntools.emotiv import prepare_data
from learntools.libs.plottools import grid_plot


def analyze_eeg_features(data, plot=True, silent=False):
    """ Analyzes the quality of features in an emotiv dataset using t-tests

    Args:
        data (Dataset): the dataset holding emotiv data
        plot (bool): whether you want to display a plot of the features

    Returns:
        (int[], int[], int[]): a tuple of the significant features (p < 0.05),
            marginal features (p < 0.10), and remaining features (p >= 0.10)
    """
    cond0_idxs, cond1_idxs = [mask_to_idx(np.equal(data.get_data('condition'), cond))
                              for cond in [0, 1]]

    def compute_feature_goodness(feature_idx):
        cond0_eeg_i = data.get_data('eeg')[cond0_idxs, feature_idx]
        cond1_eeg_i = data.get_data('eeg')[cond1_idxs, feature_idx]
        t, p = stats.ttest_ind(cond0_eeg_i, cond1_eeg_i)
        return p

    n_features = data['eeg'].width
    eeg_feature_ps = list(enumerate(map(compute_feature_goodness, xrange(n_features))))

    significant_features = {i for (i, p) in eeg_feature_ps if p < 0.05}
    marginal_features = {i for (i, p) in eeg_feature_ps if p < 0.10} - set(significant_features)
    all_features = {i for (i, p) in eeg_feature_ps}
    remaining_features = all_features - significant_features - marginal_features

    if not silent:
        print("Of the total {n} features".format(n=n_features))
        print('There are {n} significant features: {arr}'.format(n=len(significant_features),
                                                                 arr=significant_features))
        print('There are {n} marginal features: {arr}'.format(n=len(marginal_features),
                                                              arr=marginal_features))
        print('There are {n} remaining features: {arr}'.format(n=len(remaining_features),
                                                               arr=list(remaining_features)))

    if plot:
        grid_plot(xs=data.get_data('eeg').T,
                  ys=data.get_data('condition'),
                  x_labels=xrange(n_features),
                  y_labels='cond')

    return significant_features, marginal_features, remaining_features

if __name__ == '__main__':
    data = prepare_data(dataset_name='raw_data/indices_all.txt', conds=['EyesOpen', 'EyesClosed'])
    analyze_eeg_features(data)