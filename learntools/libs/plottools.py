from __future__ import division
from math import sqrt, ceil
from itertools import izip

import matplotlib.pyplot as plt


def grid_plot(xs, ys, x_labels=None, y_labels=None):
    """Create a big grid of many subplots

    Ideal for looking at many features at the same time. The grid is meant to
    fit as many subplots on the screen as possible

    Args:
        xs (number[][] or number[]): the data to be plotted on the subplot x-axes.
            If 1-dimensional, the same xs is used for all subplots.
        ys (number[][] or number[]): the data to be plotted on the subplot y-axes.
            If 1-dimensional, the same ys is used for all subplots.
        x_labels (str[] or str): the label(s) of the subplot x-axes.
            If type is str, the same x label is used for all subplots.
        y_labels (str[] or str): the label(s) of the subplot y-axes.
            If type is str, the same y label is used for all subplots.
    """
    # convert xs and ys to 2-dimensional arrays, broadcasting from 1-dimensional array
    # if necessary. This is not terribly space inefficient since we're just copying references
    n_xs = len(xs) if hasattr(xs[0], '__iter__') else None
    n_ys = len(ys) if hasattr(ys[0], '__iter__') else None
    if n_xs is not None and n_ys is not None and n_xs != n_ys:
        raise Exception('length of xs and length of ys do not match')
    n_subplots = n_xs or n_ys
    if n_xs is None:
        xs = [xs] * n_subplots
    if n_ys is None:
        ys = [ys] * n_subplots

    # convert single labels to list of the same labels.
    x_labels = [x_labels] * n_subplots if isinstance(x_labels, str) else x_labels
    y_labels = [y_labels] * n_subplots if isinstance(y_labels, str) else y_labels
    if len(x_labels) != n_subplots:
        raise Exception('length of x_labels array does not match the number of subplots requested')
    if len(y_labels) != n_subplots:
        raise Exception('length of y_labels array does not match the number of subplots requested')

    # figure out the dimensions of the grid
    h = int(sqrt(n_subplots))
    w = ceil(n_subplots / h)

    # draw the plots
    for i, (x, y, x_label, y_label) in enumerate(izip(xs, ys, x_labels, y_labels)):
        # plot the data
        plt.subplot(h, w, i + 1)
        plt.scatter(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # set the subplot boundaries
        x_min, x_max = min(x), max(x)
        x_margin = (x_max - x_min) * 0.1
        plt.xlim(x_min - x_margin, x_max + x_margin)
        y_min, y_max = min(y), max(y)
        y_margin = (y_max - y_min) * 0.1
        plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.show()