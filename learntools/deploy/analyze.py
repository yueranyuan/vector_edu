from __future__ import division
import re
import os
from itertools import ifilter, imap, chain, compress, tee, dropwhile
from collections import defaultdict
from datetime import datetime

from scipy import stats
import numpy as np

from learntools.libs import s3
from learntools.libs.utils import transpose, max_idx
from learntools.libs.plottools import grid_plot


# not using iterators in this code because log files are small
# and premature optimization will complicate the code
def parse_log(content):
    lines = content.split('\n')

    # parse arguments
    arg_lines = ifilter(lambda l: re.match(r'(.+=.+,)*(.+=.+)', l), lines[:10])
    arg_tokens = chain.from_iterable(
        imap(lambda l: re.split(r', ', l), arg_lines))
    args = dict(re.match(r'(.+)=(.+)', arg).groups() for arg in arg_tokens)

    # parse CV
    for l in lines[:10]:
        m = re.match(r'subjects (\d+) are held out', l)
        if m:
            print m.group(1)
            args['held_out'] = m.group(1)

    # parse history
    history_matches = imap(
        lambda l: re.match(r'epoch (\d+), validation error (.*)%', l),
        lines)
    history_matches = compress(*tee(history_matches, 2))  # filter out the 'Nones'
    history = [(int(h.group(1)), float(h.group(2))) for h in history_matches]

    # parse run time
    for l in lines[-3:]:
        m = re.match(r'Code ran for ran for (.+)m', l)
        if m:
            runtime = float(m.group(1))
            break

    return args, history, runtime


def modernize_args(key_name, args):
    # TODO: support further modernization
    return args


def analyze(bucket=None, subfolder=None, cache_dir=None, start_time=None,
            print_individual_trials=False):
    # load and prepare content
    if bucket and subfolder:
        contents = s3.load_all(bucket, subfolder, cache_dir=cache_dir)
    else:
        contents = {fn: None for fn in os.listdir(cache_dir)}
        for fn in contents.iterkeys():
            with open(os.path.join(cache_dir, fn), 'r') as f:
                contents[fn] = f.read()

    chronological = sorted(contents.iteritems(), key=lambda (name, _): name)
    if start_time:
        relevant_tokens = ['{:02}'.format(t) for t in start_time.timetuple()][:6]
        start_time_str = '_'.join(relevant_tokens)
        chronological = list(dropwhile(
            lambda (name, _): name[:len(start_time_str)] < start_time_str,
            chronological))

    # parse content
    num_logs = len(chronological)
    arg_all = defaultdict(lambda: np.zeros(num_logs))
    error_all = np.zeros(num_logs)
    if print_individual_trials:
        print '# Trials #'
    for i, (key_name, content) in enumerate(chronological):
        args, history, run_time = parse_log(content)

        # TODO: turn string args into enums rather than discarding
        for arg, v in args.iteritems():
            if arg == 'dataset_name':
                continue
            arg_all[arg][i] = v
        times, errors = transpose(history)
        best_epoch_idx, best_error = max_idx(errors)
        error_all[i] = best_error

        if print_individual_trials:
            best_epoch = times[best_epoch_idx]
            print ('{key_name} finished in {run_time}m and achieved {best_error}% ' +
                   'error on epoch #{best_epoch}').format(key_name=key_name,
                                                          best_error=best_error,
                                                          best_epoch=best_epoch,
                                                          run_time=run_time)

    # analyze each parameter/argument
    outcomes = [None] * len(arg_all)
    for i, (k, v) in enumerate(sorted(arg_all.iteritems())):
        # if all binary, then do t-test
        ones = np.equal(v, 1)
        zeros = np.equal(v, 0)
        if (ones | zeros).all():
            test_type = 't-test'
            _on, _off = error_all[ones], error_all[zeros]
            try:
                t, p = stats.ttest_ind(_on, _off)
            except:
                if len(_on) == 0 or len(_off) == 0:
                    p = 1
                else:
                    raise
            better = 'on' if np.mean(_on) > np.mean(_off) else 'off'
        else:
            test_type = 'pearson r'
            r, p = stats.pearsonr(v, error_all)
            better = 'high' if r > 0 else 'low'
        outcomes[i] = 'p={p} for {test_type} of {key}: {better} is better'.format(
            test_type=test_type, key=k, p=p, better=better)
    print "# Descriptive #"
    print 'n: {n} mean: {mean} variance: {variance}'.format(
        n=len(error_all), mean=np.mean(error_all), variance=np.var(error_all))
    print np.mean(error_all[np.equal(arg_all['model_type'], 0)])
    print "# Parameter Analysis #"
    for o in sorted(outcomes):
        print o

    plot_args = [arg for arg in arg_all.keys() if max(arg_all[arg]) != min(arg_all[arg])]
    grid_plot(xs=[arg_all[arg] for arg in plot_args],
              ys=error_all,
              x_labels=plot_args,
              y_labels='error')

if __name__ == '__main__':
    start_time = datetime(2015, 1, 05, 18, 00)
    analyze('cmu-data', 'vectoredu/results', cache_dir='results', start_time=start_time)
    # analyze(cache_dir='results', start_time=start_time)
