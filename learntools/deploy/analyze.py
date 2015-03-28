from __future__ import division
import re
import os
from itertools import ifilter, imap, chain, compress, tee, dropwhile, izip, islice
from collections import defaultdict
from datetime import datetime, timedelta
import json

from scipy import stats
import numpy as np

from learntools.libs import s3
from learntools.libs.utils import transpose, max_idx, exception_safe_map
from learntools.libs.plottools import grid_plot

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class BadLogFileException(BaseException):
    pass


class NotArgLineException(BaseException):
    pass


def basic_arg_parser(text):
    """parse line of arguments in the old log format.

    Args:
        text (str): the content of a argument line from a log file.

    Returns:
        (list<str, value>): list of (name, value) pairs representing the name of the argument and the value that
            argument was given.
    Raises:
        NotArgLineException: if the input is not a valid argument line.

    Examples:
        >>>basic_arg_parser('x=1, y=2')
        [('x', '1'), ('y', '2')]
    """
    if not re.match(r'(.+=.+,)*(.+=.+)', text):
        raise NotArgLineException()
    arg_tokens = re.split(r', ', text)
    arg_tokens = ifilter(lambda arg: re.search(r'=', arg), arg_tokens)  # remove ill-formed tokens
    arg_pairs = [re.match(r'(.+)=(.+)', arg).groups() for arg in arg_tokens]
    return arg_pairs


def json_arg_parser(text):
    """parse line of arguments in the json log format.

    Args:
        text (str): the content of a argument line from a log file.

    Returns:
        (list<str, value>): list of (name, value) pairs representing the name of the argument and the value that
            argument was given.
    Raises:
        NotArgLineException: if the input is not a valid argument line.

    Examples:
        >>>json_arg_parser('Arguments to f1: {x: 1, y: 2}')
        [('x', 1), ('y', 2)]
    """
    match = re.match(r'Arguments to (.*): ({.*})', text)
    if not match:
        raise NotArgLineException()
    arg_dict = json.loads(match.group(2))
    # filter out everything that aren't numbers because we don't know how to deal with those yet
    # TODO: figure out how to deal with non-number arg values
    return arg_dict.iteritems()
    return filter(lambda(k, v): isinstance(v, (int, long, float)), arg_dict.iteritems())


def parse_log(content, arg_parser=json_arg_parser):
    """ Parse important information from log files.

    These log files are small so we are making the logic a little simpler by loading all
    the content into memory at once rather than using an iostream.
    Args:
        content (string): the string content of the file

    Returns:
        args (dict<string, value>): a dictionary of function arguments of the program that
            created the log and the value those arguments were set to.
        history (list<(int, float)>): a list of tuples of time (in epoch index) and corresponding
            classification loss
        runtime (float): runtime of program in seconds
    """
    lines = content.split('\n')

    # Part 1: parse arguments
    arg_pair_lists = exception_safe_map(arg_parser, lines[:20], exception=NotArgLineException)
    args = dict(chain.from_iterable(arg_pair_lists))

    # parse CV
    for l in lines[:10]:
        m = re.match(r'subjects (\d+) are held out', l)
        if m:
            args['held_out'] = m.group(1)

    # Part 2: parse history
    history_matches = imap(
        lambda l: re.match(r'epoch (\d+), validation accuracy (.*)%', l),
        lines)
    history_matches = compress(*tee(history_matches, 2))  # filter out the 'Nones'
    history = [(int(h.group(1)), float(h.group(2))) for h in history_matches]

    # Part 3: parse run time
    runtime = None
    for l in lines[-3:]:
        m = re.match(r'Code ran for ran for (.+)m', l)
        if m:
            runtime = float(m.group(1))
            break

    if runtime is None or len(history) == 0 or len(args) == 0:
        raise BadLogFileException('file was not formatted properly')

    return args, history, runtime


def analyze_s3_default(**kwargs):
    """analyze log files from the default s3 folder"""
    return analyze(bucket='cmu-data', subfolder='vectoredu/results', **kwargs)


def analyze_recent(seconds=0, minutes=0, hours=0, days=0, delta=None, **kwargs):
    """analyze log files that were created recently.

    Analyze log files that were created within a certain time window of the present.

    Args:
        seconds (int): seconds that elapsed since first log to be analyzed
        minutes (int): minutes that elapsed since first log to be analyzed
        hours (int): hours that elapsed since first log to be analyzed
        days (int): days that elapsed since first log to be analyzed
        delta (timedelta): how much time has elapsed since the first log to be analyzed
        **kwargs: other inputs to analyze (see analyze())
    """
    if delta is None:
        delta = timedelta(days=days, seconds=seconds, minutes=minutes, hours=hours)
    start_time = datetime.now() - delta
    return analyze(start_time=start_time, **kwargs)


def analyze(local_dir=None, bucket=None, subfolder=None, start_time=None,
            print_individual_trials=False, create_plot=True, most_recent_n=None, earliest_n=None):
    """analyze log files

    Args:
        local_dir (str, optional): the directory that contains the log files we want to analyze.
            If we are loading data from s3, the s3 data is downloaded to this folder.
        bucket (str, optional): s3 bucket to load data from. Both bucket and subfolder are
            required to load log files from s3.
        subfolder (str, optional): s3 subfolder to load data from.  Both bucket and
            subfolder are required to load log files from s3.
        start_time (datetime|str, optional): only analyze logs that were written after start_time.
            if start_time is str, it must be of the format %Y-%m-%d %H:%M:%S
        print_individual_trials
    """
    # load log content
    if bucket and subfolder:
        contents = s3.load_all(bucket, subfolder, cache_dir=local_dir)
    else:
        contents = {}
        for fn in os.listdir(local_dir):
            if not re.match(r'.*\.log$', fn):
                continue
            with open(os.path.join(local_dir, fn), 'r') as f:
                contents[fn] = f.read()

    # sort logs in chronological order
    chronological = sorted(contents.iteritems(), key=lambda (name, _): name)

    # don't analyze any logs before start_time
    if start_time:
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, TIME_FORMAT)
        relevant_tokens = ['{:02}'.format(t) for t in start_time.timetuple()][:6]
        start_time_str = '_'.join(relevant_tokens)

        def _too_early(file_name):
            return file_name[:len(start_time_str)] < start_time_str
        chronological = list(dropwhile(lambda(_fn, _): _too_early(_fn), chronological))

    # only analyze the most recent n
    if most_recent_n is not None:
        most_recent_n = min(len(chronological), most_recent_n)
        chronological = chronological[-most_recent_n:]

    # only analyze the earliest n
    if earliest_n is not None:
        earliest_n = min(len(chronological), earliest_n)
        chronological = chronological[:earliest_n]

    # parse log content
    parsed_content = exception_safe_map(lambda (k, v): (k, parse_log(v)),
                                        chronological,
                                        exception=BadLogFileException)

    # parse cond_types
    COND_TYPES = [
        ['EyesClosed', 'EyesOpen'],
        ["PositiveLowArousalPictures", "NegativeLowArousalPictures"],
        ["PositiveHighArousalPictures", "NegativeHighArousalPictures"],
        ["PositiveHighArousalPictures", "PositiveLowArousalPictures"],
        ["NegativeHighArousalPictures", "NegativeLowArousalPictures"]]
    for key_name, (args, history, run_time) in parsed_content:
        args['conds'] = COND_TYPES.index(args['conds'])
    # parsed_content = filter(lambda (key_name, (args, history, run_time)): args["conds"] == 3, parsed_content)

    # parse content of each individual log file and fill a data store of
    # the arguments and best_errors of all runs
    num_logs = len(parsed_content)
    arg_all = defaultdict(lambda: np.zeros(num_logs))
    error_all = np.zeros(num_logs)
    if print_individual_trials:
        print '# Trials #'

    for log_i, (key_name, (args, history, run_time)) in enumerate(parsed_content):
        # TODO: turn string args into enums rather than discarding
        # store argument values for this log
        for arg, v in args.iteritems():
            try:
                float(v)
            except:
                continue
            arg_all[arg][log_i] = v

        # store best error for this log
        times, errors = transpose(history)
        ERROR_WINDOW = 7
        effective_error_window = min(len(errors), ERROR_WINDOW)
        smoothed_errors = [sum(window) / len(window) for window in
                           izip(*[islice(errors, i, None) for i in xrange(effective_error_window)])]
        best_epoch_idx, best_error = max_idx(smoothed_errors)
        error_all[log_i] = best_error

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
        # if arg is binary, then do t-test
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
        else:  # if not binary, do correlation
            test_type = 'pearson r'
            r, p = stats.pearsonr(v, error_all)
            better = 'high' if r > 0 else 'low'
        outcomes[i] = 'p={p:.5f} for {test_type} of {key}: {better} is better'.format(
            test_type=test_type, key=k, p=p, better=better)

    # print analysis
    print "# Descriptive #"
    print 'n: {n} mean: {mean} variance: {variance}'.format(
        n=len(error_all), mean=np.mean(error_all), variance=np.var(error_all))
    print "# Parameter Analysis #"
    for o in sorted(outcomes):
        print o

    # print cond breakdown
    for cond_i, cond in enumerate(COND_TYPES):
        cond_mask = np.equal(arg_all['conds'], cond_i)
        average_error = np.average(list(compress(error_all, cond_mask)))
        print("average error: {err} for {cond}".format(cond=cond, err=average_error))

    # plot arguments
    if create_plot:
        plot_args = [arg for arg in arg_all.keys() if max(arg_all[arg]) != min(arg_all[arg])]
        if len(plot_args) == 0:
            print 'No arguments to plot'
        else:
            grid_plot(xs=[arg_all[arg] for arg in plot_args],
                      ys=error_all,
                      x_labels=plot_args,
                      y_labels='error')

if __name__ == '__main__':
    # start_time = datetime(2015, 1, 05, 18, 00)
    # analyze(bucket='cmu-data', subfolder='vectoredu/results', cache_dir='results', start_time=start_time)
    # analyze(cache_dir='results', start_time=start_time)
    # analyze_recent(days=2, local_dir='.')
    analyze(start_time=datetime(2015, 3, 8, 3, 00), local_dir='.', most_recent_n=10)
