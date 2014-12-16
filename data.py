import numpy
import gzip
import cPickle


class DataSet:
    def __init__(self):
        self.skills = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.cond = numpy.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1])


def gen_data(fname):
    inp = numpy.array([[0], [0], [0], [1], [1], [1]])
    target = numpy.array([1, 1, 1, 0, 0, 0])
    set_ = (inp, target)
    with gzip.open(fname, 'w') as f:
        cPickle.dump((set_, set_, set_), f)


def convert_from_xls(fname, outname):
    from loader import load
    data = load(fname)
    skill = data['stim'][:, None]
    subject = data['subject'][:, None]
    correct = data['cond']
    with gzip.open(outname, 'w') as f:
        cPickle.dump((skill, subject, correct), f)

if __name__ == "__main__":
    fname = 'data/task_data2.gz'
    convert_from_xls('raw_data/task_large.xls', fname)
