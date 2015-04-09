"""
default args (10 trees), PosHi/PosLo, 100 trials. dataset is not filtered.
mean = 0.54928
stddev = 0.046899
12% under 0.50

default args with 1000 trees, PosHi/PosLo, 42 trials. dataset is not filtered.
mean = 0.61029
stddev = 0.044429
0% under 0.50
"""

from __future__ import print_function, unicode_literals, division

from learntools.emotiv.data import load_raw_data
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def cv_split(size, validation_percent=0.1):
  """Returns train_idx, valid_idx"""
  sigma = np.random.permutation(size)
  split = int(validation_percent * size)
  return (sigma[split:], sigma[:split])


def run():
  ds = load_raw_data('raw_data/raw.pickle', conds=['PositiveHighArousalPictures', 'PositiveLowArousalPictures'])
  for i in xrange(100):
    train_idx, valid_idx = cv_split(len(ds))
    X_train = map(np.ravel, ds.get_data('eeg')[train_idx])
    y_train = ds.get_data('condition')[train_idx]
    X_test = map(np.ravel, ds.get_data('eeg')[valid_idx])
    y_test = ds.get_data('condition')[valid_idx]
    rfc = RandomForestClassifier(n_estimators=1000)
    rfc.fit(X_train, y_train)
    print(rfc.score(X_test, y_test))


if __name__ == '__main__':
  run()
