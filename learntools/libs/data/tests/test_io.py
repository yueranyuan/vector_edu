from itertools import izip, islice

from learntools.libs.data.io import Column, TimeColumn, EnumColumn, Dataset

strtimes = ["2013-10-15 09:15:51.480000", "2013-10-15 09:15:55.480000", "2013-10-15 09:15:55.480000"]
timestamps = [138184295148, 138184295548, 138184295548]
enumstr = ['jan', 'feb', 'jan']
enumint = [0, 1, 0]
numstr = ['0', '1', '2']
nums = [0, 1, 2]


def test_column():
    col = Column('example_column')
    col[0] = numstr[0]
    assert col[0] == nums[0]
    col[1:3] = numstr[1:3]
    assert all(col[:3] == nums)


def test_timecolumn():
    col = TimeColumn('example_column')
    strtimes = ["2013-10-15 09:15:51.480000", "2013-10-15 09:15:55.480000", "2013-10-15 09:15:55.480000"]
    col[0] = strtimes[0]
    col[1:3] = strtimes[1:3]
    assert all(col[:3] == timestamps)
    col.mode = TimeColumn.ORIGINAL
    assert col[:3] == strtimes


def test_enumcolumn():
    col = EnumColumn('example_column')
    enumstr = ['jan', 'feb', 'jan']
    enumint = [0, 1, 0]
    col[0] = enumstr[0]
    col[1:3] = enumstr[1:3]
    assert all(col[:3] == enumint)
    col.mode = EnumColumn.ORIGINAL
    assert col[:3] == enumstr


def test_dataset():
    dataset = Dataset([('int', Dataset.INT), ('enum', Dataset.ENUM), ('time', Dataset.TIME)], n_rows=len(nums))
    for i, row in enumerate(izip(numstr, enumstr, strtimes)):
        dataset[i] = row

    for d, row in izip(islice(dataset, None, 3), izip(nums, enumint, timestamps)):
        assert tuple(d) == row
    dataset.mode = Dataset.ORIGINAL
    for d, row in izip(islice(dataset, None, 3), izip(numstr, enumstr, strtimes)):
        assert tuple(map(str, d)) == row
