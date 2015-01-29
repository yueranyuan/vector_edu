import scipy.io

from learntools.data import Dataset
from learntools.libs.utils import normalize_table

def load_file(filename):
    matlab_data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    struct = matlab_data['p']

    #TODO what do condition values of 0 and very high values mean? (also, is this actually
    #     the right source of condition data?)
    cond_labels = ['Zero', 'EyesOpen', 'EyesClosed', 'rest', 'NeutralLowArousalPictures',
                    'PositiveHighArousalPictures', 'NegativeLowArousalPictures',
                    'PositiveLowArousalPictures', 'CountBackwards', 'Ruminate',
                    'DrawStickFigure', 'start_positive', 'end_positive',
                    'start_negative', 'end_negative', 'High']
    channel_labels = struct.ChanLabels
    cond_indices = struct.OtherData[2]
    eeg_matrix = struct.EEG
    
    num_rows = eeg_matrix.shape[0]
    eeg_headers = [(x, Dataset.FLOAT) for x in channel_labels]
    headers = [('condition', Dataset.ENUM)] + eeg_headers
    ds = Dataset(headers, num_rows)

    for hi, hn in enumerate(channel_labels):
        for fi, fv in enumerate(eeg_matrix[:,hi]):
            ds.get_column(hn)[fi] = fv
        ds.get_column(hn).data = normalize_table(ds.get_column(hn))

    for i, cond in enumerate(cond_indices):
        try:
            ds.get_column('condition')[i] = cond_labels[cond]
        except IndexError:
            ds.get_column('condition')[i] = 'High'

    return ds
