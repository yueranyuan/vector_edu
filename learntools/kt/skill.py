import numpy as np

from learntools.data import gen_word_matrix
from learntools.libs.logger import log_me


class MatrixTypes(object):
    RANDOM = 0
    UNIGRAM_BIGRAM = 1


@log_me('...generating skill matrix')
def gen_skill_matrix(skill_ids, skill_enum_pairs, matrix_type=MatrixTypes.RANDOM, skill_vector_len=100,
                     **kwargs):
    if matrix_type == MatrixTypes.RANDOM:
        skill_matrix = np.random.rand(len(skill_enum_pairs), skill_vector_len)
    elif matrix_type == MatrixTypes.UNIGRAM_BIGRAM:
        skill_matrix = gen_word_matrix(skill_ids,
                                       skill_enum_pairs,
                                       vector_length=skill_vector_len)
    else:
        raise ValueError
    return skill_matrix