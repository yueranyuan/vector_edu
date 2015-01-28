from learntools.kt.data import prepare_data, cv_split
from learntools.libs.common_test_utils import use_logger_in_test
import pytest
slow = pytest.mark.slow

@use_logger_in_test
def smoke_kt_model(model_cls):
    data = prepare_data(dataset_name='data/data5.gz', top_n=14)
    train_idx, valid_idx = cv_split(data, fold_index=0)

    model = model_cls((data, train_idx, valid_idx))

    # set an absurdly high improve threshold so that the training will terminate in 20 epochs
    # no matter what
    best_validation_loss, best_epoch = model.train_full(n_epochs=20,
                                                        improvement_threshold=100.,
                                                        patience=0)

    assert best_validation_loss > 0.6
    assert best_epoch > 0

@slow
def test_kt_smoke():
    from learntools.kt.kt import KT
    smoke_kt_model(KT)
