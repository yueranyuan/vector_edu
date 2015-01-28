from learntools.kt.kt import build_model
from learntools.kt.data import prepare_fake_data
from learntools.model.train import train_model
from learntools.libs.common_test_utils import use_logger_in_test


@use_logger_in_test
def smoke_build_model(build_model):
    prepared_data = prepare_fake_data()

    f_train, f_validate, train_idx, valid_idx, train_eval, valid_eval = (
        build_model(prepared_data, batch_size=1))

    best_validation_loss, best_epoch = (
        train_model(f_train, f_validate, train_idx, valid_idx, train_eval, valid_eval,
                    n_epochs=100))

    assert best_validation_loss > 0.5
    assert best_epoch > 0


def test_kt_smoke():
    smoke_build_model(build_model)
