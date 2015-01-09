import kt.olddeepkt
import kt.data
import kt.train
from libs.logger import set_log_file


def test_deepkt_smoke():
    set_log_file("testlog.log")
    prepared_data = kt.data.prepare_fake_data()

    f_train, f_validate, train_idx, valid_idx, train_eval, valid_eval = (
        kt.olddeepkt.build_model(prepared_data, batch_size=1))

    best_validation_loss, best_epoch = (
        kt.train.train_model(f_train, f_validate, train_idx, valid_idx, train_eval, valid_eval,
                             n_epochs=200))

    assert best_validation_loss > 0.5
    assert best_epoch > 0
