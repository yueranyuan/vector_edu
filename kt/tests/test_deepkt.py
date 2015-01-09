import kt.olddeepkt
from kt.tests.kt_test import smoke_build_model


def test_deepkt_smoke():
    smoke_build_model(kt.olddeepkt.build_model)
