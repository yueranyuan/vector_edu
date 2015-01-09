import kt.lrkt
from kt.tests.kt_test import smoke_build_model


def test_lrkt_smoke():
    smoke_build_model(kt.lrkt.build_model)
