import learntools.kt.olddeepkt
from learntools.kt.tests.kt_test import smoke_build_model


def test_deepkt_smoke():
    smoke_build_model(learntools.kt.olddeepkt.build_model)
