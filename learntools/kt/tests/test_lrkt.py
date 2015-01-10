from learntools.kt.lrkt import build_model
from learntools.kt.tests.test_kt import smoke_build_model


def test_lrkt_smoke():
    smoke_build_model(build_model)
