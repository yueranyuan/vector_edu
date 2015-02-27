from learntools.kt.tests.test_kt import smoke_kt_model
import pytest
slow = pytest.mark.slow

@slow
def test_lrkt_smoke():
    from learntools.kt.lrkt import LRKT
    smoke_kt_model(LRKT)
