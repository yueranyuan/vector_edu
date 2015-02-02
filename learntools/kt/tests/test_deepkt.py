from learntools.kt.deepkt import DeepKT
from learntools.kt.tests.test_kt import smoke_kt_model
import pytest
slow = pytest.mark.slow

@slow
def test_deepkt_smoke():
    smoke_kt_model(DeepKT)
