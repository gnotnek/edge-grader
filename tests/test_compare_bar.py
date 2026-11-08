from edge_grader.compare_bar import plan


def test_family_leaves_out_test_ua_for_scientsbank_trained_adapters():
    tests = plan()
    assert len(tests) == 30 == len(set(tests))
    assert not [t for t in tests if t[1] == "scientsbank" and t[3] == "test_ua"]
    assert ("e4b", "beetle-saf", "e4b-kaggle-t4-ft-rf-beetle-saf", "test_ua") in tests
    assert ("e2b", "scientsbank", "e2b-kaggle-t4-ft", "test_uq") in tests
