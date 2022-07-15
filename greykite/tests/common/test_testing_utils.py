import datetime

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.common.testing_utils import generate_test_changepoint_df


def test_generate_df_with_reg_for_tests():
    """Basic test of generate_df_with_reg_for_tests"""
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=20,
        train_frac=0.75,
        remove_extra_cols=True,
        mask_test_actuals=True)
    # test remove_extra_cols
    assert data["df"].shape == (20, 7)
    # test mask_test_actuals
    assert not data["train_df"][TIME_COL].isna().any()
    assert not data["train_df"][VALUE_COL].isna().any()
    assert not data["test_df"][TIME_COL].isna().any()
    assert data["test_df"][VALUE_COL].isna().all()


def test_generate_df_for_tests():
    """Test generate_df_for_tests"""
    data = generate_df_for_tests(
        freq="H",
        periods=24*10,
        train_start_date=datetime.datetime(2018, 1, 1),
        train_frac=0.9,
        remove_extra_cols=False)

    assert data["df"].shape == (24*10, 52)  # Contains time_feature columns
    assert not data["train_df"].isna().any().any()
    assert not data["test_df"][TIME_COL].isna().any().any()


def test_generate_test_changepoint_df():
    df = generate_test_changepoint_df(
        freq="D",
        periods=200,
        n_changepoints=3,
        signal_strength=1/5,
        err_std=1
    )
    assert df.shape == (200, 2)
    assert not df.isna().any().any()
    assert not df["ts"].isna().any().any()
