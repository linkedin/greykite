import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.python_utils import assert_equal
from greykite.sklearn.estimator.null_model import DummyEstimator
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator
from greykite.sklearn.transform.column_selector import ColumnSelector
from greykite.sklearn.transform.null_transformer import NullTransformer
from greykite.sklearn.transform.pandas_feature_union import PandasFeatureUnion
from greykite.sklearn.transform.zscore_outlier_transformer import ZscoreOutlierTransformer


@pytest.fixture
def X():
    """dataset for test cases"""
    size = 20
    return pd.DataFrame({
        TIME_COL: pd.date_range(start="2018-01-01", periods=size, freq="H"),
        ACTUAL_COL: np.random.normal(scale=10, size=size),
        VALUE_COL: np.random.normal(scale=10, size=size)
    })


@pytest.fixture
def fs():
    """feature transformation pipeline for test cases"""
    return PandasFeatureUnion([
        ("date", Pipeline([
            ("select_date", ColumnSelector([TIME_COL]))  # leaves time column unmodified
        ])),
        ("response", Pipeline([  # applies outlier and null transformation to value column
            ("select_val", ColumnSelector([VALUE_COL])),
            ("outlier", ZscoreOutlierTransformer()),
            ("null", NullTransformer())
        ]))
    ])


def test_feature_union(X):
    """Tests PandasFeatureUnion on simple projection
    Inspired by sklearn/tests/test_pipeline.py"""
    # basic sanity check for feature union
    select_value = ColumnSelector(column_names=[VALUE_COL])
    select_actual = ColumnSelector(column_names=[ACTUAL_COL])
    select_time = ColumnSelector(column_names=[TIME_COL])
    fs = PandasFeatureUnion([("select_value", select_value),
                             ("select_actual", select_actual),
                             ("select_time", select_time),
                             ("select_time_again", select_time)])

    fs.fit(X)
    X_transformed = fs.transform(X)

    assert X_transformed.shape == (X.shape[0], 4)
    # note that columns are selected in the order specified. There is no column renaming by default
    assert np.all(X_transformed.columns.values == np.array([VALUE_COL, ACTUAL_COL, TIME_COL, TIME_COL]))
    assert X_transformed.equals(pd.concat([
        X[[VALUE_COL]],
        X[[ACTUAL_COL]],
        X[[TIME_COL]],
        X[[TIME_COL]],
    ], axis=1))


def test_transformer_union(X, fs):
    """Tests PandasFeatureUnion on a pipeline of transformers, with custom parameters"""
    # sets parameters and fits model
    z_cutoff = 2.0
    fs.set_params(response__outlier__z_cutoff=z_cutoff)
    fs.fit(X)
    X_transformed = fs.transform(X)

    # checks shape
    assert X_transformed.shape == (X.shape[0], 2)
    assert list(X_transformed.columns) == [TIME_COL, VALUE_COL]

    # checks output result
    X_after_column_select = ColumnSelector([VALUE_COL]).fit_transform(X)
    X_after_z_score = ZscoreOutlierTransformer(z_cutoff=z_cutoff).fit_transform(X_after_column_select)
    X_after_null = NullTransformer().fit_transform(X_after_z_score)

    assert_equal(X_transformed[TIME_COL], X[TIME_COL])
    assert_equal(X_transformed[VALUE_COL], X_after_null[VALUE_COL])


def test_pipeline_union(X, fs):
    """Tests PandasFeatureUnion on a pipeline of transformers and estimator, and shows
     that null model extracted from estimator in pipeline is equivalent to null model trained
     directly"""
    model_estimator = Pipeline([
        ("input", fs),
        ("estimator", SimpleSilverkiteEstimator(score_func=mean_squared_error,
                                                coverage=0.80,
                                                null_model_params={"strategy": "mean"}))
    ])

    # fits pipeline with estimator, and extract dummy null model
    z_cutoff = 2.0
    model_estimator.set_params(input__response__outlier__z_cutoff=z_cutoff)
    model_estimator.fit(X)
    output_estimator_null = model_estimator.steps[-1][-1].null_model.predict(X)

    # fits pipeline with dummy estimator
    model_dummy = Pipeline([
        ("input", fs),
        ("dummy", DummyEstimator(score_func=mean_squared_error, strategy="mean"))
    ])
    model_dummy.fit(X)
    output_dummy = model_dummy.predict(X)

    # fits dummy estimator by hand, without Pipeline
    X_after_column_select = ColumnSelector([VALUE_COL]).fit_transform(X)
    X_after_z_score = ZscoreOutlierTransformer(z_cutoff=z_cutoff).fit_transform(X_after_column_select)
    X_after_null = NullTransformer().fit_transform(X_after_z_score)
    X_after_union = pd.concat([X[TIME_COL], X_after_null], axis=1)
    model_hand = DummyEstimator(strategy="mean")
    model_hand.fit(X_after_union)
    output_by_hand = model_hand.predict(X_after_union)

    assert output_estimator_null.equals(output_by_hand)
    assert output_dummy.equals(output_by_hand)
