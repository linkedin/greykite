import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LarsCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor

from greykite.algo.common.model_summary import ModelSummary
from greykite.common.python_utils import assert_equal


def test_model_summary():
    # sets up data
    np.random.seed(1)
    x = np.concatenate([np.ones([100, 1]), np.random.randn(100, 5)], axis=1)
    beta = np.array([1, 1, 1, 1, 0, 0])
    y = np.exp(np.matmul(x, beta))
    pred_cols = ["Intercept",
                 "ct1",
                 "sin1_toy_yearly",
                 "y_lag7",
                 "ct1:sin1_toy_yearly",
                 "C(Q('events_Chinese New Year'), levels=['', 'event'])[T.event]"]
    pred_category = {
        "intercept": ["Intercept"],
        "time_features": ["ct1", "ct1:sin1_toy_yearly"],
        "event_features": ["C(Q('events_Chinese New Year'), levels=['', 'event'])[T.event]"],
        "trend_features": ["ct1", "ct1:sin1_toy_yearly"],
        "seasonality_features": ["sin1_toy_yearly", "ct1:sin1_toy_yearly"],
        "lag_features": ["y_lag7"],
        "regressor_features": [],
        "interaction_features": ["ct1:sin1_toy_yearly"]
    }
    # fit algorithms
    fit_algorithm_dict = {
        "statsmodels_ols": sm.OLS,
        "statsmodels_wls": sm.WLS,
        "statsmodels_gls": sm.GLS,
        "statsmodels_glm": sm.GLM,
        "linear": sm.OLS,
        "elastic_net": ElasticNetCV,
        "ridge": RidgeCV,
        "lasso": LassoCV,
        "sgd": SGDRegressor,
        "lars": LarsCV,
        "lasso_lars": LassoLarsCV,
        "rf": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor}
    default_fit_algorithm_params = {
        "statsmodels_ols": dict(),
        "statsmodels_wls": dict(),
        "statsmodels_gls": dict(),
        "statsmodels_glm": dict(family=sm.families.Gamma()),
        "linear": dict(),
        "elastic_net": dict(cv=5),
        "ridge": dict(cv=5, alphas=np.logspace(-5, 5, 30)),
        "lasso": dict(cv=5),
        "sgd": dict(),
        "lars": dict(cv=5),
        "lasso_lars": dict(cv=5),
        "rf": dict(n_estimators=100),
        "gradient_boosting": dict()}
    for fit_algorithm in fit_algorithm_dict:
        params = default_fit_algorithm_params.get(fit_algorithm, {})
        if "statsmodels" in fit_algorithm or fit_algorithm == "linear":
            ml_model = fit_algorithm_dict[fit_algorithm](
                endog=y,
                exog=x,
                **params)
            ml_model = ml_model.fit()
            ml_model.coef_ = ml_model.params
            ml_model.intercept_ = 0
        else:
            ml_model = fit_algorithm_dict[fit_algorithm](**params)
            ml_model.fit(x, y)
        summary = ModelSummary(
            x=x,
            y=y,
            pred_cols=pred_cols,
            pred_category=pred_category,
            fit_algorithm=fit_algorithm,
            ml_model=ml_model)
        summary.__str__()
        if fit_algorithm == "linear":
            assert_equal(summary.info_dict["reg_df"], np.trace(x @ np.linalg.pinv(x.T @ x) @ x.T))
