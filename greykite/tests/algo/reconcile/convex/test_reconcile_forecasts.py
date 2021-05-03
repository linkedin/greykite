import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import sqrtm
from sklearn.exceptions import NotFittedError

from greykite.algo.reconcile.convex.reconcile_forecasts import ReconcileAdditiveForecasts
from greykite.algo.reconcile.convex.reconcile_forecasts import apply_method_defaults
from greykite.algo.reconcile.convex.reconcile_forecasts import get_fit_params
from greykite.algo.reconcile.convex.reconcile_forecasts import get_weight_matrix
from greykite.algo.reconcile.hierarchical_relationship import HierarchicalRelationship
from greykite.common.evaluation import mean_absolute_percent_error
from greykite.common.evaluation import median_absolute_percent_error
from greykite.common.evaluation import root_mean_squared_error
from greykite.common.python_utils import assert_equal
from greykite.common.python_utils import reorder_columns


@pytest.fixture
def data():
    np.random.seed(9208)
    levels = [[3]]
    tree = HierarchicalRelationship(levels)
    m = tree.num_nodes
    n = 150  # number of observations per timeseries
    # Each row is a timeseries, generally increasing.
    Ya = 100 * np.random.uniform(0.0, 2.0, size=(m, n)).cumsum().reshape((m, n))
    Ya = tree.bottom_up_transform @ Ya  # makes values consistent
    scale = Ya.mean()
    Yf = Ya + np.random.rand(m, n)*scale/20 + scale / 2  # adds forecast error and bias to increase the constraint violation
    forecasts = pd.DataFrame(Yf.T, columns=["root", "c1", "c2", "c3"])
    actuals = pd.DataFrame(Ya.T, columns=["root", "c1", "c2", "c3"])
    # Reorders columns to test `order_dict`
    forecasts = forecasts[["c1", "c3", "root", "c2"]]
    actuals = actuals[["c1", "c3", "root", "c2"]]
    order_dict = {
        "root": 0,  # hierarchy order
        "c1": 1,
        "c2": 2,
        "c3": 2}
    return {
        "forecasts": forecasts,
        "actuals": actuals,
        "tree": tree,
        "levels": levels,
        "order_dict": order_dict,
        "Yf": Yf,
        "Ya": Ya,
        "m": m,
    }


def test_get_weight_matrix():
    """Tests get_weight_matrix"""
    # weights is None
    wmat = get_weight_matrix(
        weights=None,
        n_forecasts=3,
        name="weight_bias",
        weight_auto=None)
    assert_equal(np.eye(3), wmat)

    # weights is 'auto', weight_auto is None
    wmat = get_weight_matrix(
        weights="auto",
        n_forecasts=3,
        name="weight_bias",
        weight_auto=None)
    assert_equal(np.eye(3), wmat)

    # weights is 'auto', weight_auto is not None
    wmat = get_weight_matrix(
        weights="auto",
        n_forecasts=3,
        name="weight_bias",
        weight_auto=np.ones((3, 3)))
    assert_equal(np.ones((3, 3)), wmat)

    # weights is a list
    wmat = get_weight_matrix(
        weights=[1, 2, 3],
        n_forecasts=3,
        name="weight_bias",
        weight_auto=None)
    assert_equal(np.diag([1, 2, 3]), wmat)

    # weights is an array
    wmat = get_weight_matrix(
        weights=np.array([1, 2, 3]),
        n_forecasts=3,
        name="weight_bias",
        weight_auto=None)
    assert_equal(np.diag([1, 2, 3]), wmat)

    # exception
    with pytest.raises(ValueError, match="Expected square matrix with size 10, but `weight_bias` "
                                         "has weight matrix with shape \\(3, 3\\)"):
        get_weight_matrix(
            weights=np.array([1, 2, 3]),
            n_forecasts=10,  # doesn't match len(weights)
            name="weight_bias",
            weight_auto=None)


def test_get_fit_params():
    """Tests get_fit_params"""
    params = get_fit_params(method=None)
    assert params == {}

    params = get_fit_params(method="bottom_up")
    assert params == {}

    params = get_fit_params(method="ols")
    assert params["unbiased"]
    assert params["lam_var"] == 1.0
    assert params["covariance"] == "identity"

    params = get_fit_params(method="mint_sample")
    assert params["unbiased"]
    assert params["lam_var"] == 1.0
    assert params["covariance"] == "sample"

    with pytest.raises(ValueError, match="`method` 'unknown' is not recognized. "
                                         "Must be one of 'bottom_up', 'ols', 'mint_sample' or None"):
        get_fit_params(method="unknown")


def test_apply_method_defaults():
    """Tests apply_method_defaults"""
    def dummy_fit_func(
            self,
            forecasts,
            actuals,
            **params):
        return {
            "self": self,
            "forecasts": forecasts,
            "actuals": actuals,
            "params": params,

        }
    # checks `dummy_fit_func`
    result = dummy_fit_func(
        self="self",
        forecasts="forecasts",
        actuals="actuals",
        method="mint_sample")
    assert_equal(result, {
        "self": "self",
        "forecasts": "forecasts",
        "actuals": "actuals",
        "params": {"method": "mint_sample"},
    })

    # `method` is not None
    new_fit_func = apply_method_defaults(dummy_fit_func)
    params = {
        "method": "mint_sample",
        "covariance": "identity",
        "verbose": True}
    result = new_fit_func(
        self="self",
        forecasts="forecasts",
        actuals="actuals",
        **params)
    params.update(get_fit_params("mint_sample"))
    assert_equal(result, {
        "self": "self",
        "forecasts": "forecasts",
        "actuals": "actuals",
        "params": params,
    })
    assert params["covariance"] == "sample"  # overwritten by the default

    # `method` is None
    new_fit_func = apply_method_defaults(dummy_fit_func)
    params = {
        "method": None,
        "covariance": "identity",
        "verbose": True}
    result = new_fit_func(
        self="self",
        forecasts="forecasts",
        actuals="actuals",
        **params)
    assert_equal(result, {
        "self": "self",
        "forecasts": "forecasts",
        "actuals": "actuals",
        "params": params,
    })

    # `method` is not provided
    new_fit_func = apply_method_defaults(dummy_fit_func)
    params = {
        "covariance": "identity",
        "verbose": True}
    result = new_fit_func(
        self="self",
        forecasts="forecasts",
        actuals="actuals",
        **params)
    assert_equal(result, {
        "self": "self",
        "forecasts": "forecasts",
        "actuals": "actuals",
        "params": params,
    })


def test_raf_init():
    """Tests ReconcileAdditiveForecasts init"""
    raf = ReconcileAdditiveForecasts()
    assert raf.forecasts is None
    assert raf.covariance is None
    assert raf.covariance is None
    assert raf.constraint_violation is None
    assert raf.lam_adj == 0.0
    assert not raf.unbiased


def test_raf_form_constraints():
    """Tests ReconcileAdditiveForecasts _form_constraints"""
    levels = [[2]]
    tree = HierarchicalRelationship(levels)
    m = tree.num_nodes
    transform_variable = cp.Variable((m, m))
    # actuals that satisfy the constraints, each row is a timeseries
    Ya = np.array([
        [2.0, 3.0, 5.0, 8.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 1.0, 2.0, 4.0],
    ])

    # `Ya` is None, unbiased=False, no upper/lower bound, constraint_matrix
    raf = ReconcileAdditiveForecasts()
    raf.constraint_matrix = tree.constraint_matrix
    raf.unbiased = False
    raf.lower_bound = None
    raf.upper_bound = None
    constraints = raf._form_constraints(
        transform_variable=transform_variable,
        Ya=None)
    assert len(constraints) == 1
    assert [c.is_dcp() for c in constraints]
    constraint = constraints[0]
    assert constraint.atoms()[0] == cp.atoms.affine.binary_operators.MulExpression
    assert_equal(constraint.constants()[0].value, raf.constraint_matrix)
    assert_equal(constraint.constants()[1].value, np.array([0]))

    # `Ya` is not None, unbiased=True, constraint_matrix
    raf = ReconcileAdditiveForecasts()
    raf.constraint_matrix = tree.constraint_matrix
    raf.unbiased = True
    raf.lower_bound = None
    raf.upper_bound = None
    constraints = raf._form_constraints(
        transform_variable=transform_variable,
        Ya=Ya)
    assert len(constraints) == 2
    assert [c.is_dcp() for c in constraints]
    assert_equal(constraints[1].constants()[0].value, Ya)
    assert_equal(constraints[1].constants()[1].value, Ya)

    # `Ya` is not None, unbiased=True, upper/lower bound, tree
    raf = ReconcileAdditiveForecasts()
    raf.unbiased = True
    raf.tree = tree
    raf.constraint_matrix = np.zeros_like(tree.constraint_matrix)
    raf.lower_bound = -1.0
    raf.upper_bound = 1.0
    constraints = raf._form_constraints(
        transform_variable=transform_variable,
        Ya=Ya)
    assert len(constraints) == 4
    assert_equal(constraints[0].constants()[0].value, np.array(raf.lower_bound))
    assert_equal(constraints[1].constants()[0].value, np.array(raf.upper_bound))
    assert_equal(constraints[2].constants()[0].value, np.array(raf.tree.constraint_matrix))  # prefers raf.tree.constraint_matrix
    assert_equal(constraints[3].constants()[0].value, np.array(raf.tree.sum_matrix))
    assert_equal(constraints[3].constants()[1].value, np.array(raf.tree.sum_matrix))

    with pytest.warns(UserWarning, match="Actuals do not satisfy the constraints!"):
        Ya[0] += 1.0
        raf._form_constraints(
            transform_variable=transform_variable,
            Ya=Ya)

    with pytest.raises(ValueError, match="`Ya` must be provided if `unbiased` and `tree` is None."):
        raf.tree = None
        raf._form_constraints(
            transform_variable=transform_variable,
            Ya=None)


def test_raf_form_objective():
    """Tests ReconcileAdditiveForecasts _form_objective"""
    # variables
    levels = [[3], [2, 3, 3]]
    tree = HierarchicalRelationship(levels)
    m = tree.num_nodes
    n = 150
    transform_variable = cp.Variable((m, m))
    Ya = 100 * np.random.rand(m, n)
    Ya = tree.bottom_up_transform @ Ya  # makes values consistent
    Yf = Ya + np.random.rand(m, n)  # adds forecast error
    transform_matrix = np.eye(m) + np.random.rand(m, m)
    # derived variables
    weight_auto = np.diag(np.nanmedian(np.abs(Ya - Yf) / np.abs(Ya), axis=1))
    weight_auto = weight_auto * np.sqrt(m) / np.linalg.norm(weight_auto)
    cov = np.cov(Ya - Yf)

    # covariance is None, "auto" weight_bias
    raf = ReconcileAdditiveForecasts()
    raf.weight_bias = "auto"
    obj = raf._form_objective(
        Yf=Yf,
        Ya=Ya,
        transform_variable=transform_variable,
        covariance=None)
    assert_equal(raf.objective_weights, {
        "weight_adj": np.eye(m),
        "weight_bias": weight_auto,
        "weight_coef": np.ones((m, m)),
        "weight_train": np.eye(m),
        "weight_var": np.eye(m),
        "covariance": None})
    assert raf.objective_fn(None) is None
    assert raf.objective_fn(transform_matrix)["var"] == 0.0  # variance term is 0.0
    assert obj.is_dcp()
    assert len(obj.constants()) == 25  # lams, weights, other constants

    # "identity" covariance, "auto" weight_coef, weight_adj
    raf = ReconcileAdditiveForecasts()
    raf.weight_adj = "auto"
    raf.weight_coef = "auto"
    obj = raf._form_objective(
        Yf=Yf,
        Ya=Ya,
        transform_variable=transform_variable,
        covariance="identity")
    assert_equal(raf.objective_weights, {
        "weight_adj": weight_auto,
        "weight_bias": np.eye(m),
        "weight_coef": np.ones((m, m)) - np.eye(m),
        "weight_train": np.eye(m),
        "weight_var": np.eye(m),
        "covariance": np.eye(m)})
    assert raf.objective_fn(transform_matrix) == {
        "adj": 0.0,  # all the self.lam_* are 0.0
        "bias": 0.0,
        "coef": 0.0,
        "train": 0.0,
        "var": 0.0,
        "total": 0.0}
    assert len(obj.constants()) == 28  # now includes covariance term

    # "sample" covariance
    raf = ReconcileAdditiveForecasts()
    raf.lam_adj = 1.1
    raf.lam_bias = 2.2
    raf.lam_coef = 3.3
    raf.lam_train = 4.4
    raf.lam_var = 5.5
    obj = raf._form_objective(
        Yf=Yf,
        Ya=Ya,
        transform_variable=transform_variable,
        covariance="sample")
    assert obj.is_dcp()
    assert len(obj.constants()) == 28
    assert obj.constants()[0].value == raf.lam_adj
    assert obj.constants()[1].value == raf.lam_bias
    assert obj.constants()[2].value == raf.lam_coef
    assert obj.constants()[3].value == raf.lam_train
    assert obj.constants()[4].value == raf.lam_var
    assert_equal(obj.constants()[5].value, raf.objective_weights["weight_adj"])
    assert_equal(obj.constants()[6].value, Yf)
    assert_equal(obj.constants()[7].value, Yf)
    assert_equal(raf.objective_weights, {
        "weight_adj": np.eye(m),
        "weight_bias": np.eye(m),
        "weight_coef": np.ones((m, m)),
        "weight_train": np.eye(m),
        "weight_var": np.eye(m),
        "covariance": cov})
    # Tests objective_fn value
    expected_adj = raf.lam_adj * np.linalg.norm(raf.objective_weights["weight_adj"] @ (transform_matrix @ Yf - Yf))**2 / np.size(Yf)
    expected_bias = raf.lam_bias * np.linalg.norm(raf.objective_weights["weight_bias"] @ (transform_matrix @ Ya - Ya))**2 / np.size(Ya)
    expected_coef = raf.lam_coef * np.linalg.norm(np.multiply(raf.objective_weights["weight_coef"], transform_matrix - np.eye(m)))**2 / (m**2)
    expected_train = raf.lam_train * np.linalg.norm(raf.objective_weights["weight_train"] @ (transform_matrix @ Yf - Ya))**2 / np.size(Yf)
    expected_var = raf.lam_var * np.linalg.norm(raf.objective_weights["weight_var"] @ transform_matrix @ sqrtm(cov))**2 / m
    assert_equal(raf.objective_fn(transform_matrix), {
        "adj": expected_adj,
        "bias": expected_bias,
        "coef": expected_coef,
        "train": expected_train,
        "var": expected_var,
        "total": expected_adj + expected_bias + expected_coef + expected_train + expected_var})
    assert_equal(raf.objective_fn(transform_matrix, forecast_matrix=Yf*2, actual_matrix=Ya*2), {
        "adj": 4*expected_adj,
        "bias": 4*expected_bias,
        "coef": expected_coef,
        "train": 4*expected_train,
        "var": expected_var,
        "total": 4*expected_adj + 4*expected_bias + expected_coef + 4*expected_train + expected_var})
    # Tests equivalence of variance formula with trace
    var_alternative = raf.lam_var * np.trace(transform_matrix @ cov @ transform_matrix.T) / m
    assert_equal(var_alternative, raf.objective_fn(transform_matrix)["var"])
    # Tests objective value
    constraints = [
        transform_variable >= -1.0,
        transform_variable <= 1.0,
        tree.constraint_matrix @ transform_variable == 0]
    prob = cp.Problem(obj, constraints=constraints)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # sometimes there is a warning about solver accuracy
        prob.solve(feastol=1e-5, reltol=1e-8, abstol=1e-5, verbose=True)
    assert_equal(prob.objective.value, raf.objective_fn(transform_variable.value)["total"])

    with pytest.raises(ValueError, match="`covariance` not recognized. Provide a valid string in \\['identity', 'sample'\\] or a matrix."):
        raf = ReconcileAdditiveForecasts()
        raf._form_objective(
            Yf=Yf,
            Ya=Ya,
            transform_variable=transform_variable,
            covariance="unknown-covariance")


def test_raf_fit(data):
    """Tests ReconcileAdditiveForecasts fit"""
    forecasts = data["forecasts"]
    actuals = data["actuals"]
    tree = data["tree"]
    levels = data["levels"]
    order_dict = data["order_dict"]
    Yf = data["Yf"]
    Ya = data["Ya"]
    m = data["m"]

    raf = ReconcileAdditiveForecasts()
    with pytest.raises(ValueError, match="Either `constraint_matrix` or `levels` must be provided."):
        raf.fit(
            forecasts=forecasts,
            actuals=actuals,
            method="bottom_up")

    with pytest.raises(ValueError, match="Must provide `levels` if `method='bottom_up'`."):
        raf.fit(
            forecasts=forecasts,
            actuals=actuals,
            constraint_matrix=tree.constraint_matrix,
            method="bottom_up")

    with pytest.raises(ValueError, match="Only one of `constraint_matrix` and `levels` can be provided."):
        raf.fit(
            forecasts=forecasts,
            actuals=actuals,
            levels=levels,
            constraint_matrix=tree.constraint_matrix,
            method="bottom_up")

    with pytest.raises(ValueError, match="`lower_bound` 1.5 should not be greater than `upper_bound` 1.0"):
        raf.fit(
            forecasts=forecasts,
            actuals=actuals,
            levels=levels,
            method="bottom_up",
            lower_bound=1.5,
            upper_bound=1.0)

    with pytest.raises(ValueError, match="Forecast shape \\(20, 4\\) does not match actuals shape \\(150, 4\\)"):
        raf.fit(
            forecasts=forecasts[:20],
            actuals=actuals,
            levels=levels,
            method="bottom_up")

    with pytest.raises(
            ValueError,
            match="The number of forecasts 4 does not match "
                  "the number of columns in `constraint_matrix` 3. "
                  "Make sure `levels` or `constraint_matrix` matches the data."):
        raf.fit(
            forecasts=forecasts,
            actuals=actuals,
            constraint_matrix=np.eye(3))

    with pytest.raises(
            ValueError,
            match="The number of forecasts 4 does not match "
                  "the number of columns in `constraint_matrix` 8. "
                  "Make sure `levels` or `constraint_matrix` matches the data."):
        raf.fit(
            forecasts=forecasts,
            actuals=actuals,
            levels=[[2], [3, 2]])

    # Tests order_dict, scaling, objective function, method
    raf = ReconcileAdditiveForecasts()
    raf.fit(
        forecasts=forecasts,
        actuals=actuals,
        levels=levels,
        order_dict=order_dict,
        method="mint_sample",
        lower_bound=-1.5,
        upper_bound=1.5)
    assert raf.lower_bound is None  # overridden by "mint_sample"
    assert raf.upper_bound is None  # overridden by "mint_sample"
    assert_equal(raf.forecasts, reorder_columns(forecasts, order_dict=order_dict))
    residuals = Ya / Ya.mean() - Yf / Ya.mean()
    covariance = np.cov(residuals)
    assert_equal(raf.objective_weights["covariance"], covariance)
    assert raf.prob is not None
    assert raf.is_optimization_solution
    assert_equal(raf.transform_variable.value, raf.transform_matrix)
    assert np.linalg.norm(raf.constraint_matrix @ raf.transform_matrix) <= 1e-5  # satisfies constraints
    assert_equal(raf.objective_fn_val["total"], raf.prob.objective.value)
    assert_equal(raf.objective_fn_val["total"], raf.objective_fn_val["var"])  # "mint_sample" has only variance in the objective

    with pytest.warns(
            UserWarning,
            match="Variance of residuals is underestimated if the estimator is biased."):
        raf = ReconcileAdditiveForecasts()
        raf.fit(
            forecasts=forecasts,
            actuals=actuals,
            levels=levels,
            order_dict=order_dict,
            unbiased=False,
            covariance="sample",
            lam_var=1.0)
        assert raf.transform_matrix is not None

    # Tests custom covariance matrix, unbiased=Tue, all terms in objective
    covariance = 20 * np.random.randn(m, m)
    raf = ReconcileAdditiveForecasts()
    raf.fit(
        forecasts=forecasts,
        actuals=actuals,
        levels=levels,
        order_dict=order_dict,
        lower_bound=-1.5,
        upper_bound=1.5,
        unbiased=True,
        lam_adj=1.0,
        lam_bias=1.0,
        lam_coef=1.0,
        lam_train=1.0,
        lam_var=1.0,
        covariance=covariance,
        weight_bias="auto",
        weight_coef="auto",
        weight_train=list(range(m)),
        weight_var=list(range(m, 0, -1)),
        reltol=1e-7,  # solver params
        abstol=1e-7,
        feastol=1e-7)
    scaled_covariance = covariance / Ya.mean()**2
    assert_equal(raf.objective_weights["covariance"], scaled_covariance)
    assert_equal(raf.objective_weights["weight_train"], np.diag(list(range(m))))
    assert_equal(raf.objective_weights["weight_var"], np.diag(list(range(m, 0, -1))))
    assert raf.prob is not None
    assert_equal(raf.transform_variable.value, raf.transform_matrix)
    # satisfies constraints
    assert np.linalg.norm(raf.constraint_matrix @ raf.transform_matrix) <= 1e-5
    assert_equal(raf.transform_matrix @ Ya, Ya)
    assert raf.objective_fn_val["bias"] < 1e-10

    # Tests bottom up forecast
    raf = ReconcileAdditiveForecasts()
    raf.fit(
        forecasts=forecasts,
        actuals=actuals,
        levels=levels,
        order_dict=order_dict,
        method="bottom_up")
    assert_equal(raf.transform_matrix, raf.tree.bottom_up_transform)
    assert raf.prob is None
    assert raf.transform_variable is None

    # No solution
    with pytest.warns(UserWarning, match="Failed to find a solution. Falling back to bottom-up method"):
        raf = ReconcileAdditiveForecasts()
        raf.fit(
            forecasts=forecasts,
            actuals=actuals,
            levels=levels,
            order_dict=order_dict,
            unbiased=True,
            lam_adj=1.0,
            lower_bound=3.0,
            upper_bound=3.0)
        assert_equal(raf.transform_matrix, raf.tree.bottom_up_transform)

    # With constraint_matrix
    constraint_matrix = np.array([
        [-1., 2., 1., 0.],
        [0., -1., 1., 1.],
    ])
    with pytest.warns(
            UserWarning,
            match="Actuals do not satisfy the constraints!"):
        raf = ReconcileAdditiveForecasts()
        raf.fit(
            forecasts=forecasts,
            actuals=actuals,
            constraint_matrix=constraint_matrix,
            order_dict=order_dict,
            unbiased=False,
            lam_adj=1.0,
            lam_bias=1.0)
        assert_equal(raf.constraint_matrix, constraint_matrix)
        assert np.linalg.norm(raf.constraint_matrix @ raf.transform_matrix) < 1e-5

    with pytest.warns(
            UserWarning,
            match="Failed to find a solution. Try setting CVXPY solver parameters, changing the "
                  "constraints, or changing the objective weights"):
        raf = ReconcileAdditiveForecasts()
        raf.fit(
            forecasts=forecasts,
            actuals=actuals,
            constraint_matrix=constraint_matrix,
            order_dict=order_dict,
            unbiased=True,
            lam_adj=1.0,
            lower_bound=3.0,
            upper_bound=3.0)
        assert raf.transform_matrix is None
        assert raf.objective_fn_val is None
        assert raf.prob.status != cp.OPTIMAL
        assert not raf.is_optimization_solution


def test_raf_transform(data):
    """Tests ReconcileAdditiveForecasts transform and fit_transform"""
    forecasts = data["forecasts"]
    actuals = data["actuals"]
    levels = data["levels"]
    order_dict = data["order_dict"]
    raf = ReconcileAdditiveForecasts()

    with pytest.raises(NotFittedError, match="Must call `fit` first."):
        raf.transform()

    raf.fit(
        forecasts=forecasts,
        actuals=actuals,
        levels=levels,
        order_dict=order_dict,
        method="mint_sample")

    # transforms training data
    raf.transform()
    reordered_forecasts = reorder_columns(forecasts, order_dict=order_dict)
    assert raf.adjusted_forecasts is not None
    assert_equal(raf.adjusted_forecasts.columns, reordered_forecasts.columns)  # columns are returned according to `order_dict` order
    assert raf.adjusted_forecasts.shape == forecasts.shape
    assert np.linalg.norm(raf.constraint_matrix @ np.array(raf.adjusted_forecasts).T) < 1e-5
    assert raf.adjusted_forecasts_test is None
    assert raf.forecasts_test is None

    # transforms test data
    forecasts_test = forecasts + 1.0
    raf.transform(forecasts_test=forecasts_test)
    reordered_forecasts_test = reorder_columns(forecasts_test, order_dict=order_dict)
    assert_equal(raf.forecasts_test, reordered_forecasts_test)
    assert raf.adjusted_forecasts_test is not None
    assert_equal(raf.adjusted_forecasts_test.columns, reordered_forecasts_test.columns)
    assert raf.adjusted_forecasts_test.shape == forecasts_test.shape
    assert np.linalg.norm(raf.constraint_matrix @ np.array(raf.adjusted_forecasts_test).T) < 1e-5

    # fit and transform training data
    raf2 = ReconcileAdditiveForecasts()
    raf2.fit_transform(
        forecasts=forecasts,
        actuals=actuals,
        levels=levels,
        order_dict=order_dict,
        method="ols")
    assert raf2.adjusted_forecasts is not None
    assert_equal(raf2.adjusted_forecasts.columns, reordered_forecasts.columns)
    assert raf2.adjusted_forecasts.shape == forecasts.shape
    assert np.linalg.norm(raf2.constraint_matrix @ np.array(raf2.adjusted_forecasts).T) < 1e-5
    assert raf2.adjusted_forecasts_test is None
    assert raf2.forecasts_test is None


def test_raf_evaluate(data):
    """Tests ReconcileAdditiveForecasts evaluate, transform_evaluate, fit_transform_evaluate"""
    forecasts = data["forecasts"]
    actuals = data["actuals"]
    levels = data["levels"]
    order_dict = data["order_dict"]
    raf = ReconcileAdditiveForecasts()
    raf.fit_transform(
        forecasts=forecasts,
        actuals=actuals,
        levels=levels,
        order_dict=order_dict,
        method="mint_sample")

    # Tests `evaluate` on training set
    raf.evaluate(is_train=True)
    assert raf.constraint_violation["actual"] < 1e-5
    assert raf.constraint_violation["adjusted"] < 1e-5
    assert raf.constraint_violation["forecast"] > 1e-5
    assert list(raf.evaluation_df.columns) == [
        "RMSE % change",
        "MAPE pp change",
        "MedAPE pp change",
        "Base MAPE",
        "Base MedAPE",
        "Base RMSE",
        "Adjusted MAPE",
        "Adjusted MedAPE",
        "Adjusted RMSE",
    ]
    assert_equal(
        raf.evaluation_df["RMSE % change"],
        100 * (raf.evaluation_df["Adjusted RMSE"] / raf.evaluation_df["Base RMSE"] - 1.0),
        check_names=False)
    assert_equal(
        raf.evaluation_df["MAPE pp change"],
        raf.evaluation_df["Adjusted MAPE"] - raf.evaluation_df["Base MAPE"],
        check_names=False)
    assert_equal(
        raf.evaluation_df["MedAPE pp change"],
        raf.evaluation_df["Adjusted MedAPE"] - raf.evaluation_df["Base MedAPE"],
        check_names=False)
    assert list(raf.evaluation_df["RMSE % change"] > 0) == [True, False, False, False]
    assert list(raf.evaluation_df["MAPE pp change"] > 0) == [True, False, False, False]
    assert list(raf.evaluation_df["MedAPE pp change"] > 0) == [True, False, False, False]
    assert_equal(
        mean_absolute_percent_error(raf.actuals["root"], raf.adjusted_forecasts["root"]),
        raf.evaluation_df.loc["root", "Adjusted MAPE"])
    assert_equal(
        mean_absolute_percent_error(raf.actuals["c1"], raf.forecasts["c1"]),
        raf.evaluation_df.loc["c1", "Base MAPE"])
    assert_equal(
        median_absolute_percent_error(raf.actuals["c2"], raf.adjusted_forecasts["c2"]),
        raf.evaluation_df.loc["c2", "Adjusted MedAPE"])
    assert_equal(
        median_absolute_percent_error(raf.actuals["c3"], raf.forecasts["c3"]),
        raf.evaluation_df.loc["c3", "Base MedAPE"])
    assert_equal(
        root_mean_squared_error(raf.actuals["root"], raf.adjusted_forecasts["root"]),
        raf.evaluation_df.loc["root", "Adjusted RMSE"])
    assert_equal(
        root_mean_squared_error(raf.actuals["c1"], raf.forecasts["c1"]),
        raf.evaluation_df.loc["c1", "Base RMSE"])

    # Changing the parameters affects the evaluation results
    raf = ReconcileAdditiveForecasts()
    raf.fit_transform(
        forecasts=forecasts,
        actuals=actuals,
        levels=levels,
        order_dict=order_dict,
        lower_bound=None,
        upper_bound=None,
        unbiased=True,
        lam_adj=0.0,
        lam_bias=0.0,
        lam_coef=0.0,
        lam_train=1.0,
        lam_var=0.0,
        covariance=None,
        weight_bias=None,
        weight_coef=None,
        weight_train=[2, 1, 1, 1],
        weight_var=None)
    # Tests `evaluate` on training set
    raf.evaluate(is_train=True)
    assert list(raf.evaluation_df["RMSE % change"] > 0) == [False, False, False, False]  # All have improved
    assert list(raf.evaluation_df["MAPE pp change"] > 0) == [False, False, False, False]
    assert list(raf.evaluation_df["MedAPE pp change"] > 0) == [False, False, False, False]

    # Tests `evaluate` on test set
    actuals_test = actuals + 2.0
    actuals_test.iloc[0, :] = 0.0  # forces divide by 0.0 in MAPE/MedAPE calculations
    forecasts_test = forecasts + np.random.randn(*actuals_test.shape)
    raf.transform(forecasts_test=forecasts_test)
    with pytest.raises(ValueError, match="`actuals_test` must be provided to evaluate on test set"):
        raf.evaluate(is_train=False)
    with np.errstate(divide='ignore'):
        raf.evaluate(is_train=False, actuals_test=actuals_test)
    assert_equal(raf.actuals_test, actuals_test)
    assert raf.evaluation_df_test is not None
    assert raf.constraint_violation_test is not None
    assert_equal(
        root_mean_squared_error(raf.actuals_test["c1"], raf.forecasts_test["c1"]),
        raf.evaluation_df_test.loc["c1", "Base RMSE"])

    # Tests `fit_transform_evaluate`
    # (and that plot/ipython_display run without error)
    raf = ReconcileAdditiveForecasts()
    raf.fit_transform(
        forecasts=forecasts,
        actuals=actuals,
        levels=levels,
        order_dict=order_dict,
        method="bottom_up")
    raf.evaluate(is_train=True)

    raf2 = ReconcileAdditiveForecasts()
    raf2.fit_transform_evaluate(
        forecasts=forecasts,
        actuals=actuals,
        fit_kwargs=dict(
            levels=levels,
            order_dict=order_dict,
            method="bottom_up"),
        evaluate_kwargs=dict(
            plot=True,
            ipython_display=True)
    )
    assert_equal(raf.evaluation_df, raf2.evaluation_df)
    assert_equal(raf.adjusted_forecasts, raf2.adjusted_forecasts)
    assert raf2.evaluation_df_test is None

    # Tests `transform_evaluate`
    with np.errstate(divide='ignore'):
        raf2.transform_evaluate(
            forecasts_test=forecasts_test,
            actuals_test=actuals_test,
            plot=True,
            ipython_display=True)
    assert_equal(
        root_mean_squared_error(raf2.actuals_test["c1"], raf2.forecasts_test["c1"]),
        raf2.evaluation_df_test.loc["c1", "Base RMSE"])
