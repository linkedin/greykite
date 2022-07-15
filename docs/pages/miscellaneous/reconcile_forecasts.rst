Reconcile Forecasts
===================

In many real-world scenarios, we need a set of forecasts
that satisfy inter-forecast additivity constraints. For example, the forecast
of total company revenue must be consistent with the
sum of forecasts for each business unit. The forecast of total population
must match the sum of forecasts for each geographic region.

To generate consistent forecasts, we could either use an algorithm that produces
consistent forecasts by design, or we could use a post-hoc forecast reconciliation
method that takes forecasts and makes them consistent.

`~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
takes the latter approach. You can use any algorithm to generate the base forecasts, and then use
`~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
to reconcile them.

This page explains the method details. For usage and examples,
see the tutorial (:doc:`/gallery/quickstart/04_postprocessing/0100_reconcile_forecasts`).

Intuition
---------

In `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`,
adjusted forecasts are computed as a linear transformation of the base forecasts. This allows
the adjusted forecasts to satisfy the linear constraints.

Let :math:`F_{base}` be an :math:`m \times n` matrix containing forecasts for :math:`m` time series
over :math:`n` time steps (wide format, each row is a time series).

The adjustment function is a linear operator defined by :math:`T`, an :math:`m \times m` matrix.
Applying :math:`T` to :math:`F_{base}` produces adjusted forecasts :math:`F_{adj}`,
an :math:`m \times n` matrix:

.. math::

    F_{adj} = T F_{base}.

Let :math:`C` be a :math:`c \times m` matrix encoding :math:`c` constraints
for the :math:`m` forecasts. :math:`C` defines linear constraints as follows:

.. math::

    C F_{adj} = C T F_{base} = 0.

The constraints are satisfied for all :math:`F_{base}` if every
column of :math:`T` is in the nullspace of :math:`C`.

For a given :math:`C`, there could be multiple possible transforms :math:`T`.
Our goal is to find the :math:`T` that returns the best adjusted forecasts.
`~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
tries to find a :math:`T` that minimizes the mean squared error (MSE).

We have two ways to estimate MSE:

#. Empirical MSE (error of :math:`F_{adj}` on the training set)
#. Decomposed MSE (estimated squared bias + variance from the adjustment)

Using empirical MSE alone could result in overfitting to the training set.
In addition to measuring decomposed MSE, we could mitigate this by:

* Requiring :math:`T` to be unbiased (details below). This results in a more stable adjustment,
  which is useful when extrapolating into the future. Unbiasedness is especially
  appropriate if the base forecasts are unbiased to start with.
* Introducing a regularization term on the adjustment size. This is appropriate if the
  base forecasts are good. We want to make a small adjustment to satisfy the constraints.

Optimization problem
--------------------

With this intuition,
`~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
defines :math:`T` as the solution to the following convex optimization problem:

:math:`\begin{align} & \text{minimize (w.r.t. $T$)}  && \frac{1}{m} \lambda_{var}  \left\lVert W_{var} T \sqrt{W_{h}} \right\rVert _{F}^{2} & \text{variance} \\ & \quad && + \frac{1}{mn} \lambda_{bias}  \left\lVert W_{bias} (TA-A) \right\rVert _{F}^{2} & \text{squared bias}\\ & \quad && + \frac{1}{mn} \lambda_{train}  \left\lVert W_{train} (TF_{base}-A) \right\rVert _{F}^{2}  & \text{train MSE}\\ & \quad&& + \frac{1}{mn} \lambda_{adj}  \left\lVert W_{adj} (TF_{base}-F_{base}) \right\rVert _{F}^{2}  & \text{regularization}\\ & \text{subject to} \quad && CT = 0 & \text{inter-forecast constraints}\\ & \quad&& TA = A & \text{optional, unbiasedness}\\ & \quad&& T \geq b_{lower} & \text{optional, lower bound}\\ & \quad&& T \leq b_{upper} & \text{optional, upper bound}\\ \end{align}`

..
    % Multi-line latex for the equation above.
    % Must use {align} instead of {split} (as in math:: directive) to align the comments.
    \begin{align}
          & \text{minimize (w.r.t. $T$)}  && \frac{1}{m} \lambda_{var}  \left\lVert W_{var} T \sqrt{W_{h}} \right\rVert _{F}^{2} & \text{variance} \\
          & \quad && + \frac{1}{mn} \lambda_{bias}  \left\lVert W_{bias} (TA-A) \right\rVert _{F}^{2} & \text{squared bias}\\
          & \quad && + \frac{1}{mn} \lambda_{train}  \left\lVert W_{train} (TF_{base}-A) \right\rVert _{F}^{2}  & \text{train MSE}\\
          & \quad&& + \frac{1}{mn} \lambda_{adj}  \left\lVert W_{adj} (TF_{base}-F_{base}) \right\rVert _{F}^{2}  & \text{regularization}\\
        & \text{subject to} \quad && CT = 0 & \text{inter-forecast constraints}\\
          & \quad&& TA = A & \text{optional, unbiasedness}\\
          & \quad&& T \geq b_{lower} & \text{optional, lower bound}\\
          & \quad&& T \leq b_{upper} & \text{optional, upper bound}\\
    \end{align}

Notation:

* Variable
    * :math:`T`, the :math:`m \times m` transform
* Constraint
    * :math:`C`, the :math:`c \times m` linear constraints
* Inputs
    * :math:`F_{base}`, the :math:`m \times n` base forecasts
    * :math:`A`, the :math:`m \times n` actual values (corresponding to the base forecasts)
* Tuning parameters
    * :math:`\lambda_{var}`, :math:`\lambda_{bias}`, :math:`\lambda_{train}`, :math:`\lambda_{adj}`,
      scalars that define the relative weight of each objective term
    * :math:`W_{h}`, the :math:`m \times m` variance-covariance matrix of base forecast errors
    * :math:`W_{var}`, :math:`W_{bias}`, :math:`W_{train}`, :math:`W_{adj}`,
      diagonal :math:`m \times m` weight matrices that define the relative weight of each
      time series for the penalty
    * :math:`b_{lower}`, optional lower bound for the entries in :math:`T`
    * :math:`b_{upper}`, optional upper bound for the entries in :math:`T`
* :math:`\left\lVert \cdot \right\rVert _{F}^{2}`, the squared Frobenius norm

.. note::

    `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
    pre-scales forecasts and actuals so that the actuals have mean 1 before fitting the optimization.
    This makes the optimization more stable.

    `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
    pre-scales the weight matrices so they have the same norm as the identity matrix of the same size.

Let's unpack this a bit.

Objective
---------

There are four terms (penalties) in the objective:

1. ``Var`` (for decomposed MSE)
2. ``Bias`` (for decomposed MSE)
3. ``Train`` (train MSE)
4. ``Adj`` (regularization)

The ``var`` term estimates the variance of the adjusted forecast errors,
assuming base forecasts and transform are unbiased. The derivation comes from
Wickramasuriya, Athanasopoulos & Hyndman 2019, lemma 1
(`link <https://robjhyndman.com/papers/mint.pdf>`_).

:math:`W_h` is positive semidefinite and symmetric, so its square root is symmetric.
Thus, the first term can be rewritten:

.. math::

    \left\lVert W_{var} T \sqrt{W_{h}} \right\rVert _{F}^{2}
    & = \mathrm{Tr}({W_{var}T\sqrt{W_{h}}\sqrt{W_{h}}'T'W_{var}'})\\
    & = \mathrm{Tr}({W_{var}TW_{h}T'W_{var}'})\\
    & = \mathrm{Tr}({W_{var}^{2}TW_{h}T'})

Modulo the tuning parameter :math:`W_{var}`, this is the variance of reconciled
forecast errors by Wickramasuriya et al. (:math:`T` here is equivalent to
:math:`SP` in their notation).

The normalizing constant :math:`\frac{1}{m}` on the variance term gives the average
for a single forecast.

The ``bias`` term estimates the squared bias of the transform.
Because actuals satisfy the constraints, we use actuals to assess bias,
computed as the difference between actuals and transformed actuals.
For unbiased transforms, :math:`TA=A`, so this term is 0.

The ``train`` term measures the MSE of the adjusted forecast on the training set.
Since the base forecast MSE is constant, it can also be interpreted as the change in
training MSE after adjustment.

The ``adj`` term adds regularization to prevent overfitting. It penalizes differences
between the forecasts and adjusted forecasts.

For the bias, train, and adj terms, the normalizing constant :math:`\frac{1}{mn}`
gives the average over the observed distribution.

Constraints
-----------

\1. :math:`CT = 0` requires the inter-forecast additivity constraints to be satisfied,
represented as a system of linear equations. For example, :math:`C` could require
:math:`X_{1}=X_{2}+X_{3}` and :math:`X_{2}=X_{4}+X_{5}`.

2. :math:`TA = A` is an optional constraint that enforces unbiasedness. This is helpful
to prevent overfitting. Additionally, the variance term in the objective assumes the
tranform is unbiased, so this is needed for a better variance estimate.

.. note::

    :math:`TA = A` represents unbiasedness for a particular :math:`A`.
    If the constraints are derived from a hierarchy (where each node's value is the sum of
    its children's), the unbiasedness constraint is :math:`TS = S`, where :math:`S` is
    the summing matrix for the tree (see ``sum_matrix`` in
    `~greykite.algo.reconcile.hierarchical_relationship.HierarchicalRelationship`
    for a definition). `~greykite.algo.reconcile.convex.reconcile_forecasts.ReconcileAdditiveForecasts`
    represents the constraint as :math:`TS = S` when possible and :math:`TA = A` otherwise.

3. Because large elements in :math:`T` can result in unstable forecasts, we allow
optional constraints on its entries, :math:`T \geq b_{lower}` and :math:`T \leq b_{upper}`.
For example, we could have :math:`-1.5 \leq T \leq 1.5`. In practice, these constraints are
often superfluous. Note that negative values in :math:`T` should be allowed; for hierarchical
constraints, this allows information to propagate "down" the tree from parent to children.

Tuning parameters
-----------------

Depending on the data, some terms in the objective may be more useful than others.
:math:`\lambda_{var}`, :math:`\lambda_{bias}`, :math:`\lambda_{train}`, :math:`\lambda_{adj}`,
allow you to tune these relative weight of each term.

Sometimes, it is more important to be accurate for some timeseries than for others.
If so, :math:`W_{var}`, :math:`W_{bias}`, :math:`W_{train}`, :math:`W_{adj}` can be used
to weigh the timeseries by their relative importance. If not, the weights
can still be used to fine tune the adjustment:

* Setting :math:`W_{var}`, :math:`W_{bias}`, :math:`W_{train}` proportional
  to the base forecast error for each time series can improve the result
  for base forecasts that start with high error.
* Setting :math:`W_{adj}` inversely proportional to the base forecast error for each
  time series puts greater penalty on adjustments to base forecasts that are already accurate.

See the tutorial (:doc:`/gallery/quickstart/04_postprocessing/0100_reconcile_forecasts`)
for details and suggested settings.

Related methods
---------------

The **bottom up method** is equivalent to setting :math:`\lambda_{adj}=1`, other :math:`\lambda\text{'s}` to 0,
:math:`W_{adj}` to only penalize adjustments to the leaf nodes, and adding
the unbiasedness constraint. This generalizes the "bottom up" method to work with multiple
overlapping trees, such as for constraints :math:`X_{1}=X_{2}+X_{3}`
and :math:`X_{1}=X_{4}+X_{5}+X_{6}`.

The **OLS method** from Hyndman et al. (`link <https://robjhyndman.com/papers/Hierarchical6.pdf>`_)
is equivalent to setting :math:`\lambda_{var}=1`, the other :math:`\lambda\text{'s}` to 0,
:math:`W_{h}` to the identity matrix, :math:`W_{var}` to the identity matrix,
and adding the unbiasedness constraint.

The **MinT method** with sample covariance from Wickramasuriya et al.
(`link <https://robjhyndman.com/papers/mint.pdf>`_)
is equivalent to setting :math:`\lambda_{var}=1`, the other :math:`\lambda\text{'s}` to 0,
:math:`W_{h}` to the sample covariance, :math:`W_{var}` to the identity matrix,
and adding the unbiasedness constraint.
