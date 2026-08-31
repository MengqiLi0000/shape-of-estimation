# The Shape of Estimation

**A visual research laboratory for geometry-aware statistical estimation, robustness, and strategic inference.**

Question:

> How does the behavior of an estimator change when the geometry of the data or the structure of the observations changes?

The project studies their behavior through four components:

**estimator + geometry + data distribution + perturbation**
<img width="1847" height="1204" alt="01_robustness_curve" src="https://github.com/user-attachments/assets/31cca747-5c99-4895-bae4-a77faf418c5a" />

<img width="1735" height="1336" alt="03_contamination_geometry" src="https://github.com/user-attachments/assets/285bca68-f2c7-4fb6-967d-ab97748442f3" />


<img width="1595" height="1341" alt="04_simplex_belief_trajectory_revised" src="https://github.com/user-attachments/assets/0ca94892-e631-429e-9067-dc68ec633eaf" />


<img width="1512" height="1576" alt="05_sphere_intrinsic_vs_extrinsic" src="https://github.com/user-attachments/assets/1ad97b36-3ffe-42b0-a5a7-188cce65a34a" />


## Overview

Many familiar statistical estimators implicitly assume Euclidean geometry.

For example, the sample mean is

$$
\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} X_i.
$$

On a general metric space, a natural analogue is the Fréchet mean:

```math
\hat{\mu}_F
=
\arg\min_{\theta}
\frac{1}{n}
\sum_{i=1}^{n}
d^2(X_i,\theta)
```

Once the geometry changes, several questions become interesting:

* How does curvature affect estimation?
* How do outliers move an estimator?
* When does an extrinsic approximation disagree with an intrinsic estimator?
* Which estimators remain stable under adversarial contamination?
* How should we average probability distributions?
* How should an agent update beliefs when observations may be deceptive?

The goal of this repository is to study these questions experimentally and visually.


## Research Questions

### 1. Geometry

How does the same estimator behave under different geometries?

Examples:

* Euclidean space
* sphere
* hyperbolic space
* probability simplex
* symmetric positive-definite matrices
* Wasserstein space

The goal is to compare estimation while changing as little else as possible.


### 2. Robustness

Suppose the observed distribution is contaminated:

```math
P_{\epsilon}
=
(1-\epsilon)P+\epsilon Q
```

As contamination increases, the estimator follows a path:

```math

\epsilon
\longrightarrow
T(P_{\epsilon}).
```

We visualize this path directly.

This gives a geometric view of robustness.


### 3. Model Misspecification

Suppose the data naturally live on a manifold, but estimation is performed in an ambient Euclidean space.

We compare:

```math
\hat{\theta}_{\text{intrinsic}}
```

and

```math
\hat{\theta}_{\text{extrinsic}}.
```

The question is:

> When does ignoring geometry become statistically expensive?


### 4. Strategic Inference

Suppose an agent maintains a belief about another agent's latent behavioral type:

```math
\theta_t \in \Delta^{K-1}.
```

As new actions are observed,

```math
a_1,a_2,\ldots,a_t,
```

the belief estimate evolves:

```math
\hat{\theta}_1
\rightarrow
\hat{\theta}_2
\rightarrow
\cdots
\rightarrow
\hat{\theta}_T.
```

If observations are strategically selected or deceptive, this becomes a robust estimation problem on the probability simplex.


## Experiment 1: Geometry of Influence

The first experiment studies robust location estimation on curved spaces.

We generate observations around a population center

$$
\theta^\star
$$

and introduce contamination:

```math
P_{\epsilon}
=
(1-\epsilon)P
+
\epsilon Q.
```

We compare:

* Fréchet mean
* geometric median
* Huber estimator
* extrinsic mean

across several geometries.

The main quantity is intrinsic estimation error:

```math
d(\hat{\theta},\theta^\star).
```

But the more interesting object is the entire estimator trajectory:

```math
\epsilon
\longrightarrow
\hat{\theta}_{\epsilon}.
```


## What We Visualize

### Estimator Trajectory

We visualize how the estimate moves as contamination increases.

For example:

```text
clean data

       ● ●
    ●       ●
       ★

small contamination

       ● ●
    ●       ●
       ★  →

large contamination

       ● ●
    ●       ●
             → ★
                  × × ×
```

Here:

* `●` = clean observations
* `×` = contaminating observations
* `★` = estimated center

The path of `★` is the object of interest.


### Error Curve

For each estimator, we measure

```math
\epsilon
\longrightarrow
d(\hat{\theta}_{\epsilon},\theta^\star).
```

This produces a traditional robustness curve.


### Objectives

For the Fréchet mean, define

```math
F(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
d^2(X_i,\theta).
```

Instead of showing only the minimizing point, we visualize the full objective surface.

This helps explain why the estimate moves.



### Influence Direction

A small contamination can be approximated through

```math
I_{\epsilon}
=
\frac{1}{\epsilon}
\log_{\hat{\theta}_0}
(\hat{\theta}_{\epsilon}).
```

This produces a tangent vector describing:

* how strongly the estimator moves
* in which direction it moves

This gives a geometric analogue of an influence function.


## Minimal Example

The following example estimates a Fréchet mean on the sphere.

```python
import matplotlib.pyplot as plt
import numpy as np

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean


sphere = Hypersphere(dim=2)

true_center = np.array([0.0, 0.0, 1.0])

X = sphere.random_von_mises_fisher(
    mu=true_center,
    kappa=20,
    n_samples=100,
)

estimator = FrechetMean(sphere)
estimator.fit(X)

estimated_center = estimator.estimate_

error = sphere.metric.dist(
    estimated_center,
    true_center,
)

print("True center:", true_center)
print("Estimated center:", estimated_center)
print("Intrinsic error:", error)


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X[:, 0],
    X[:, 1],
    X[:, 2],
    s=15,
    alpha=0.5,
    label="observations",
)

ax.scatter(
    *true_center,
    s=120,
    marker="*",
    label="true center",
)

ax.scatter(
    *estimated_center,
    s=100,
    marker="X",
    label="Fréchet mean",
)

ax.set_box_aspect((1, 1, 1))
ax.legend()

plt.show()
```

---

## Estimator Interface

Estimators use a common interface so that experiments do not depend on a particular implementation.

```python
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class Estimate:
    point: np.ndarray
    diagnostics: dict[str, Any]


class Estimator(Protocol):

    name: str

    def fit(
        self,
        X: np.ndarray,
        space: Any,
    ) -> Estimate:
        ...
```

Example:

```python
estimators = [
    FrechetEstimator(),
    GeometricMedian(),
    HuberLocation(delta=0.4),
]

for estimator in estimators:

    result = estimator.fit(
        X,
        space=sphere,
    )

    print(
        estimator.name,
        result.point,
    )
```


## Fréchet Mean

The Fréchet mean solves

```math
\hat{\theta}
=
\arg\min_{\theta}
\sum_i d^2(X_i,\theta).
```

Implementation:

```python
import numpy as np

from geomstats.learning.frechet_mean import FrechetMean


class FrechetEstimator:

    name = "frechet_mean"

    def __init__(self, method="adaptive"):
        self.method = method

    def fit(self, X, space):

        estimator = FrechetMean(
            space,
            method=self.method,
        )

        estimator.fit(X)

        point = np.asarray(
            estimator.estimate_
        )

        return Estimate(
            point=point,
            diagnostics={
                "method": self.method,
            },
        )
```



## Geometric Median

The geometric median solves

```math
\hat{\theta}
=
\arg\min_{\theta}
\sum_i d(X_i,\theta).
```

A simple manifold version can be implemented with iterative tangent-space updates.

```python
import numpy as np


class GeometricMedian:

    name = "geometric_median"

    def __init__(
        self,
        max_iter=200,
        tol=1e-7,
        eps=1e-8,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def fit(self, X, space):

        current = np.asarray(
            X[0]
        ).copy()

        converged = False

        for iteration in range(
            self.max_iter
        ):

            distances = np.asarray(
                space.metric.dist(
                    current,
                    X,
                )
            )

            weights = (
                1.0
                / np.maximum(
                    distances,
                    self.eps,
                )
            )

            tangent_vectors = np.asarray(
                space.metric.log(
                    point=X,
                    base_point=current,
                )
            )

            direction = np.sum(
                weights[:, None]
                * tangent_vectors,
                axis=0,
            )

            direction /= weights.sum()

            step_size = float(
                space.metric.norm(
                    direction,
                    current,
                )
            )

            if step_size < self.tol:
                converged = True
                break

            current = np.asarray(
                space.metric.exp(
                    direction,
                    current,
                )
            )

        return Estimate(
            point=current,
            diagnostics={
                "iterations": iteration + 1,
                "converged": converged,
            },
        )
```



## Huber Location Estimator

The Huber loss is

```math
\rho_\delta(r)
=
\begin{cases}
\frac{1}{2}r^2,
& r \leq \delta \\
\delta r-\frac{1}{2}\delta^2,
& r > \delta.
\end{cases}
```

A corresponding manifold estimator can use distance-dependent weights.

```python
import numpy as np


class HuberLocation:

    name = "huber"

    def __init__(
        self,
        delta=0.5,
        max_iter=200,
        tol=1e-7,
        eps=1e-8,
    ):
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def fit(self, X, space):

        current = np.asarray(
            X[0]
        ).copy()

        converged = False

        for iteration in range(
            self.max_iter
        ):

            distances = np.asarray(
                space.metric.dist(
                    current,
                    X,
                )
            )

            weights = np.minimum(
                1.0,
                self.delta
                / np.maximum(
                    distances,
                    self.eps,
                ),
            )

            tangent_vectors = np.asarray(
                space.metric.log(
                    point=X,
                    base_point=current,
                )
            )

            direction = np.sum(
                weights[:, None]
                * tangent_vectors,
                axis=0,
            )

            direction /= weights.sum()

            step_size = float(
                space.metric.norm(
                    direction,
                    current,
                )
            )

            if step_size < self.tol:
                converged = True
                break

            current = np.asarray(
                space.metric.exp(
                    direction,
                    current,
                )
            )

        return Estimate(
            point=current,
            diagnostics={
                "iterations": iteration + 1,
                "converged": converged,
                "delta": self.delta,
            },
        )
```

## Extrinsic Mean

For points on the sphere, the extrinsic estimator computes the Euclidean mean and projects it back onto the sphere.

```python
import numpy as np


class ExtrinsicSphereMean:

    name = "extrinsic_mean"

    def fit(self, X, space):

        mean = np.mean(
            X,
            axis=0,
        )

        norm = np.linalg.norm(mean)

        if norm < 1e-12:
            raise ValueError(
                "Extrinsic mean is undefined."
            )

        point = mean / norm

        return Estimate(
            point=point,
            diagnostics={
                "ambient_norm": float(norm),
            },
        )
```

This allows direct comparison between intrinsic and extrinsic estimation.


## Contamination Models

Contamination is treated as part of the experiment rather than hidden preprocessing.

A simple directional contamination model on the sphere places outliers close to the antipode of the true center.

```python
import numpy as np


class AntipodalContamination:

    def __init__(
        self,
        kappa=80.0,
        seed=0,
    ):
        self.kappa = kappa
        self.rng = np.random.default_rng(
            seed
        )

    def apply(
        self,
        X,
        epsilon,
        *,
        space,
        true_center,
    ):

        X_new = np.array(
            X,
            copy=True,
        )

        n = len(X_new)

        n_outliers = int(
            round(
                epsilon * n
            )
        )

        if n_outliers == 0:
            return X_new

        indices = self.rng.choice(
            n,
            size=n_outliers,
            replace=False,
        )

        outliers = (
            space.random_von_mises_fisher(
                mu=-np.asarray(
                    true_center
                ),
                kappa=self.kappa,
                n_samples=n_outliers,
            )
        )

        X_new[indices] = outliers

        return X_new
```


## Contamination Sweep

Experiments compare estimators over increasing contamination levels.

```python
import pandas as pd


def run_contamination_sweep(
    *,
    X_sampler,
    estimators,
    perturbation,
    space,
    true_center,
    epsilons,
    trials,
    sample_size,
):

    rows = []

    for epsilon in epsilons:

        for trial in range(trials):

            X = X_sampler(
                sample_size,
                trial,
            )

            X_noisy = perturbation.apply(
                X,
                epsilon,
                space=space,
                true_center=true_center,
            )

            for estimator in estimators:

                result = estimator.fit(
                    X_noisy,
                    space,
                )

                error = float(
                    space.metric.dist(
                        result.point,
                        true_center,
                    )
                )

                rows.append(
                    {
                        "epsilon": epsilon,
                        "trial": trial,
                        "estimator": (
                            estimator.name
                        ),
                        "error": error,
                    }
                )

    return pd.DataFrame(rows)
```


## Experiment Configuration

Example:

```yaml
experiment:
  name: geometry_of_influence
  seed: 42

space:
  name: sphere
  dimension: 2

distribution:
  name: von_mises_fisher
  kappa: 20.0

sample:
  n: 250

contamination:
  type: antipodal

  epsilon:
    - 0.00
    - 0.01
    - 0.02
    - 0.05
    - 0.10
    - 0.15
    - 0.20
    - 0.25

estimators:
  - frechet_mean
  - geometric_median
  - huber
  - extrinsic_mean

evaluation:
  trials: 100
```


## Risk Curves

```python
import matplotlib.pyplot as plt


def plot_risk_curves(df):

    summary = (
        df
        .groupby(
            [
                "epsilon",
                "estimator",
            ],
            as_index=False,
        )
        .error
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for estimator, group in (
        summary.groupby(
            "estimator"
        )
    ):

        ax.plot(
            group["epsilon"],
            group["error"],
            marker="o",
            label=estimator,
        )

    ax.set_xlabel(
        "Contamination fraction"
    )

    ax.set_ylabel(
        "Mean intrinsic error"
    )

    ax.set_title(
        "Estimator robustness"
    )

    ax.legend()

    return fig, ax
```


## Experiment 2: Intrinsic vs Extrinsic Estimation

This experiment studies when Euclidean approximations stop working well.

We compare:

* intrinsic Fréchet mean
* extrinsic projected mean

and measure

```math
d(
\hat{\theta}_{\text{intrinsic}},
\hat{\theta}_{\text{extrinsic}}
).
```

We vary:

* sample concentration
* sample size
* multimodality
* contamination

The main question is:

> How far can the data spread before the geometry becomes important?


## Experiment 3: KDE Under Geometry

Standard KDE uses Euclidean distance:

```math
\hat{f}_h(x)
=
\frac{1}{nh^d}
\sum_i
K
\left(
\frac{\|x-X_i\|}{h}
\right).
```

A geometry-aware version replaces Euclidean distance with intrinsic distance:

```math
\hat{f}^{M}_h(x)
\propto
\sum_i
K
\left(
\frac{d_M(x,X_i)}{h}
\right).
```

The experiment asks:

> At what scale does using the wrong geometry meaningfully distort density estimation?

We vary:

* bandwidth
* curvature
* dispersion
* sample size


## Experiment 4: Wasserstein Averaging

Suppose we want to estimate an average distribution.

An arithmetic mixture is

```math
\bar{P}
=
\frac{1}{K}
\sum_i P_i.
```

A Wasserstein barycenter instead solves

```math
P^\star
=
\arg\min_P
\sum_i
\lambda_i
W_2^2(P,P_i).
```

These represent different notions of averaging.

Example:

```python
import numpy as np
import ot


grid = np.linspace(
    -5.0,
    5.0,
    200,
)

difference = (
    grid[:, None]
    - grid[None, :]
)

cost = difference ** 2


def gaussian_histogram(
    mean,
    sigma,
):

    density = np.exp(
        -0.5
        * (
            (grid - mean)
            / sigma
        ) ** 2
    )

    return (
        density
        / density.sum()
    )


p1 = gaussian_histogram(
    mean=-2.0,
    sigma=0.6,
)

p2 = gaussian_histogram(
    mean=2.0,
    sigma=0.6,
)

distributions = np.column_stack(
    [p1, p2]
)

wasserstein_average = ot.barycenter(
    distributions,
    cost,
    reg=1e-2,
)

linear_average = (
    0.5 * p1
    + 0.5 * p2
)
```

The visualization compares how probability mass moves under the two notions of averaging.

---

## Experiment 5: Strategic Belief Geometry

This experiment connects geometric statistics to multi-agent reasoning.

Suppose another agent has three possible behavioral modes:

```text
cooperative
opportunistic
adversarial
```

Our belief is

```math
\theta
=
(
p_{\text{coop}},
p_{\text{opp}},
p_{\text{adv}}
).
```

Because

```math
p_{\text{coop}}
+
p_{\text{opp}}
+
p_{\text{adv}}
=
1,
```

the belief lies on a probability simplex.

A sequence of observations creates a trajectory through this space.


### Bayesian Baseline

```python
import numpy as np


def bayesian_update(
    prior,
    likelihood,
):

    posterior = (
        prior
        * likelihood
    )

    posterior = (
        posterior
        / posterior.sum()
    )

    return posterior
```

Sequential inference:

```python
theta = np.array(
    [0.33, 0.34, 0.33]
)

trajectory = [
    theta.copy()
]

for observation in observations:

    likelihood = (
        observation_model[
            observation
        ]
    )

    theta = bayesian_update(
        theta,
        likelihood,
    )

    trajectory.append(
        theta.copy()
    )
```


### Deceptive Observations

Now suppose another agent chooses actions partly because of how those actions will affect our beliefs.

The observation process is no longer passive.

An action may depend on:

* the opponent's true type
* the opponent's objective
* the opponent's belief about us
* the belief they want us to form

The inference problem becomes:

```math
P(
a_t
\mid
\text{type},
\text{beliefs},
\text{strategic objective}
).
```

The experiment asks:

> How quickly can different estimators recover after strategically misleading evidence?

Useful metrics include:

* maximum belief displacement
* recovery time
* cumulative belief error
* downstream strategic regret


## Simplex Visualization

For three latent types, beliefs can be mapped into a triangle.

```python
import numpy as np


SIMPLEX_VERTICES = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [
            0.5,
            np.sqrt(3.0) / 2.0,
        ],
    ]
)


def simplex_to_cartesian(
    theta,
):

    theta = np.asarray(
        theta
    )

    if not np.isclose(
        theta.sum(),
        1.0,
    ):
        raise ValueError(
            "Beliefs must sum to one."
        )

    return (
        theta
        @ SIMPLEX_VERTICES
    )
```

Plotting the sequence produces a direct visualization of belief movement.

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_belief_trajectory(
    trajectory,
):

    xy = np.vstack(
        [
            simplex_to_cartesian(
                theta
            )
            for theta
            in trajectory
        ]
    )

    triangle = np.vstack(
        [
            SIMPLEX_VERTICES,
            SIMPLEX_VERTICES[0],
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.plot(
        triangle[:, 0],
        triangle[:, 1],
    )

    ax.plot(
        xy[:, 0],
        xy[:, 1],
        marker="o",
    )

    ax.scatter(
        *xy[0],
        marker="*",
        s=140,
        label="initial belief",
    )

    ax.scatter(
        *xy[-1],
        marker="X",
        s=120,
        label="final belief",
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend()

    return fig, ax
```

---

## Evaluation

Each experiment should report several views of estimator behavior.

### Estimation Error

```math
R
=
\mathbb{E}
\left[
d^2(
\hat{\theta},
\theta^\star
)
\right].
```

### Perturbation Sensitivity

```math
S(\epsilon)
=
d(
T(P),
T(P_{\epsilon})
).
```

### Trajectory Length

For contamination levels

```math
\epsilon_1,\ldots,\epsilon_m,
```

define

```math
L
=
\sum_{j=1}^{m-1}
d(
\hat{\theta}_{\epsilon_j},
\hat{\theta}_{\epsilon_{j+1}}
).
```

### Recovery Time

For sequential inference,

```math
\tau_\delta
=
\min
\left\{
t:
d(
\hat{\theta}_t,
\theta^\star
)
<
\delta
\right\}.
```


## Reproducibility

Run one experiment:

```bash
python experiments/01_geometry_of_influence/run.py
```

Run all experiments:

```bash
python scripts/reproduce_all.py
```

Generate figures from saved results:

```bash
python scripts/reproduce_figures.py
```

Important results should always be reproducible without running a notebook.

Notebooks are used for exploration.

Scripts are used for experiments.

---

## Result Format

Raw experiment output should use long-form records:

```text
experiment
space
trial
seed
sample_size
epsilon
estimator
error
objective
iterations
```

Example:

```text
geometry_of_influence,sphere,0,42,250,0.05,frechet_mean,0.083,...
geometry_of_influence,sphere,0,42,250,0.05,huber,0.041,...
geometry_of_influence,sphere,0,42,250,0.05,geometric_median,0.029,...
```

Figures should be generated from saved result files instead of simulation code.

---

## License 
MIT
