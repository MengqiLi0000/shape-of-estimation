# The Shape of Estimation

### Geometry, robustness, and statistical inference beyond Euclidean assumptions.

** A visual research laboratory for studying how statistical estimators behave when the geometry of the data, the observation model, or the data-generating process changes.

The object of the project is an estimator viewed as a functional

$$
T:\mathcal P(\mathcal X)\rightarrow\Theta,
$$

where both the observation space \(\mathcal X\) and parameter space \(\Theta\) may be non-Euclidean.

This project studies their behavior through four interacting objects:

$$
\boxed{
\text{Estimator behavior}
=
\text{objective}
+
\text{geometry}
+
\text{data distribution}
+
\text{perturbation}
}
$$

The goal is to make these interactions **visible**.

---

## Why this project?

A sample mean in \(\mathbb R^d\) looks trivial:

$$
\hat\mu
=
\frac1n\sum_{i=1}^n X_i.
$$

But this formula quietly relies on Euclidean structure.

On a general metric space \((\mathcal X,d)\), a more intrinsic notion of center is the Fréchet mean:

$$
\hat\mu_F
=
\arg\min_{\theta\in\mathcal X}
\frac1n
\sum_{i=1}^n d^2(X_i,\theta).
$$

Now several questions appear.

* What happens when the space has positive or negative curvature?
* How does contamination move the estimator?
* When does an extrinsic approximation stop agreeing with the intrinsic estimator?
* Which estimators remain stable under adversarial perturbation?
* How does the geometry of probability distributions change the meaning of "average"?
* Can the same ideas explain belief estimation in strategic multi-agent systems?

This repository treats these questions experimentally.

It is designed around **estimator trajectories, loss landscapes, influence fields, and phase diagrams**.

---

# Research program

The project studies four related questions.

## Q1. Geometry

How does changing the geometry while holding the statistical task fixed change the estimator?

$$
(\mathcal X,d_1)
\quad\longrightarrow\quad
T_{d_1}(P)
$$

versus

$$
(\mathcal X,d_2)
\quad\longrightarrow\quad
T_{d_2}(P).
$$

Examples include:

* Euclidean space
* spheres
* hyperbolic space
* probability simplices
* symmetric positive-definite matrices
* Wasserstein spaces

---

## Q2. Robustness

For a contamination path

$$
P_\epsilon
=
(1-\epsilon)P+\epsilon Q,
$$

we study the estimator trajectory

$$
\epsilon
\mapsto
T(P_\epsilon).
$$

We visualize **how the estimator moves through its parameter space**.

This gives a geometric view of robustness.

---

## Q3. Misspecification

Suppose the observations intrinsically live on a manifold, but inference is performed after embedding them into Euclidean space.

How large is

$$
d\left(
T_{\text{intrinsic}}(P),
T_{\text{extrinsic}}(P)
\right)?
$$

When does ignoring geometry become statistically meaningful?

---

## Q4. Strategic inference

Suppose an agent maintains a belief

$$
\theta_t
\in
\Delta^{K-1}
$$

about another agent's latent strategy.

As observations arrive,

$$
a_1,a_2,\ldots,a_t,
$$

the estimator follows a trajectory

$$
\hat\theta_1
\rightarrow
\hat\theta_2
\rightarrow
\cdots
\rightarrow
\hat\theta_T.
$$

If some observations are deceptive or strategically generated, we can ask:

> How does adversarial information move belief estimators through the simplex?

This connects robust statistics and geometric inference to artificial social intelligence.

---

# Abstraction

Every experiment in this repository is defined by six objects:

$$
\boxed{
E =
(\mathcal X,P,T,C,R,V)
}
$$

where

| Symbol         | Meaning                                 |
| -------------- | --------------------------------------- |
| \(\mathcal X\) | statistical space and geometry          |
| \(P\)          | data-generating process                 |
| \(T\)          | estimator                               |
| \(C\)          | contamination or perturbation mechanism |
| \(R\)          | risk functional                         |
| \(V\)          | visualization                           |

This structure is intentionally strict.

A new estimator does not automatically constitute a new experiment.

A useful experiment should isolate a statistical phenomenon.

---

# Flagship experiment

## Experiment 01 — The Geometry of Influence

The first experiment studies robust location estimation on curved spaces.

We generate observations around a population center

$$
\theta^\star
$$

and introduce contamination

$$
P_\epsilon
=
(1-\epsilon)P
+
\epsilon Q.
$$

We then compare:

* Fréchet mean
* geometric median
* Huber location estimator
* extrinsic Euclidean mean

across:

* spherical geometry
* Euclidean geometry
* hyperbolic geometry

while varying:

* sample size \(n\)
* contamination \(\epsilon\)
* concentration
* contamination direction
* geometric curvature

The primary quantity is intrinsic estimation risk:

$$
R(T)
=
\mathbb E
\left[
d^2
\left(
T(\hat P_n),
\theta^\star
\right)
\right].
$$

But the principal output is not the scalar risk.

It is the **estimator trajectory**

$$
\epsilon
\mapsto
\hat\theta_\epsilon.
$$

---

# What we visualize

Each estimator should eventually have a visual "portrait."

## 1. Estimator trajectory

$$
\epsilon
\mapsto
T(P_\epsilon)
$$

shows where the estimator moves as contamination increases.

---

## 2. Error trajectory

$$
\epsilon
\mapsto
d(T(P_\epsilon),\theta^\star).
$$

This gives the traditional robustness view.

---

## 3. Objective landscape

For a Fréchet estimator:

$$
F(\theta)
=
\frac1n
\sum_{i=1}^{n}
d^2(X_i,\theta).
$$

Instead of showing only

$$
\arg\min_\theta F(\theta),
$$

we visualize the entire landscape.

---

## 4. Influence field

For a contamination point \(z\), approximate

$$
\operatorname{IF}(z;T,P)
=
\lim_{\epsilon\rightarrow 0}
\frac{
T((1-\epsilon)P+\epsilon\delta_z)
-
T(P)
}{
\epsilon
}.
$$

On manifolds, subtraction is replaced by the logarithmic map.

We therefore study the tangent-space quantity

$$
\operatorname{IF}_{\mathcal M}
(z;T,P)
\approx
\frac{
\log_{T(P)}
T(P_\epsilon)
}{
\epsilon
}.
$$

This produces a vector field showing **where individual contaminating observations try to pull an estimator**.

---

## 5. Phase diagram

The flagship summary figure is

$$
(\text{geometry},\epsilon)
\mapsto
R(T).
$$

For parameterized spaces, we can study

$$
(\kappa,\epsilon)
\mapsto
R(T),
$$

where \(\kappa\) represents curvature.

This produces an empirical robustness phase diagram.

---

# Repository structure

```text
shape-of-estimation/
│
├── README.md
├── LICENSE
├── pyproject.toml
│
├── src/
│   └── shape_estimation/
│       │
│       ├── spaces/
│       │   ├── base.py
│       │   ├── euclidean.py
│       │   ├── sphere.py
│       │   ├── hyperbolic.py
│       │   ├── simplex.py
│       │   └── spd.py
│       │
│       ├── estimators/
│       │   ├── base.py
│       │   ├── frechet.py
│       │   ├── geometric_median.py
│       │   ├── huber.py
│       │   ├── extrinsic.py
│       │   ├── kde.py
│       │   └── barycenter.py
│       │
│       ├── distributions/
│       │   ├── directional.py
│       │   ├── mixtures.py
│       │   └── simplex.py
│       │
│       ├── perturbations/
│       │   ├── contamination.py
│       │   ├── directional.py
│       │   ├── adversarial.py
│       │   └── label_noise.py
│       │
│       ├── metrics/
│       │   ├── risk.py
│       │   ├── influence.py
│       │   ├── stability.py
│       │   └── calibration.py
│       │
│       ├── experiments/
│       │   ├── runner.py
│       │   ├── sweep.py
│       │   └── registry.py
│       │
│       └── visualization/
│           ├── sphere.py
│           ├── simplex.py
│           ├── landscapes.py
│           ├── trajectories.py
│           └── phase_diagram.py
│
├── experiments/
│   │
│   ├── 01_geometry_of_influence/
│   │   ├── config.yaml
│   │   ├── run.py
│   │   └── README.md
│   │
│   ├── 02_intrinsic_vs_extrinsic/
│   │   ├── config.yaml
│   │   └── run.py
│   │
│   ├── 03_kde_under_curvature/
│   │   ├── config.yaml
│   │   └── run.py
│   │
│   ├── 04_wasserstein_averaging/
│   │   ├── config.yaml
│   │   └── run.py
│   │
│   └── 05_deceptive_beliefs/
│       ├── config.yaml
│       └── run.py
│
├── notebooks/
│   ├── 01_frechet_mean.ipynb
│   ├── 02_influence_geometry.ipynb
│   ├── 03_manifold_kde.ipynb
│   ├── 04_wasserstein_barycenters.ipynb
│   └── 05_belief_simplex.ipynb
│
├── tests/
│   ├── test_estimators.py
│   ├── test_spaces.py
│   └── test_reproducibility.py
│
├── results/
│   ├── raw/
│   ├── aggregated/
│   └── figures/
│
└── scripts/
    ├── reproduce_figure_1.py
    ├── reproduce_figure_2.py
    └── reproduce_all.py
```

---

# Installation

Python 3.11+ is recommended.

```bash
git clone <repository>
cd shape-of-estimation

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

Development installation:

```bash
pip install -e ".[dev]"
```

---

# Dependencies

A minimal `pyproject.toml` can begin with:

```toml
[project]
name = "shape-estimation"
version = "0.1.0"
description = "Visual experiments in geometry-aware statistical estimation and robustness."
requires-python = ">=3.11"

dependencies = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "geomstats",
    "pot",
    "pyyaml",
    "tqdm",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
    "jupyter",
]

optimization = [
    "pymanopt",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

---

# Minimal example

The following example generates observations on \(S^2\), estimates their intrinsic Fréchet mean, and visualizes the result.

```python
import matplotlib.pyplot as plt
import numpy as np

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean


rng = np.random.default_rng(7)

sphere = Hypersphere(dim=2)

true_center = np.array([0.0, 0.0, 1.0])

X = sphere.random_von_mises_fisher(
    mu=true_center,
    kappa=20,
    n_samples=100,
)

estimator = FrechetMean(sphere)
estimator.fit(X)

frechet_mean = estimator.estimate_

print("True center:", true_center)
print("Estimated center:", frechet_mean)
print(
    "Intrinsic error:",
    sphere.metric.dist(true_center, frechet_mean),
)


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
    label="population center",
)

ax.scatter(
    *frechet_mean,
    s=100,
    marker="X",
    label="Fréchet mean",
)

ax.set_box_aspect((1, 1, 1))
ax.legend()
plt.show()
```

---

# Estimator interface

Every estimator should expose the same conceptual interface.

```python
from __future__ import annotations

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

This allows experiments to remain estimator-agnostic.

```python
estimators = [
    FrechetEstimator(),
    GeometricMedian(),
    HuberLocation(delta=0.4),
    ExtrinsicMean(),
]

for estimator in estimators:
    result = estimator.fit(X, space=sphere)
```

---

# Fréchet mean

The Fréchet estimator solves

$$
\hat\theta_F
=
\arg\min_\theta
\sum_i
d^2(X_i,\theta).
$$

Wrapper:

```python
import numpy as np

from geomstats.learning.frechet_mean import FrechetMean

from shape_estimation.estimators.base import Estimate


class FrechetEstimator:
    name = "frechet_mean"

    def __init__(self, method: str = "adaptive"):
        self.method = method

    def fit(self, X, space):
        model = FrechetMean(
            space,
            method=self.method,
        )

        model.fit(X)

        point = np.asarray(model.estimate_)

        objective = np.mean(
            np.asarray(
                space.metric.squared_dist(
                    point,
                    X,
                )
            )
        )

        return Estimate(
            point=point,
            diagnostics={
                "objective": float(objective),
                "method": self.method,
            },
        )
```

---

# Riemannian geometric median

The intrinsic geometric median solves

$$
\hat\theta_{\mathrm{GM}}
=
\arg\min_\theta
\sum_i
d(X_i,\theta).
$$

A simple generalized Weiszfeld-style implementation:

```python
import numpy as np

from shape_estimation.estimators.base import Estimate


class GeometricMedian:
    name = "geometric_median"

    def __init__(
        self,
        max_iter: int = 256,
        tol: float = 1e-7,
        eps: float = 1e-8,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def fit(self, X, space):
        current = np.asarray(X[0]).copy()

        converged = False

        for iteration in range(self.max_iter):
            distances = np.asarray(
                space.metric.dist(
                    current,
                    X,
                )
            )

            weights = 1.0 / np.maximum(
                distances,
                self.eps,
            )

            tangent_vectors = np.asarray(
                space.metric.log(
                    point=X,
                    base_point=current,
                )
            )

            direction = np.sum(
                weights.reshape((-1,) + (1,) * (tangent_vectors.ndim - 1))
                * tangent_vectors,
                axis=0,
            )

            direction /= np.sum(weights)

            step_norm = float(
                space.metric.norm(
                    direction,
                    current,
                )
            )

            if step_norm < self.tol:
                converged = True
                break

            current = np.asarray(
                space.metric.exp(
                    direction,
                    current,
                )
            )

        objective = np.sum(
            np.asarray(
                space.metric.dist(
                    current,
                    X,
                )
            )
        )

        return Estimate(
            point=current,
            diagnostics={
                "objective": float(objective),
                "iterations": iteration + 1,
                "converged": converged,
            },
        )
```

This implementation is intended as a transparent research baseline.

Production experiments should explicitly handle degenerate cases in which an iterate approaches an observation.

---

# Riemannian Huber estimator

Define the robust loss

$$
\rho_\delta(r)
=
\begin{cases}
\frac12r^2,
&
r\leq\delta,
\\
\delta r-\frac12\delta^2,
&
r>\delta.
\end{cases}
$$

The corresponding weight behaves approximately as

$$
w(r)
=
\min\left(1,\frac{\delta}{r}\right).
$$

Implementation:

```python
import numpy as np

from shape_estimation.estimators.base import Estimate


class HuberLocation:
    name = "huber"

    def __init__(
        self,
        delta: float = 0.5,
        max_iter: int = 256,
        tol: float = 1e-7,
        eps: float = 1e-8,
    ):
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def fit(self, X, space):
        current = np.asarray(X[0]).copy()

        converged = False

        for iteration in range(self.max_iter):
            distances = np.asarray(
                space.metric.dist(
                    current,
                    X,
                )
            )

            weights = np.minimum(
                1.0,
                self.delta / np.maximum(distances, self.eps),
            )

            tangent_vectors = np.asarray(
                space.metric.log(
                    point=X,
                    base_point=current,
                )
            )

            reshape = (-1,) + (1,) * (tangent_vectors.ndim - 1)

            direction = np.sum(
                weights.reshape(reshape) * tangent_vectors,
                axis=0,
            ) / np.sum(weights)

            step_norm = float(
                space.metric.norm(
                    direction,
                    current,
                )
            )

            if step_norm < self.tol:
                converged = True
                break

            current = np.asarray(
                space.metric.exp(
                    direction,
                    current,
                )
            )

        distances = np.asarray(
            space.metric.dist(
                current,
                X,
            )
        )

        quadratic = distances <= self.delta

        loss = np.where(
            quadratic,
            0.5 * distances**2,
            self.delta * distances - 0.5 * self.delta**2,
        )

        return Estimate(
            point=current,
            diagnostics={
                "objective": float(np.sum(loss)),
                "iterations": iteration + 1,
                "converged": converged,
                "delta": self.delta,
            },
        )
```

---

# Extrinsic mean

For an embedded manifold

$$
\mathcal M
\subset
\mathbb R^D,
$$

an extrinsic estimator first averages in the ambient Euclidean space and then projects back onto the manifold.

For \(S^{d}\),

$$
\hat\mu_{\text{ext}}
=
\frac{
\sum_i X_i
}{
\left\|
\sum_i X_i
\right\|
}.
$$

```python
import numpy as np

from shape_estimation.estimators.base import Estimate


class ExtrinsicSphereMean:
    name = "extrinsic_mean"

    def fit(self, X, space):
        ambient_mean = np.mean(X, axis=0)

        norm = np.linalg.norm(ambient_mean)

        if norm < 1e-12:
            raise ValueError(
                "Extrinsic mean is undefined because the ambient mean is approximately zero."
            )

        point = ambient_mean / norm

        return Estimate(
            point=point,
            diagnostics={
                "ambient_norm": float(norm),
            },
        )
```

One important experiment is therefore simply

$$
d(
\hat\mu_{\mathrm{ext}},
\hat\mu_F
)
$$

as the distribution becomes broader or the geometry becomes more important.

---

# Contamination models

Contamination should be treated as a first-class object.

```python
from typing import Protocol

import numpy as np


class Perturbation(Protocol):
    def apply(
        self,
        X: np.ndarray,
        epsilon: float,
        **kwargs,
    ) -> np.ndarray:
        ...
```

---

## Directional contamination on the sphere

Suppose the population is centered around

$$
\theta^\star.
$$

A strong adversarial direction is the antipode

$$
-\theta^\star.
$$

```python
import numpy as np


class AntipodalContamination:
    def __init__(
        self,
        kappa: float = 80.0,
        seed: int = 0,
    ):
        self.kappa = kappa
        self.rng = np.random.default_rng(seed)

    def apply(
        self,
        X,
        epsilon,
        *,
        space,
        true_center,
    ):
        X_new = np.array(X, copy=True)

        n = len(X_new)
        k = int(round(epsilon * n))

        if k == 0:
            return X_new

        indices = self.rng.choice(
            n,
            size=k,
            replace=False,
        )

        outliers = space.random_von_mises_fisher(
            mu=-np.asarray(true_center),
            kappa=self.kappa,
            n_samples=k,
        )

        X_new[indices] = outliers

        return X_new
```

This creates a controlled path from clean to adversarial data.

---

# Experiment runner

The experiment runner should separate simulation from visualization.

```python
from dataclasses import dataclass

import pandas as pd


@dataclass
class ExperimentConfig:
    epsilons: list[float]
    trials: int
    sample_size: int


def run_contamination_sweep(
    *,
    X_sampler,
    estimators,
    perturbation,
    space,
    true_center,
    config,
):
    rows = []

    for epsilon in config.epsilons:
        for trial in range(config.trials):

            X = X_sampler(
                config.sample_size,
                trial,
            )

            X_eps = perturbation.apply(
                X,
                epsilon,
                space=space,
                true_center=true_center,
            )

            for estimator in estimators:
                result = estimator.fit(
                    X_eps,
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
                        "estimator": estimator.name,
                        "error": error,
                        **result.diagnostics,
                    }
                )

    return pd.DataFrame(rows)
```

---

# Reproducible experiment configuration

`experiments/01_geometry_of_influence/config.yaml`

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
  - name: frechet_mean

  - name: geometric_median

  - name: huber
    delta: 0.40

  - name: extrinsic_mean

evaluation:
  trials: 100

outputs:
  save_raw: true
  save_aggregated: true
  save_figures: true
```

---

# Risk curves

Given experiment results:

```python
import matplotlib.pyplot as plt


def plot_risk_curves(df):
    summary = (
        df.groupby(
            ["epsilon", "estimator"],
            as_index=False,
        )
        .error.mean()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for estimator, group in summary.groupby("estimator"):
        ax.plot(
            group["epsilon"],
            group["error"],
            marker="o",
            label=estimator,
        )

    ax.set_xlabel("Contamination fraction")
    ax.set_ylabel("Mean intrinsic estimation error")
    ax.set_title("Estimator robustness under contamination")
    ax.legend()

    return fig, ax
```

The important output is the functional relationship

$$
\epsilon
\mapsto
\mathbb E[
d(\hat\theta_\epsilon,\theta^\star)
].
$$

---

# Estimator trajectories

Risk curves destroy directional information.

The trajectory itself contains substantially more structure:

$$
\hat\theta_0,
\hat\theta_{0.01},
\hat\theta_{0.02},
\dots
$$

For a sphere:

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_sphere_trajectory(
    trajectory,
    observations=None,
):
    trajectory = np.asarray(trajectory)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    if observations is not None:
        observations = np.asarray(observations)

        ax.scatter(
            observations[:, 0],
            observations[:, 1],
            observations[:, 2],
            s=8,
            alpha=0.25,
        )

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        marker="o",
    )

    ax.scatter(
        *trajectory[0],
        marker="*",
        s=150,
        label="clean estimate",
    )

    ax.scatter(
        *trajectory[-1],
        marker="X",
        s=120,
        label="final estimate",
    )

    ax.set_box_aspect((1, 1, 1))
    ax.legend()

    return fig, ax
```

The geometry of this path can itself become an object of study.

For example, define path length

$$
L_T
=
\sum_j
d(
\hat\theta_{\epsilon_j},
\hat\theta_{\epsilon_{j+1}}
).
$$

This distinguishes a smoothly degrading estimator from one that undergoes abrupt transitions.

---

# Local sensitivity

A finite-difference approximation to the manifold-valued influence function is

$$
I_\epsilon
=
\frac1\epsilon
\log_{\hat\theta_0}
\hat\theta_\epsilon.
$$

```python
import numpy as np


def approximate_influence(
    *,
    clean_estimate,
    contaminated_estimate,
    epsilon,
    space,
):
    tangent = np.asarray(
        space.metric.log(
            point=contaminated_estimate,
            base_point=clean_estimate,
        )
    )

    return tangent / epsilon
```

Its magnitude is

$$
\|I_\epsilon\|_{\hat\theta_0}.
$$

```python
def influence_norm(
    influence,
    *,
    base_point,
    space,
):
    return float(
        space.metric.norm(
            influence,
            base_point,
        )
    )
```

This makes robustness directly visualizable in the tangent space.

---

# Experiment 02 — Intrinsic versus extrinsic estimation

The second experiment asks a deliberately simple question:

> When does pretending that curved data are Euclidean become statistically expensive?

For a manifold-valued distribution \(P\), compare

$$
\hat\theta_{\text{intrinsic}}
$$

and

$$
\hat\theta_{\text{extrinsic}}.
$$

Study

$$
D(P)
=
d(
\hat\theta_{\text{intrinsic}},
\hat\theta_{\text{extrinsic}}
).
$$

Vary:

* concentration
* multimodality
* sample size
* distance from a geodesically convex neighborhood
* contamination

A useful output is:

$$
\text{dispersion}
\mapsto
D(P).
$$

The expected phenomenon is that local Euclidean approximations are often reasonable for sufficiently concentrated distributions, while differences become increasingly visible as observations explore larger regions of the manifold.

The experiment should determine where that transition actually occurs.

---

# Experiment 03 — KDE under geometry

For Euclidean KDE,

$$
\hat f_h(x)
=
\frac1{nh^d}
\sum_i
K\left(
\frac{\|x-X_i\|}{h}
\right).
$$

A geometry-aware kernel instead uses intrinsic distance:

$$
\hat f_h^{\mathcal M}(x)
\propto
\sum_i
K\left(
\frac{d_{\mathcal M}(x,X_i)}{h}
\right).
$$

The central question becomes:

$$
\boxed{
\text{At what statistical scale does the metric matter?}
}
$$

Study

$$
D(h,P)
=
D_{\mathrm{KL}}
\left(
\hat f_h^{\text{intrinsic}}
\|
\hat f_h^{\text{extrinsic}}
\right)
$$

or another appropriate divergence.

Primary visualization:

$$
(h,\text{dispersion})
\mapsto
D(h,P).
$$

---

# Experiment 04 — The geometry of averaging distributions

Arithmetic averaging of probability vectors and Wasserstein averaging encode different notions of interpolation.

Given distributions

$$
P_1,\ldots,P_k,
$$

the Wasserstein barycenter solves

$$
P^\star
=
\arg\min_P
\sum_j
\lambda_j
W_2^2(P,P_j).
$$

A simple discrete example:

```python
import numpy as np
import ot


grid = np.linspace(-5.0, 5.0, 200)

dx = grid[:, None] - grid[None, :]
cost = dx**2


def gaussian_histogram(mean, sigma):
    density = np.exp(
        -0.5
        * ((grid - mean) / sigma) ** 2
    )

    return density / density.sum()


p1 = gaussian_histogram(
    mean=-2.0,
    sigma=0.6,
)

p2 = gaussian_histogram(
    mean=2.0,
    sigma=0.6,
)

A = np.column_stack(
    [p1, p2]
)

barycenter = ot.barycenter(
    A,
    cost,
    reg=1e-2,
)

linear_average = 0.5 * p1 + 0.5 * p2
```

The resulting distributions answer two different questions.

The arithmetic mixture asks:

> What distribution results from randomly selecting one source distribution?

The Wasserstein barycenter asks:

> What distribution lies geometrically between them under transportation cost?

This experiment visualizes that difference.

---

# Experiment 05 — Strategic belief geometry

This is the bridge from geometric statistics to multi-agent reasoning.

Suppose an opponent has one of \(K\) latent behavioral modes.

Represent our current estimate as

$$
\theta
=
(\theta_1,\ldots,\theta_K)
\in
\Delta^{K-1}.
$$

For \(K=3\), the entire belief state is visible inside a triangle.

For example:

$$
\theta
=
(
P(\text{cooperative}),
P(\text{opportunistic}),
P(\text{adversarial})
).
$$

Every observation updates the estimator:

$$
\hat\theta_t
=
T(a_{1:t}).
$$

---

## Simple Bayesian baseline

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

    total = posterior.sum()

    if total <= 0:
        raise ValueError(
            "Posterior has zero mass."
        )

    return posterior / total
```

Sequential inference:

```python
theta = np.array(
    [0.33, 0.34, 0.33]
)

trajectory = [theta.copy()]

for observation in observations:

    likelihood = observation_model[
        observation
    ]

    theta = bayesian_update(
        theta,
        likelihood,
    )

    trajectory.append(
        theta.copy()
    )
```

---

## Deceptive observations

Now suppose observations are generated strategically.

A deceptive agent may choose action \(a_t\) according to

$$
a_t
=
\arg\max_a
\mathbb E
\left[
U_j
\mid
\text{belief update caused by }a
\right].
$$

The observation process is therefore endogenous.

The inferential model

$$
P(a_t\mid\theta)
$$

is misspecified because

$$
P(a_t\mid\theta)
$$

also depends on what the other agent believes **we will infer from the action**.

This creates the research problem:

$$
P(
a_t
\mid
\theta_j,
b_j(\theta_i),
\text{strategic objective}
).
$$

The experiment asks:

> Which estimators recover most quickly after strategically selected misleading evidence?

Metrics include:

### Recovery time

$$
\tau_\delta
=
\min
\left\{
t:
d(
\hat\theta_t,
\theta^\star
)
<
\delta
\right\}.
$$

### Maximum belief displacement

$$
D_{\max}
=
\max_t
d(
\hat\theta_t,
\theta^\star
).
$$

### Cumulative belief error

$$
R_T
=
\sum_{t=1}^{T}
d^2(
\hat\theta_t,
\theta^\star
).
$$

### Strategic regret

If beliefs control actions,

$$
\operatorname{Regret}_T
=
\sum_{t=1}^{T}
\left[
U(a_t^\star)
-
U(a_t)
\right].
$$

This connects estimation quality to downstream decision quality.

---

# Visualizing the probability simplex

For three latent types,

$$
\theta_1+\theta_2+\theta_3=1.
$$

Map simplex coordinates into \(\mathbb R^2\):

```python
import numpy as np


SIMPLEX_VERTICES = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0],
    ]
)


def simplex_to_cartesian(theta):
    theta = np.asarray(theta)

    if not np.isclose(
        theta.sum(),
        1.0,
    ):
        raise ValueError(
            "Simplex coordinates must sum to one."
        )

    return theta @ SIMPLEX_VERTICES
```

Then plot the belief trajectory:

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_belief_trajectory(
    trajectory,
):
    trajectory = np.asarray(
        trajectory
    )

    xy = np.vstack(
        [
            simplex_to_cartesian(theta)
            for theta in trajectory
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

The resulting path provides a direct geometric representation of how evidence changes social belief.

---

# Benchmark matrix

The repository is organized around a matrix of controlled comparisons.

| Experiment             | Geometry                        | Estimator                 | Perturbation                 | Main question                                       |
| ---------------------- | ------------------------------- | ------------------------- | ---------------------------- | --------------------------------------------------- |
| Geometry of Influence  | Sphere / Hyperbolic / Euclidean | Mean / Median / Huber     | Contamination                | How does geometry change robustness?                |
| Intrinsic vs Extrinsic | Manifolds                       | Fréchet / projected mean  | Dispersion                   | When does ignoring geometry matter?                 |
| KDE Under Curvature    | Manifolds                       | KDE                       | Bandwidth / misspecification | When does local geometry affect density estimation? |
| Distribution Averaging | Wasserstein                     | Barycenter / mixture      | Distribution shift           | What does "average" mean for distributions?         |
| Strategic Beliefs      | Simplex                         | Bayes / robust / adaptive | Deception                    | How do misleading signals distort inferred intent?  |

---

# Evaluation philosophy

The project intentionally avoids ranking estimators using one scalar.

Every experiment should report at least four views.

## Statistical risk

$$
R
=
\mathbb E[
d^2(
\hat\theta,
\theta^\star
)
].
$$

## Variability

$$
\operatorname{Var}_d(
\hat\theta
).
$$

## Perturbation sensitivity

$$
S(\epsilon)
=
d(
T(P),
T(P_\epsilon)
).
$$

## Geometry of failure

Where does the estimator move?

How quickly?

Along which directions?

Does it degrade continuously?

Does it undergo a qualitative transition?

This last category is often hidden by aggregate metrics.

---

# Research hypotheses

The repository distinguishes measured results from hypotheses.

Initial hypotheses include:

### H1

The robustness ordering of location estimators depends on both contamination structure and geometry.

### H2

Extrinsic estimators approximate intrinsic estimators well in sufficiently local regimes, with disagreement increasing as distributions explore larger geometric scales.

### H3

Curvature interacts with contamination direction, producing anisotropic estimator sensitivity.

### H4

Estimator trajectories contain information about robustness that cannot be recovered from final scalar risk alone.

### H5

In strategic inference, robust belief estimators can outperform maximum-likelihood or naive Bayesian estimators when observations are intentionally selected to manipulate inference.

These are hypotheses to test, not conclusions assumed by the codebase.

---

# A more interesting notion of robustness

Classical robustness often asks:

> How much does the estimate change?

In structured spaces we can additionally ask:

> In what direction does it change?

Suppose

$$
v(z)
=
\operatorname{IF}_{\mathcal M}
(z;T,P)
\in
T_{\theta}\mathcal M.
$$

Then contamination produces an **influence vector field**

$$
z
\mapsto
v(z).
$$

This allows us to study:

* influence magnitude
* influence direction
* anisotropy
* alignment with curvature
* alignment with downstream decision boundaries

A scalar robustness statistic cannot capture these distinctions.

---

# Geometry-aware downstream risk

Statistical distance and decision relevance need not coincide.

Suppose an estimate controls a policy

$$
\pi(a\mid\hat\theta).
$$

Two errors

$$
\delta_1,\delta_2
$$

may have identical geometric magnitude

$$
\|\delta_1\|
=
\|\delta_2\|
$$

but dramatically different strategic consequences.

We therefore distinguish:

$$
R_{\text{stat}}
=
d^2(
\hat\theta,
\theta^\star
)
$$

from

$$
R_{\text{decision}}
=
V(\theta^\star)
-
V(
\pi(\hat\theta),
\theta^\star
).
$$

This becomes especially important in multi-agent systems.

---

# Reproducing experiments

Run one experiment:

```bash
python experiments/01_geometry_of_influence/run.py \
    --config experiments/01_geometry_of_influence/config.yaml
```

Run all experiments:

```bash
python scripts/reproduce_all.py
```

Generate a single figure:

```bash
python scripts/reproduce_figure_1.py
```

The repository should never require a notebook to reproduce a paper figure.

Notebooks are for exploration.

Scripts are the source of reproducible results.

---

# Results format

Every experiment writes long-form records.

```text
results/raw/geometry_of_influence.csv
```

Example schema:

```text
experiment
space
trial
seed
n
epsilon
estimator
estimate
error
objective
iterations
converged
```

Aggregated results are generated separately.

```text
results/aggregated/geometry_of_influence_summary.csv
```

Figures should be reproducible entirely from saved result files.

---

# Reproducibility

Every reported experiment stores:

```python
metadata = {
    "seed": seed,
    "space": space_name,
    "estimator": estimator_name,
    "sample_size": n,
    "contamination": epsilon,
    "config": config,
}
```

Randomness should never depend implicitly on execution order.

Prefer:

```python
rng = np.random.default_rng(seed)
```

to global random state whenever possible.

For trial \(j\), derive a deterministic seed:

```python
trial_seed = base_seed + trial
```

or use `SeedSequence` for larger experiments.

---

# Testing philosophy

Tests should verify mathematical properties whenever possible.

Example:

```python
import numpy as np


def test_frechet_mean_lies_on_sphere(
    sphere,
    sphere_sample,
):
    estimator = FrechetEstimator()

    result = estimator.fit(
        sphere_sample,
        sphere,
    )

    assert bool(
        sphere.belongs(
            result.point
        )
    )
```

Estimator equivariance tests are even more useful.

For a rotation \(R\),

$$
T(RX)
\approx
R\,T(X).
$$

```python
def test_rotation_equivariance(
    sphere,
    estimator,
    X,
    rotation,
):
    original = estimator.fit(
        X,
        sphere,
    ).point

    X_rotated = (
        rotation
        @ X.T
    ).T

    transformed = estimator.fit(
        X_rotated,
        sphere,
    ).point

    expected = rotation @ original

    error = sphere.metric.dist(
        transformed,
        expected,
    )

    assert error < 1e-5
```

This is more informative than testing only numerical outputs for a single fixed sample.

---

# Design principles

### 1. Mathematical objects first

A module should correspond to a statistical or geometric concept.

Avoid organization by notebook number or plotting convenience.

### 2. Experiments are hypotheses

Every experiment directory should contain a clear scientific question.

### 3. Geometry stays explicit

Do not silently convert manifold-valued observations into Euclidean coordinates.

### 4. Perturbations are first-class

Noise, contamination, deception, and misspecification should be explicit experimental objects.

### 5. Visualizations expose mechanism

Figures should help explain **why** an estimator behaves differently.

### 6. Reproducibility beats interactivity

Every important result should be reproducible from the command line.

### 7. Baselines stay simple

A strong experiment should survive comparison with simple estimators.

Complexity is justified only when the experiment demonstrates what it buys.

---

# What this repository is not

This is not intended to become a catalog of implementations such as:

```text
mean.py
median.py
kde.py
mle.py
map.py
```

The number of estimators implemented is not the contribution.

The project is centered on the comparative question:

$$
\boxed{
\text{What changes when the statistical geometry changes?}
}
$$

and its robust counterpart:

$$
\boxed{
\text{How does an estimator move when the data-generating process is perturbed?}
}
$$

---

# Roadmap

## v0.1 — Location

* [ ] Fréchet mean
* [ ] geometric median
* [ ] Huber location
* [ ] extrinsic mean
* [ ] sphere experiments
* [ ] contamination trajectories
* [ ] intrinsic risk
* [ ] estimator trajectory visualization
* [ ] approximate influence visualization

## v0.2 — Geometry

* [ ] Euclidean comparison
* [ ] hyperbolic experiments
* [ ] SPD geometry
* [ ] intrinsic/extrinsic comparison
* [ ] curvature × contamination sweeps

## v0.3 — Distribution estimation

* [ ] geometry-aware KDE
* [ ] bandwidth sweeps
* [ ] density misspecification experiments
* [ ] Wasserstein barycenters
* [ ] barycenter visualization

## v0.4 — Strategic inference

* [ ] simplex belief trajectories
* [ ] latent opponent types
* [ ] corrupted observations
* [ ] deceptive observation policies
* [ ] robust belief estimators
* [ ] recovery-time metrics
* [ ] downstream strategic regret

## v0.5 — Research benchmark

* [ ] standardized experiment registry
* [ ] reproducible benchmark suite
* [ ] uncertainty intervals
* [ ] publication-quality figure pipeline
* [ ] documented theoretical conjectures
* [ ] benchmark report

---

# Longer-term questions

The project is ultimately interested in several broader questions.

### Geometry × robustness

Can curvature qualitatively change robustness properties?

### Geometry × sample complexity

Does the appropriate geometric inductive bias reduce the number of observations required for reliable estimation?

### Geometry × misspecification

How costly is an incorrect metric?

### Robustness × strategic behavior

How should robust inference change when contamination is chosen by another optimizing agent?

### Estimation × decision-making

When do statistically small errors create strategically large losses?

### Structure × model scale

Can explicitly structured statistical state representations allow smaller reasoning systems to outperform substantially larger generic models in sequential strategic inference?

---

# Research direction

The long-term view of this project is:

$$
\text{observations}
\longrightarrow
\text{structured estimator}
\longrightarrow
\text{geometric belief state}
\longrightarrow
\text{decision}
$$

rather than

$$
\text{observations}
\longrightarrow
\text{unstructured context}
\longrightarrow
\text{prediction}.
$$

The underlying hypothesis is that geometry provides useful inductive structure for inference under uncertainty.

In passive statistical settings, this structure determines how observations should be aggregated.

In strategic settings, it determines how beliefs about other agents should evolve.

The same mathematical question appears in both:

> **What structure should an estimator preserve in order to remain reliable when observations are limited, corrupted, or strategically generated?**

---

# Citation

If this repository becomes useful in academic work, please cite:

```bibtex
@software{shape_of_estimation,
  title  = {The Shape of Estimation:
            Geometry, Robustness, and Statistical Inference
            Beyond Euclidean Assumptions},
  year   = {2026},
  note   = {Research software}
}
```

---

# License

MIT

---

## One-sentence summary

**The Shape of Estimation studies statistical estimators as geometric objects by visualizing how they move, fail, and recover when geometry, contamination, and information structure change.**
