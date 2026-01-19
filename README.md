# ORFA / IORFA — An Optimal RuleFit Algorithm

<p align="center">
  <a href="https://github.com/PaulRoeseler/An-Optimal-RuleFit-Algorithm/blob/main/An_Optimal_RuleFit_Algorithm.pdf">
    <img alt="Report" src="https://img.shields.io/badge/PDF-Report-2ea44f?logo=adobeacrobatreader&logoColor=white">
  </a>
</p>

This repository contains the implementation of **ORFA** (“Optimal RuleFit Algorithm”) and **IORFA** (“Integrated Optimal RuleFit Algorithm”).

- **ORFA**: an *optimal-tree* variant of Friedman & Popescu’s **RuleFit** that replaces greedy CART-style rule generation with **Optimal Regression Trees (ORT)** as rule generators.
- **IORFA**: a **single-step mixed-integer optimization (MIO)** formulation that *jointly* learns the tree structure and the downstream linear model.

---

## Why this project exists

**RuleFit** is attractive because it blends:
- a linear model (good for **main/linear effects**),
- with decision-tree rules (good for **interactions / non-linearities**).

However, classic RuleFit typically uses **greedy CART-style trees** to generate rules. To capture complex structure, this can lead to:

- **long rules**,  
- **very sparse rule indicators** (few samples satisfy a deep path),  
- **many trees ⇒ many candidate rules**.  

All of these make the resulting rule ensemble **harder to interpret and reason about**.

**ORFA** addresses this by using **Optimal Regression Trees (ORT)** as the rule generator. ORTs aim to achieve strong predictive performance with **shallower trees**, yielding **shorter, denser, more interpretable rules**.

**IORFA** goes one step further: instead of the usual two-stage pipeline (“fit tree → extract rules → fit linear model”), it uses an **integrated mixed-integer optimization** model that learns both components **together**.

---

## Method overview

### ORFA (two-stage “RuleFit-style” pipeline, but with optimal trees)

**Step A — Rule generation (tree stage)**  
Fit an **Optimal Regression Tree (ORT)** and extract the set of root-to-leaf path rules.

A typical rule has the form (conceptually):
- a conjunction of threshold comparisons along a path
- producing an indicator feature r_m(x) ∈ {0, 1}

**Step B — Sparse linear model over original features + rules**  
Fit a linear model on:
- the original features x,
- plus the rule indicator features r_m(x).

---

### IORFA (single-step integrated MIO)

Classic RuleFit/ORFA is *disaggregated*:
1) fit tree  
2) fit linear model using tree-generated rules

This can be suboptimal because the tree is not trained “knowing” what will be best for the downstream linear objective.

**IORFA** integrates both steps into **one optimization problem**:
- decision variables include:
  - linear coefficients for original features,
  - coefficients for leaf/rule assignments,
  - and tree assignment / split-selection variables,
- the objective is a regression loss with the tree and linear components coupled.

---

## Repository layout

- `rulefit_benchmark/`  
  Benchmark harness used to **benchmark ORFA** and baselines across datasets.

- `configs/`  
  YAML configuration files for benchmark runs (e.g., PMLB-style regression configs).

- `IORFA.py`  
  **Integrated Optimal RuleFit Algorithm** (MIO formulation).  
  Requires **Gurobi**.

---

## Installation

Create and activate a Python environment (conda/venv/uv all work).

Example (venv):
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows (PowerShell)
````

Install this repository:

```bash
pip install -e .
# or: pip install .
```

---

## Optional dependencies / solvers

### ORFA: Interpretable AI OptimalTrees (required for ORT rule generation)

Required for: **training Optimal Regression Trees (ORT)** and extracting rules.

Install + license:
- Interpretable AI installation & licensing: https://docs.interpretable.ai/stable/installation/
- OptimalTrees docs (overview): https://docs.interpretable.ai/stable/OptimalTrees/

### IORFA: Gurobi (required)

Required for: solving the **integrated mixed-integer optimization (MIO)** problem.

Install + license:
- Install Gurobi for Python (gurobipy): https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python
- Obtain a Gurobi license: https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license

---

## Running the benchmark suite (ORFA + baselines)

Run the benchmark using a config file, e.g.:

```bash
python -m rulefit_benchmark.cli --config configs/pmlb_regression.yaml
```

Outputs (path controlled by `outdir` in your config):

* `resolved_config.yaml` — exact configuration used
* `results.csv` — tidy results table (dataset × model × iteration)
* `results.pkl` — pandas DataFrame dump
* `results.parquet` — if `pyarrow` is available

---

## Running IORFA

`IORFA.py` is a research prototype of the integrated MIO formulation.

Basic checklist:

1. Install `gurobipy`
2. Confirm your Gurobi license is working (e.g., `gurobi_cl --license` if available)
3. Run `IORFA.py` according to its CLI / main function (see file header / code)

Because IORFA is an exact MIO approach, expect runtime to grow quickly with:

* number of samples,
* number of features,
* allowed tree depth / leaf count.

---

## References

* Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles. *The Annals of Applied Statistics*.
* Bertsimas, D., Dunn, J., & Paschalidis, A. (2017). Regression and classification using optimal decision trees.
* Olson, R. S., & La Cava, W. (2017). PMLB: a large benchmark suite for machine learning evaluation and comparison.
