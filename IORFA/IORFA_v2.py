# iorfa_l1_fixed2.py
from __future__ import annotations

from collections import namedtuple
import numpy as np

import gurobipy as gp
from gurobipy import GRB
from sklearn import tree


class OptimalRegressionTree:
    """
    Optimal decision tree with integrated linear term (IORFA), L1 loss.

    Prediction: f(x) = x @ beta + gamma_leaf

    MILP:
      min (1/n) * sum_i |y_i - x_i^T beta - lambda_i| + alpha * sum_{branch t} d_t
      lambda_i = sum_{leaf t} varkappa_{i,t}
      varkappa_{i,t} = gamma_t * z_{i,t} (linearized with bounds)
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,          # min samples per active leaf (in MILP)
        alpha: float = 0.0,
        warmstart: bool = True,
        timelimit: float = 600.0,
        output: bool = True,
        gamma_bounds: tuple[float, float] = (-10000.0, 10000.0),
        beta_bounds: tuple[float, float] | None = None,
        eps_tie: float = 1e-9,
        threads: int = 0,
        seed: int = 0,
    ):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.alpha = float(alpha)
        self.warmstart = bool(warmstart)
        self.timelimit = float(timelimit)
        self.output = bool(output)
        self.gamma_bounds = (float(gamma_bounds[0]), float(gamma_bounds[1]))
        self.beta_bounds = beta_bounds
        self.eps_tie = float(eps_tie)
        self.threads = int(threads)
        self.seed = int(seed)

        self.trained = False
        self.optgap = None

        self.feature_names_ = None
        self.feature_mins_ = None
        self.feature_maxs_ = None
        self.big_m_ = None

        self._beta = None
        self._gamma = None
        self._a = None
        self._b = None
        self._d = None

        # nodes: 1..(2^(D+1)-1)
        self.n_index = [i + 1 for i in range(2 ** (self.max_depth + 1) - 1)]
        self.b_index = self.n_index[: -2**self.max_depth]
        self.l_index = self.n_index[-2**self.max_depth :]

        self.n = None
        self.p = None
        self.m = None

    def fit(self, x, y):
        if hasattr(x, "columns"):
            self.feature_names_ = list(x.columns)

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if x.ndim != 2:
            raise ValueError("x must be 2D (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be 1D (n_samples,)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same number of rows")

        self.n, self.p = x.shape
        self.feature_mins_ = np.min(x, axis=0)
        self.feature_maxs_ = np.max(x, axis=0)

        if self.output:
            print(f"Training data include {self.n} instances, {self.p} features.")

        m, a, b, d, z, l, beta, gamma, varkappa, lamb, r, u, N = self._buildMILP(x, y)

        # CRITICAL: finalize var indices before assigning Start
        m.update()

        if self.warmstart:
            self._setStart(
                x, y,
                a, b, d, z, l, beta, gamma, varkappa, lamb, r, u, N
            )

        m.optimize()

        self.m = m
        self.optgap = getattr(m, "MIPGap", None)

        self._a = {(j, t): a[j, t].X for j in range(self.p) for t in self.b_index}
        self._b = {t: b[t].X for t in self.b_index}
        self._d = {t: d[t].X for t in self.b_index}
        self._beta = np.array([beta[j].X for j in range(self.p)], dtype=float)
        self._gamma = {t: gamma[t].X for t in self.l_index}

        self.trained = True
        return self

    def predict(self, x):
        if not self.trained:
            raise AssertionError("This instance is not fitted yet.")

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, got {x.shape[1]}")

        y_pred = np.zeros(x.shape[0], dtype=float)

        for i, xi in enumerate(x):
            t = 1
            while t not in self.l_index:
                if self._d.get(t, 0.0) < 0.5:
                    t = 2 * t + 1
                    continue
                split_val = 0.0
                for j in range(self.p):
                    split_val += self._a.get((j, t), 0.0) * xi[j]
                if split_val + self.eps_tie >= self._b[t]:
                    t = 2 * t + 1
                else:
                    t = 2 * t
            y_pred[i] = float(np.dot(xi, self._beta) + self._gamma[t])

        return y_pred

    def _buildMILP(self, x: np.ndarray, y: np.ndarray):
        m = gp.Model("iorfa_l1")

        m.Params.OutputFlag = 1 if self.output else 0
        m.Params.LogToConsole = 1 if self.output else 0
        m.Params.TimeLimit = self.timelimit
        m.Params.Threads = self.threads
        m.ModelSense = GRB.MINIMIZE
        m.Params.Heuristics = 0 

        # variables
        a = m.addVars(self.p, self.b_index, vtype=GRB.BINARY, name="a")
        b = m.addVars(self.b_index, vtype=GRB.CONTINUOUS, name="b")
        d = m.addVars(self.b_index, vtype=GRB.BINARY, name="d")
        z = m.addVars(self.n, self.l_index, vtype=GRB.BINARY, name="z")
        l = m.addVars(self.l_index, vtype=GRB.BINARY, name="l")

        if self.beta_bounds is None:
            beta = m.addVars(self.p, vtype=GRB.CONTINUOUS, name="beta")
        else:
            lb, ub = self.beta_bounds
            beta = m.addVars(self.p, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="beta")

        Lower, Upper = self.gamma_bounds
        gamma = m.addVars(self.l_index, lb=Lower, ub=Upper, vtype=GRB.CONTINUOUS, name="gamma")

        varkappa = m.addVars(self.n, self.l_index, vtype=GRB.CONTINUOUS, name="varkappa")
        lamb = m.addVars(self.n, vtype=GRB.CONTINUOUS, name="lambda")

        r = m.addVars(self.n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="r")
        u = m.addVars(self.n, lb=0.0, vtype=GRB.CONTINUOUS, name="abs_r")

        N = m.addVars(self.l_index, lb=0.0, vtype=GRB.CONTINUOUS, name="N")

        # big-M
        min_dis = self._calMinDist(x)
        feature_min = self.feature_mins_
        feature_max = self.feature_maxs_
        max_range = float(np.max(feature_max - feature_min)) if self.p else 0.0
        eps_max = float(np.max(min_dis)) if len(min_dis) else 0.0
        big_m = max_range + eps_max
        if big_m <= 0:
            big_m = 1.0
        self.big_m_ = big_m

        # objective
        complexity = gp.quicksum(d[t] for t in self.b_index)
        m.setObjective((1.0 / self.n) * u.sum() + self.alpha * complexity)

        # lambda = sum varkappa
        m.addConstrs((lamb[i] == gp.quicksum(varkappa[i, t] for t in self.l_index) for i in range(self.n)), name="lambda_def")

        # residual + abs
        for i in range(self.n):
            m.addConstr(
                r[i] == y[i] - gp.quicksum(x[i, j] * beta[j] for j in range(self.p)) - lamb[i],
                name=f"res_def[{i}]",
            )
            m.addConstr(u[i] >= r[i], name=f"abs_pos[{i}]")
            m.addConstr(u[i] >= -r[i], name=f"abs_neg[{i}]")

        # varkappa linearization
        Lower, Upper = self.gamma_bounds
        for i in range(self.n):
            for t in self.l_index:
                m.addConstr(Lower * z[i, t] <= varkappa[i, t])
                m.addConstr(varkappa[i, t] <= Upper * z[i, t])
                m.addConstr(Lower * (1 - z[i, t]) <= gamma[t] - varkappa[i, t])
                m.addConstr(gamma[t] - varkappa[i, t] <= Upper * (1 - z[i, t]))

        # N[t] = sum z
        m.addConstrs((gp.quicksum(z[i, t] for i in range(self.n)) == N[t] for t in self.l_index), name="leaf_counts")

        # assign one leaf
        m.addConstrs((gp.quicksum(z[i, t] for t in self.l_index) == 1 for i in range(self.n)), name="assign_one_leaf")

        # leaf activation + min leaf size
        m.addConstrs((z[i, t] <= l[t] for i in range(self.n) for t in self.l_index), name="leaf_active_if_used")
        m.addConstrs((gp.quicksum(z[i, t] for i in range(self.n)) >= self.min_samples_split * l[t] for t in self.l_index), name="min_leaf_size")

        # split structure
        m.addConstrs((gp.quicksum(a[j, t] for j in range(self.p)) == d[t] for t in self.b_index), name="one_feat_if_split")
        m.addConstrs((b[t] <= gp.quicksum(feature_max[j] * a[j, t] for j in range(self.p)) for t in self.b_index), name="b_ub")
        m.addConstrs((b[t] >= gp.quicksum(feature_min[j] * a[j, t] for j in range(self.p)) for t in self.b_index), name="b_lb")
        m.addConstrs((d[t] <= d[t // 2] for t in self.b_index if t != 1), name="parent_active")

        # routing constraints
        for leaf in self.l_index:
            left = (leaf % 2 == 0)
            ta = leaf // 2
            while ta != 0:
                if left:
                    m.addConstrs(
                        (
                            gp.quicksum(a[j, ta] * (x[i, j] + min_dis[j]) for j in range(self.p))
                            + big_m * (1 - d[ta])
                            <= b[ta] + big_m * (1 - z[i, leaf])
                            for i in range(self.n)
                        )
                    )
                else:
                    m.addConstrs(
                        (
                            gp.quicksum(a[j, ta] * x[i, j] for j in range(self.p))
                            >= b[ta] - big_m * (1 - z[i, leaf])
                            for i in range(self.n)
                        )
                    )
                left = (ta % 2 == 0)
                ta //= 2

        return m, a, b, d, z, l, beta, gamma, varkappa, lamb, r, u, N

    @staticmethod
    def _calMinDist(x: np.ndarray):
        min_dis = []
        for j in range(x.shape[1]):
            xj = np.unique(x[:, j])
            if len(xj) <= 1:
                min_dis.append(1.0)
                continue
            xj = np.sort(xj)
            diffs = np.diff(xj)
            diffs = diffs[diffs > 0]
            min_dis.append(float(np.min(diffs)) if diffs.size else 1.0)
        return min_dis

    def _setStart(self, x, y, a, b, d, z, l, beta, gamma, varkappa, lamb, r, u, N):
        Lower, Upper = self.gamma_bounds
        min_dis = self._calMinDist(x)

        # CART warm start: IMPORTANT use min_samples_leaf to match MILP leaf size
        clf = tree.DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_split,
            random_state=self.seed,
        )
        clf.fit(x, y)
        rules = self._getRules(clf)

        # We'll keep start values in dicts (avoid reading Var.Start)
        d0 = {t: 0 for t in self.b_index}
        b0 = {t: 0.0 for t in self.b_index}
        split_feat = {}   # t -> feature index
        cart_thr = {}     # t -> CART threshold (only for partitioning; final b uses observed rmin)

        # initial splits from CART structure
        for t in self.b_index:
            rt = rules[t]
            if rt.feat is None or rt.feat == tree._tree.TREE_UNDEFINED:
                continue
            f = int(rt.feat)
            split_feat[t] = f
            cart_thr[t] = float(rt.threshold)
            d0[t] = 1

        # enforce parent-active in d0
        for t in self.b_index:
            if t != 1 and d0[t] == 1 and d0.get(t // 2, 0) == 0:
                d0[t] = 0
                split_feat.pop(t, None)
                cart_thr.pop(t, None)

        # compute feasible b0[t] = min right observed value; disable if strict-left infeasible
        left_vals = {t: [] for t in split_feat}
        right_vals = {t: [] for t in split_feat}

        for i in range(self.n):
            t = 1
            while t not in self.l_index:
                if d0.get(t, 0) == 0:
                    t = 2 * t + 1
                    continue
                f = split_feat[t]
                thr = cart_thr[t]
                val = x[i, f]
                if val + self.eps_tie >= thr:
                    right_vals[t].append(val)
                    t = 2 * t + 1
                else:
                    left_vals[t].append(val)
                    t = 2 * t

        for t in list(split_feat.keys()):
            f = split_feat[t]
            if len(left_vals[t]) == 0 or len(right_vals[t]) == 0:
                d0[t] = 0
                split_feat.pop(t, None)
                cart_thr.pop(t, None)
                continue
            rmin = float(np.min(right_vals[t]))
            lmax = float(np.max(left_vals[t]))
            if lmax + float(min_dis[f]) > rmin + 1e-12:
                # not feasible under strict-left encoding -> disable split
                d0[t] = 0
                split_feat.pop(t, None)
                cart_thr.pop(t, None)
                continue
            b0[t] = rmin

        # enforce parent-active again after disables
        for t in self.b_index:
            if t != 1 and d0.get(t, 0) == 1 and d0.get(t // 2, 0) == 0:
                d0[t] = 0
                split_feat.pop(t, None)
                cart_thr.pop(t, None)
                b0[t] = 0.0

        # beta start: least squares (clip if bounded)
        try:
            beta_start = np.linalg.lstsq(x, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_start = np.zeros(self.p, dtype=float)
        if self.beta_bounds is not None:
            lb, ub = self.beta_bounds
            beta_start = np.clip(beta_start, lb, ub)

        residuals = y - x @ beta_start

        # route using (d0, split_feat, b0)
        def route_model(xi):
            t = 1
            while t not in self.l_index:
                if d0.get(t, 0) == 0:
                    t = 2 * t + 1
                    continue
                f = split_feat[t]
                thr = b0[t]
                if xi[f] + self.eps_tie >= thr:
                    t = 2 * t + 1
                else:
                    t = 2 * t
            return t

        assignments = [route_model(x[i, :]) for i in range(self.n)]

        leaf_counts = {t: 0 for t in self.l_index}
        res_by_leaf = {t: [] for t in self.l_index}
        for i, lf in enumerate(assignments):
            leaf_counts[lf] += 1
            res_by_leaf[lf].append(float(residuals[i]))

        # if any nonempty leaf violates min size -> single leaf fallback (always feasible)
        if any((cnt > 0 and cnt < self.min_samples_split) for cnt in leaf_counts.values()):
            self._seed_single_leaf_start(x, y, beta_start, residuals, a, b, d, z, l, beta, gamma, varkappa, lamb, r, u, N)
            return

        # ===== Now write the starts to Gurobi vars (after model.update() has already run) =====
        # branch vars
        for t in self.b_index:
            if d0.get(t, 0) == 0:
                d[t].Start = 0
                b[t].Start = 0.0
                for f in range(self.p):
                    a[f, t].Start = 0
            else:
                d[t].Start = 1
                bt = float(b0[t])
                b[t].Start = bt
                fsel = split_feat[t]
                for f in range(self.p):
                    a[f, t].Start = 1 if f == fsel else 0

        # beta vars
        for j in range(self.p):
            beta[j].Start = float(beta_start[j])

        # gamma per leaf: median residual for L1, 0 for empty
        gamma0 = {}
        for t in self.l_index:
            if leaf_counts[t] > 0:
                g = float(np.median(res_by_leaf[t]))
            else:
                g = 0.0
            g = float(np.clip(g, Lower, Upper))
            gamma[t].Start = g
            gamma0[t] = g
            l[t].Start = 1 if leaf_counts[t] > 0 else 0
            N[t].Start = float(leaf_counts[t])

        # assignment + linked continuous vars
        for i in range(self.n):
            for t in self.l_index:
                z[i, t].Start = 0
                varkappa[i, t].Start = 0.0

            lf = assignments[i]
            z[i, lf].Start = 1
            varkappa[i, lf].Start = gamma0[lf]
            lamb[i].Start = gamma0[lf]

            ri = float(residuals[i] - gamma0[lf])
            r[i].Start = ri
            u[i].Start = abs(ri)

    def _seed_single_leaf_start(self, x, y, beta_start, residuals, a, b, d, z, l, beta, gamma, varkappa, lamb, r, u, N):
        Lower, Upper = self.gamma_bounds
        default_leaf = self.l_index[-1]
        gval = float(np.clip(np.median(residuals), Lower, Upper)) if self.n > 0 else 0.0

        for t in self.b_index:
            d[t].Start = 0
            b[t].Start = 0.0
            for f in range(self.p):
                a[f, t].Start = 0

        for j in range(self.p):
            beta[j].Start = float(beta_start[j])

        for t in self.l_index:
            l[t].Start = 1 if t == default_leaf else 0
            N[t].Start = float(self.n) if t == default_leaf else 0.0
            gamma[t].Start = gval if t == default_leaf else 0.0

        for i in range(self.n):
            for t in self.l_index:
                z[i, t].Start = 1 if t == default_leaf else 0
                varkappa[i, t].Start = gval if t == default_leaf else 0.0
            lamb[i].Start = gval
            ri = float(residuals[i] - gval)
            r[i].Start = ri
            u[i].Start = abs(ri)

    def _getRules(self, clf):
        node_map = {1: 0}
        for t in self.b_index:
            node_map[2 * t] = -1
            node_map[2 * t + 1] = -1
            if node_map[t] == -1:
                continue
            node_map[2 * t] = clf.tree_.children_left[node_map[t]]
            node_map[2 * t + 1] = clf.tree_.children_right[node_map[t]]

        rule = namedtuple("Rules", ("feat", "threshold", "value"))
        rules = {}
        for t in self.b_index:
            i = node_map[t]
            rules[t] = rule(None, None, None) if i == -1 else rule(clf.tree_.feature[i], clf.tree_.threshold[i], clf.tree_.value[i, 0])
        for t in self.l_index:
            i = node_map[t]
            rules[t] = rule(None, None, None) if i == -1 else rule(None, None, clf.tree_.value[i, 0])
        return rules
