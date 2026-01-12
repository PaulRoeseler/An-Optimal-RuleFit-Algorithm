#!/usr/bin/env python
# coding: utf-8
# author: Bo Tang

from collections import namedtuple
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn import tree

class optimalDecisionTreeClassifier:
    """
    Optimal regression tree with an integrated linear term (IORFA).
    Prediction: f(x) = x @ beta + gamma_leaf.
    """
    def __init__(self, max_depth=3, min_samples_split=2, alpha=0, warmstart=True, timelimit=600, output=True,
                 gamma_bounds=(-10000, 10000)):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.warmstart = warmstart
        self.timelimit = timelimit
        self.output = output
        self.gamma_bounds = gamma_bounds
        self.trained = False
        self.optgap = None
        self.feature_names_ = None
        self.feature_mins_ = None
        self.feature_maxs_ = None
        self.big_m_ = None
        self._beta = None
        self._gamma = None

        # node index
        self.n_index = [i+1 for i in range(2 ** (self.max_depth + 1) - 1)]
        self.b_index = self.n_index[:-2**self.max_depth] # branch nodes
        self.l_index = self.n_index[-2**self.max_depth:] # leaf nodes

    def fit(self, x, y):
        """
        fit training data
        """
        if hasattr(x, "columns"):
            self.feature_names_ = list(x.columns)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # data size
        self.n, self.p = x.shape
        self.feature_mins_ = np.min(x, axis=0)
        self.feature_maxs_ = np.max(x, axis=0)
        if self.output:
            print('Training data include {} instances, {} features.'.format(self.n,self.p))

        # solve MIP
        self.m, self.a, self.b, self.d, self.l, self.beta, self.gamma = self._buildMIP(x, y)
        
        if self.warmstart:
            self._setStart(x, y, self.a, self.b, self.d, self.l)

        self.m.optimize()
        self.optgap = self.m.MIPGap

        # get parameters
        self._a = {ind:self.a[ind].x for ind in self.a}
        self._b = {ind:self.b[ind].x for ind in self.b}
        self._d = {ind:self.d[ind].x for ind in self.d}
        self._beta = np.array([self.beta[p].x for p in range(self.p)])
        self._gamma = {t:self.gamma[t].x for t in self.l_index}

        self.trained = True

    def predict(self, x):
        """
        model prediction
        """
        if not self.trained:
            raise AssertionError('This optimalDecisionTreeClassifier instance is not fitted yet.')
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.p:
            raise ValueError(f'Expected {self.p} features, got {x.shape[1]}.')

        y_pred = np.zeros(x.shape[0], dtype=float)
        for i, xi in enumerate(x):
            t = 1
            while t not in self.l_index:
                if self._d.get(t, 0.0) < 0.5:
                    t = 2 * t + 1
                    continue
                split_val = sum(self._a[j, t] * xi[j] for j in range(self.p))
                if split_val + 1e-9 >= self._b[t]:
                    t = 2 * t + 1
                else:
                    t = 2 * t
            y_pred[i] = float(np.dot(xi, self._beta) + self._gamma[t])

        return y_pred

    def _buildMIP(self, x, y):
        """
        build MIP formulation for Optimal Decision Tree
        """
        # create a model
        m = gp.Model('m')

        # output
        m.Params.outputFlag = self.output
        m.Params.LogToConsole = self.output
        # time limit
        m.Params.timelimit = self.timelimit
        # parallel
        m.params.threads = 0

        # model sense
        m.modelSense = GRB.MINIMIZE

        # variables
        a = m.addVars(self.p, self.b_index, vtype=GRB.BINARY, name='a') # splitting feature
        b = m.addVars(self.b_index, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
        d = m.addVars(self.b_index, vtype=GRB.BINARY, name='d') # splitting option
        z = m.addVars(self.n, self.l_index, vtype=GRB.BINARY, name='z') # leaf node assignment
        l = m.addVars(self.l_index, vtype=GRB.BINARY, name='l') # leaf node activation
        beta = m.addVars(self.p, vtype=GRB.CONTINUOUS, name='beta')
        varkappa = m.addVars(self.n, self.l_index, vtype=GRB.CONTINUOUS, name='varkappa')
        gamma = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='gamma')
        lamb = m.addVars(self.n, vtype=GRB.CONTINUOUS, name='f')
        N = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='N') # leaf node samples

        # calculate minimum distance
        min_dis = self._calMinDist(x)

        feature_min = self.feature_mins_ if self.feature_mins_ is not None else np.min(x, axis=0)
        feature_max = self.feature_maxs_ if self.feature_maxs_ is not None else np.max(x, axis=0)
        feature_range = feature_max - feature_min
        max_range = float(np.max(feature_range)) if feature_range.size else 0.0
        eps_max = float(np.max(min_dis)) if len(min_dis) > 0 else 0.0
        big_m = max_range + eps_max
        if big_m <= 0:
            big_m = 1.0
        self.big_m_ = big_m

        objExp = gp.QuadExpr()

        self.Lower, self.Upper = self.gamma_bounds

        # add single terms using add
        for i in range(self.n):
            var = y[i] - gp.quicksum(x[i, p] * beta[p] for p in range(self.p)) - lamb[i]

            objExp.add(var * var) 
            
            m.addConstr(lamb[i] == gp.quicksum(varkappa[i, t] for t in self.l_index))
            
            for t in self.l_index:
                m.addConstr(self.Lower*z[i, t] <= varkappa[i, t])
                m.addConstr(varkappa[i, t] <= self.Upper*z[i, t])
                m.addConstr(self.Lower*(1-z[i, t]) <= gamma[t]-varkappa[i, t])
                m.addConstr(gamma[t]-varkappa[i, t] <= self.Upper*(1-z[i, t]))
                
            # gp.quicksum(gamma[t]*z[i, t] for t in self.l_index))

        complexity = gp.quicksum(d[t] for t in self.b_index)
        m.setObjective((1.0 / self.n) * objExp + self.alpha * complexity)

        # (16)
        m.addConstrs(z.sum('*', t) == N[t] for t in self.l_index)
        # (13) and (14)
        for t in self.l_index:
            left = (t % 2 == 0)
            ta = t // 2
            while ta != 0:
                if left:
                    m.addConstrs(gp.quicksum(a[j,ta] * (x[i,j] + min_dis[j]) for j in range(self.p))
                                 +
                                 big_m * (1 - d[ta])
                                 <=
                                 b[ta] + big_m * (1 - z[i,t])
                                 for i in range(self.n))
                else:
                    m.addConstrs(gp.quicksum(a[j,ta] * x[i,j] for j in range(self.p))
                                 >=
                                 b[ta] - big_m * (1 - z[i,t])
                                 for i in range(self.n))
                left = (ta % 2 == 0)
                ta //= 2

        # (8)
        m.addConstrs(z.sum(i, '*') == 1 for i in range(self.n))
        # (6)
        m.addConstrs(z[i,t] <= l[t] for t in self.l_index for i in range(self.n))
        # (7)
        m.addConstrs(z.sum('*', t) >= self.min_samples_split * l[t] for t in self.l_index)
        # (2)
        m.addConstrs(a.sum('*', t) == d[t] for t in self.b_index)
        # (3)
        m.addConstrs(b[t] <= gp.quicksum(feature_max[j] * a[j, t] for j in range(self.p)) for t in self.b_index)
        m.addConstrs(b[t] >= gp.quicksum(feature_min[j] * a[j, t] for j in range(self.p)) for t in self.b_index)
        # (5)
        m.addConstrs(d[t] <= d[t//2] for t in self.b_index if t != 1)

        return m, a, b, d, l, beta, gamma

    @staticmethod
    def _calMinDist(x):
        """
        get the smallest non-zero distance of features
        """
        min_dis = []
        for j in range(x.shape[1]):
            xj = x[:,j]
            # drop duplicates
            xj = np.unique(xj)
            # sort
            xj = np.sort(xj)[::-1]
            # distance
            dis = [1]
            for i in range(len(xj)-1):
                dis.append(xj[i] - xj[i+1])
            # min distance
            min_dis.append(np.min(dis) if np.min(dis) else 1)
        return min_dis

    def _setStart(self, x, y, a, b, d, l):
        """
        set warm start from CART
        """
        # train with CART
        if self.min_samples_split > 1:
            clf = tree.DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        else:
            clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
        
        clf.fit(x, y)

        # get splitting rules
        rules = self._getRules(clf)

        # fix branch node
        for t in self.b_index:
            # not split
            if rules[t].feat is None or rules[t].feat == tree._tree.TREE_UNDEFINED:
                d[t].start = 0
                b[t].start = 0
                for f in range(self.p):
                    a[f,t].start = 0
            # split
            else:
                d[t].start = 1
                b[t].start = rules[t].threshold
                for f in range(self.p):
                    if f == int(rules[t].feat):
                        a[f,t].start = 1
                    else:
                        a[f,t].start = 0

        # fix leaf nodes
        for t in self.l_index:
            # terminate early
            if rules[t].value is None:
                l[t].start = int(t % 2)
                # flows go to right
                # if t % 2:
                #     t_leaf = t
                #     while rules[t].value is None:
                #         t //= 2
                #     for k in self.labels:
                #         if k == np.argmax(rules[t].value):
                #             c[k, t_leaf].start = 1
                #         else:
                #             c[k, t_leaf].start = 0
                # nothing in left
                # else:
                #     for k in self.labels:
                #         c[k, t].start = 0
            # terminate at leaf node
            else:
                l[t].start = 1
                # for k in self.labels:
                #     if k == np.argmax(rules[t].value):
                #         c[k, t].start = 1
                #     else:
                #         c[k, t].start = 0

    def _getRules(self, clf):
        """
        get splitting rules
        """
        # node index map
        node_map = {1:0}
        for t in self.b_index:
            # terminal
            node_map[2*t] = -1
            node_map[2*t+1] = -1
            if node_map[t] == -1:
                continue
            # left
            l = clf.tree_.children_left[node_map[t]]
            node_map[2*t] = l
            # right
            r = clf.tree_.children_right[node_map[t]]
            node_map[2*t+1] = r

        # rules
        rule = namedtuple('Rules', ('feat', 'threshold', 'value'))
        rules = {}
        # branch nodes
        for t in self.b_index:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(clf.tree_.feature[i], clf.tree_.threshold[i], clf.tree_.value[i,0])
            rules[t] = r
        # leaf nodes
        for t in self.l_index:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(None, None, clf.tree_.value[i,0])
            rules[t] = r

        return rules
