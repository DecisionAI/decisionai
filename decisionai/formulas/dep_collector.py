import ast
from typing import Set, Dict
from collections import defaultdict

from decisionai.variable import BaseVariable


class DependencyCollector(ast.NodeVisitor):
    """Walks an AST and collects named variables on which the expression depends.
    We say that X depends on Y iff there exists a t such that the calculation
    of X[t] requires Y[t]. If an equation involves complex index expressions,
    we may fail to detect a dependency - it's not something we can generally
    assess statically. For example, we know that 'x[t]' creates a dependency on
    x and that 'x[t-1]' does not. But we're not smart enough to answer the same
    question for something like 'x[where(t < 0, t, t-1)]' - we err on the side
    of assuming no dependency in these cases.

    Note: doesn't collect t itself - all formulas depend on t, at least implicitly.
    """
    deps: Set[str]

    def __init__(self):
        super().__init__()
        self.deps = set()

    def visit_SimpleVarNode(self, node):
        if node.offset == 0:
            self.deps.add(node.var.dotted_name)

class LagCollector(ast.NodeVisitor):
    """For each variable, tracks the deepest past reference to that variable
    among all formulas. e.g. if we have the variables and formulas...
    - x = x[t-1] + 1
    - y = x[t-2]
    - z = x[t-1] + z[t-3]

    Then, after processing all of these, self.lags will look like
    { x: 2, y: 0, z: 3 }

    Used as part of our optimization for keeping only a limited rolling window of
    timestep data for datasets to reduce memory usage.
    """
    # Maps vars to the furthest offset observed for that var
    lags: Dict[BaseVariable, int]

    def __init__(self):
        super().__init__()
        self.lags = defaultdict(lambda: 0)

    def visit_SimpleVarNode(self, node):
        self.lags[node.var] = max(
                self.lags[node.var],
                -node.offset
        )

    def visit_RichlyIndexedVarNode(self, node):
        # If we have an expression like x[t - y], or x[t - where(y==0, 1, 2)], we
        # throw up our hands and take the pessimistic view that the reference might
        # extend arbitrarily far into the past.
        self.lags[node.var] = float('inf')
