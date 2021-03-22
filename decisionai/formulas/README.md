This sub-package deals with the analysis, transformation, and evaluation of the abstract syntax trees that we compile for user-supplied variable formulas, such as `where(x[t-1] > 0, y[t], sum_from_dataset(MY_DATA.column_1))`.

Our handling of formulas proceeds in the following steps:
1. We parse the source using `ast.parse`.
2. We transform the tree, replacing expressions referring to variables or dataset columns with more abstract representations, allowing us to deal with variations in surface details in one place. This step also has the side effect of identifying and reporting a number of error conditions, such as references to non-existent variables or use of disallowed syntax. See `transformer.py` for details.
3. `DependencyCollector` walks each tree to identify which variables depend on which others. (This ends up being trivial under the transformations performed in the previous step.)
4. We use `TreeEvaluator` to resolve trees to actual arrays of values. This is repeated many times, for each variable, timestep, simulation, and (sometimes) policy.

The `Simulator` class (in `../evalsim.py`) steers all of these steps.

Each of the above steps can uncover errors. These are caught at the Simulator level, and typically lead to 1) recording a user-visible error message, and 2) marking the corresponding tree as "poisoned", meaning that it will subsequently be ignored, and its remaining values left as nans.
