import ast
from typing import Optional, Tuple
import numpy as np
from contextlib import contextmanager

from decisionai.datasets import DatasetAdditionVar
from decisionai.policies import PolicyAttribute
from decisionai.errors import NeedInitialValueError, SilentError, CriticalError, visibly_wrapped_exception
from decisionai.variables import BaseVariable, Variable

from .provided_fns import OPERATORS, summed_join
from .randomness import sample, RANDOM_SAMPLING_FNS
from .errors import EvaluationError
from .synthetic_nodes import SimpleVarNode, ColumnNode
from .tree_utils import dataset_for_node


class TreeEvaluator(ast.NodeVisitor):
    """
    A NodeVisitor with some computation context (most importantly, the timestep
    in question) which resolves nodes to values. These will usually be ndarrays.
    """
    time: int
    policy_index: Optional[int]
    sim: 'Simulation'
    qv: BaseVariable
    # Shape that non-dataset expressions should evaluate to
    default_shape: Tuple[int]
    # Shape of output of enter(). If qv is a dataset variable, this will
    # be like default shape with an additional 'row' dimension. Otherwise, it
    # will be identical to default_shape.
    output_shape: Tuple[int]
    scalar_mode: bool
    # Whether we're inside a join() call
    join_mode: bool

    def __init__(self, t, policy_index, sim, qv):
        self.time = t
        self.policy_index = policy_index
        self.default_shape = ()
        if self.policy_index is None:
            self.default_shape += (sim.num_policies,)
        self.default_shape += (sim.num_sims,)
        self.output_shape = self.default_shape
        if isinstance(qv, DatasetAdditionVar):
            ds = sim.datasets[qv.parent_table_label]
            self.output_shape += (ds.n_rows,)
        self.sim = sim
        self.qv = qv
        self.scalar_mode = False
        self.join_mode = False
        super().__init__()

    def enter(self, root):
        res = self.visit(root)
        output_dim = len(self.output_shape)
        if res.shape == self.output_shape:
            return res
        elif res.ndim == output_dim - 1:
            # It must be that res has shape self.default_shape, and that self.output_shape
            # includes an extra n_rows dimension.
            return self._add_dataset_dimension(res)
        elif res.ndim == output_dim + 1:
            assert isinstance(self.qv, (PolicyAttribute, Variable))
            raise EvaluationError(f"Got extra dimension when evaluating {self.qv.name}."
                " When referring to a dataset variable in the definition of a "
                f"{self.qv.__class__.__name__}, the row dimension must be reduced"
                " using a function such as sum_from_dataset.")
        else:
            raise CriticalError(
                f"{self.qv.__class__.__name__} evaluated to shape {res.shape}."
                f" Output shape was {self.output_shape}. Default shape was {self.default_shape}."
            )

    @contextmanager
    def in_join_mode(self, col_dataset):
        # Yeah this won't handle nested joins, but that sounds insane anyways.
        self.join_mode = True
        # Keep a reference to the label of the dataset controlling the second arg to the 
        # join. It'll be used in various places that care about join mode.
        self._join_col_dataset_label = col_dataset
        try:
            yield None
        finally:
            self.join_mode = False
            self._join_col_dataset_label = None

    def _add_dataset_dimension(self, arr):
        """Precondition: arr passed _validate_visit_result_shape, i.e. it is
        either of shape self.output_shape or self.output_shape[:-1]
        """
        if arr.shape == self.output_shape:
            return arr
        assert arr.shape == self.output_shape[:-1]
        nrows = self.output_shape[-1]
        tile_counts = tuple([1 for _ in self.default_shape]) + (nrows,)
        ix = tuple(slice(None) for _ in self.default_shape) + (np.newaxis,)
        return np.tile(arr[ix], tile_counts)

    def _broadcasting_fn_call(self, fn, *args):
        """Manipulates args to have compatible shapes, then returns the result
        of calling the given function on them.
        """
        # Probably some possibility of weirdness if in scalar mode.
        if self.scalar_mode:
            return fn(*args)
        rowcounts = set()
        # Whether we've seen any args that have *two* row dimensions (this
        # would correspond to a cross-dataset comparison, which can only
        # occur inside a join)
        rowxrow = False
        for arg in args:
            shape = arg.shape
            if shape == self.default_shape:
                continue
            rows = shape[-1]
            rowcounts.add(rows)
            if arg.ndim == 4:
                assert self.join_mode, ("Results with >3 dimensions should only"
                        " be possible inside a join expression.")
                rowxrow = True
        if not rowcounts:
            # Everything is of the default shape. Nothing to do.
            return fn(*args)
        if len(rowcounts) == 1:
            # Modify everything to have the one row dimension size that appears
            rows = rowcounts.pop()
            if rowxrow:
                # In this scenario, we have at least one arg with a shape like
                # (npols, nsims, rowsA, rowsB). Therefore we want to ensure that
                # every arg has 4 dimensions, with the last two either being 1
                # or rowsA/rowsB respectively.
                def normalize(arg):
                    if arg.ndim == 4:
                        return arg
                    elif arg.ndim == 3:
                        # arg of shape (npols, nsims, rows). Add a penultimate dim.
                        return np.expand_dims(arg, -2)
                    elif arg.ndim == 2:
                        # arg of shape (npols, nsims). Add two final dims.
                        return np.expand_dims(arg, (-1, -2))
                    else:
                        # TODO: Would be nice to have an exception type we can
                        # raise that will not be swallowed and turned into a
                        # user-facing error. i.e. for cases where *our* code is
                        # doing something unexpected, rather than the user's formula
                        # doing something sus
                        raise ValueError(f"Got arg of unexpected shape: {arg.shape}")
                normalized_args = [normalize(arg) for arg in args]
                return fn(*normalized_args)


            # TODO: should probably try to unify this, _add_dataset_dimension,
            # and anywhere else we're doing this 'puffing up' pattern in one
            # place.
            tile_shape = tuple([1 for _ in self.default_shape])
            tile_shape += (rows,)
            # NB: It would be nice if we could *just* add a final dimension of size
            # 1 without also having to tile. This works almost everywhere, but causes
            # issues with evaluating external model methods, since these expect all
            # arguments to have the same shape.
            puffed_args = [
                np.tile(np.expand_dims(arg, -1), tile_shape)
                if arg.shape == self.default_shape else arg
                for arg in args
            ]
            return fn(*puffed_args)
        else:
            raise EvaluationError(f"Got incompatible row counts: {rowcounts}"
                    " - are you mixing different datasets in one operation?")

    def _validate_visit_result_shape(self, res):
        """The result of every visit_ call should either have shape self.default_shape
        or that shape plus a dataset (nrows) dimension. This verifies that this
        is the case. (Exception: if we're in 'scalar mode')
        """
        if self.scalar_mode and isinstance(res, (type(None), int, float)):
            return
        shape = res.shape
        if shape == self.default_shape:
            return
        if self.join_mode:
            # Inside a join, our results might have the default shape, or the
            # default shape plus a dataset dimension, or in a very rare corner
            # case, the default shape plus *two* dataset dimensions.
            assert shape[:len(self.default_shape)] == self.default_shape
            return
        assert len(shape) == len(self.default_shape) + 1, (
            f"Shape mismatch: shape={shape!r}, default_shape={self.default_shape!r}")
        assert shape[:-1] == self.default_shape
        # Last dimension can be any size.

    def visit(self, node):
        res = super().visit(node)
        try:
            self._validate_visit_result_shape(res)
        except AssertionError as e:
            msg = f"Result of visiting {node} failed validation. Error was: {str(e)}"
            raise RuntimeError(msg)
        return res

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        fn = OPERATORS[type(node.op)]
        return self._broadcasting_fn_call(fn, left, right)

    def visit_BoolOp(self, node):
        args = [self.visit(child) for child in node.values]
        fn = OPERATORS[type(node.op)]
        return self._broadcasting_fn_call(fn, *args)

    def visit_UnaryOp(self, node):
        return OPERATORS[type(node.op)](self.visit(node.operand))

    def visit_Compare(self, node):
        [op] = node.ops
        left = self.visit(node.left)
        [right_node] = node.comparators
        right = self.visit(right_node)
        fn = OPERATORS[type(op)]
        # Special case: if we're inside a join and one of our comparison operands
        # has the dimensionality of a dataset *other than* the dataset that
        # the second arg to join belongs to. In this case, we do a sort of
        # cartesian join to effect the comparison.
        if self.join_mode:
            # It would be nice if we could fold this and the _fancy_join_compare
            # logic into _broadcasting_fn_call, so we could get its benefit for
            # all node types inside a join, however a fundamental issue with that
            # is that _broadcasting_fn_call only sees the evaluated args to the
            # function, meaning it can't distinguish between an intra-dataset 
            # operation and an inter-dataset operation between two datasets that
            # coincidentally have the same row count.
            ds1 = dataset_for_node(node.left)
            ds2 = dataset_for_node(right_node)
            # Dataset other than the one referred to in second arg of join
            other_ds = None
            if ds1 is not None and ds1 != self._join_col_dataset_label:
                other_ds = ds1
            if ds2 is not None and ds2 != self._join_col_dataset_label:
                if other_ds is not None:
                    raise EvaluationError("Join involves *three* distinct datasets. That's too many.")
                other_ds = ds2
            if other_ds:
                return self._fancy_join_compare(left, right, ds1, ds2, fn, other_ds)
            
        return self._broadcasting_fn_call(fn, left, right)

    def _fancy_join_compare(self, left, right, ds1, ds2, fn, other_ds_label):
        """left and right are ndarrays to be compared, having respective shapes:
            (npols, nsims, ds1_rows)
            (npols, nsims, ds2_rows)
        (Where the final dimension may not be present if the corresponding value
        of ds1/ds2 is None)
        These arrays are not directly comparable via == operators and similar,
        but we effect the comparison by doing a cartesian product of the final
        dimensions.
        """
        # dataset corresponding to second arg of join
        other_ds = self.sim.datasets[self._join_col_dataset_label]
        # a second dataset involved in the mask expression of the join (the first arg)
        # we refer to this as the 'destination' ds because 99% of the time, this formula
        # belongs to a variable for this dataset. (The rare counterexample is if there's
        # a sum_from_dataset upstream, in which case this could be a 'global' variable)
        dest_ds = self.sim.datasets[other_ds_label]
        # Call the number of rows in dest and other d, o respectively.
        # Then for args of rowcount d, we want to blow them up from shape
        # (d,) to (d, o)
        # For args of shape (o,) we reshape to (1, o) so broadcasting can do its thing.
        # For args without a dataset dimension, we reshape to (1, 1).
        # (For simplicity, I'm dropping the sims/pols/time dimensions from the above)
        # Thus the result of evaluating this expression will have shape (d, o)
        # (Normally this would be unacceptable to _validate_visit_result_shape,
        # but we have a special dispensation for this case.)
        args = []
        for arg, ds_label in zip([left, right], [ds1, ds2]):
            # NB: It's *not* sufficient to inspect the arg's shape and see
            # if it matches d or o. Even if d and o are coincidentally equal,
            # we still want to blow up args corresponding to the dest dataset
            if ds_label == other_ds_label:
                # Not clear if this will work
                tile_shape = tuple(1 for _ in arg.shape) + (other_ds.n_rows,)
                puffed = np.tile(np.expand_dims(arg, -1), tile_shape)
            else:
                if arg.ndim == 3:
                    # A dataset variable for other_ds. Add a penultimate dim
                    # to match the dest_ds dimension.
                    puffed = np.expand_dims(arg, -2)
                elif arg.ndim == 2:
                    # A non-dataset variable (rare, but possible). Add two final dims.
                    puffed = np.expand_dims(arg, (-1, -2))
                else:
                    raise ValueError(f"Operand for {fn} had shape {arg.shape}"
                            " but expected to have either 2 or 3 dimensions.")
            args.append(puffed)
        res = fn(*args)
        return res

    def visit_Constant(self, node):
        # relevant for Python 3.7. ast.Num is replaced with ast.Constant in 3.8
        if self.scalar_mode:
            return node.value
        return np.full(self.default_shape, node.value)

    def visit_Num(self, node):
        if self.scalar_mode:
            return node.n
        return np.full(self.default_shape, node.n)

    def _reshape_column_data(self, dat):
        assert dat.ndim == 1
        # add some new axes in preparation for tiling
        ix = tuple(np.newaxis for _ in self.default_shape)
        ix += (slice(None),)
        puffed = dat[ix]
        tile_counts = self.default_shape + (1,)
        return np.tile(puffed, tile_counts)

    def visit_FunctionCallNode(self, node):
        # Special casing join fn because it has some tricky dynamics (possibly
        # should be parsed into a specialized node type?)
        if node.fn_name == 'join':
            return self._eval_join(node)
        evaled_args = [self.visit(arg_node) for arg_node in node.args]
        try:
            # TODO: This is sort of a hack. Should come up with a more principled
            # approach.
            if node.fn_name in RANDOM_SAMPLING_FNS:
                # For randomness functions, it's important that we proactively
                # expand the input args to the full dataset dimensionality. Otherwise,
                # we'll end up sampling a set of values for one row, then later
                # repeat those values across the row dimension.
                args = [self._add_dataset_dimension(arg) for arg in evaled_args]
                return sample(node.fn, args)
            return self._broadcasting_fn_call(node.fn, *evaled_args)
        except Exception as e:
            raise visibly_wrapped_exception(
                    e,
                    f"calling function {node.fn_name}",
                    EvaluationError,
            )

    def _eval_join(self, node):
        if len(node.args) != 2:
            raise EvaluationError(f"Join takes exactly 2 arguments, got {len(node.args)}.")
        mask_node, col_node = node.args
        ds = dataset_for_node(col_node)
        if ds is None:
            raise EvaluationError("Second argument to join() must be a "
                    "dataset variable or column name.")
        with self.in_join_mode(ds):
            mask, col = self.visit(mask_node), self.visit(col_node)
            try:
                # node.fn should be provided_functions.join
                return self._broadcasting_fn_call(node.fn, mask, col)
            except Exception as e:
                raise visibly_wrapped_exception(
                        e,
                        f"calling function {node.fn_name}",
                        EvaluationError,
                )

    def visit_SumJoinNode(self, node):
        ds = dataset_for_node(node.col_node)
        if ds is None:
            raise EvaluationError("Second argument to join() must be a "
                    "dataset variable or column name.")
        with self.in_join_mode(ds):
            mask, col = self.visit(node.mask_node), self.visit(node.col_node)
            return self._broadcasting_fn_call(summed_join, mask, col)

    def visit_ModelMethodNode(self, node):
        evaled_args = [self.visit(arg_node) for arg_node in node.args]
        try:
            return self._broadcasting_fn_call(node.method, *evaled_args)
        except Exception as e:
            raise visibly_wrapped_exception(
                    e,
                    f"calling {node.model.name}.{node.method_name}",
                    EvaluationError,
            )

    def visit_TNode(self, node):
        if self.scalar_mode:
            return self.time
        return np.full(self.default_shape, self.time)

    def _resolve_var_lookup(self, var, ix_):
        # Time indices before 0 get translated to 0
        ix = max(0, ix_)
        if ix > self.time:
            raise EvaluationError(f"Evaluating {self.qv.name} at t={self.time}"
                f" involved looking into the future (to t={ix})")
        res = self.sim.lookup_var(var.dotted_name, ix, self.policy_index)
        # The last condition here ensures that the nan relates to a lookup that's
        # aimed backwards in time (even if it gets clipped to 0). This help avoids
        # false positive 'need initial value' errors when the nan is the result of
        # some other set of circumstances (e.g. an expression involving 0/0).
        if ix == 0 and np.isnan(res).any() and ix_ < self.time:
            err = NeedInitialValueError(var)
            # We don't actually *raise* any exception here, because it's not fatal.
            var.record_exception(err)
        return res

    def visit_SimpleVarNode(self, node):
        return self._resolve_var_lookup(node.var, self.time+node.offset)

    def visit_RichlyIndexedVarNode(self, node):
        # This is not going to be very robust atm.
        # TODO: could be fancy and use a context manager!
        old_scalar = self.scalar_mode
        self.scalar_mode = True
        ix = self.visit(node.index_tree)
        self.scalar_mode = old_scalar
        if isinstance(ix, np.ndarray):
            # It's possible this resolves to a larger array (which may or may not
            # be uniform). But just handling the easy case for now.
            assert ix.size == 1
            ix = int(ix.flatten()[0])
        return self._resolve_var_lookup(node.var, ix)

    def visit_ColumnNode(self, node):
        ds = self.sim.datasets[node.dataset_name]
        dat = ds.df.loc[:, node.column].values
        return self._reshape_column_data(dat)

    def generic_visit(self, node):
        # Note that we've avoided having to deal with Name, Attribute, or 
        # Subscript nodes as a result of our earlier transformation.
        raise EvaluationError(f"Unexpected node type: {node!r}")


