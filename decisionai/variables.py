import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict
from typing import Optional, List, Any
import ast
from numbers import Number
import keyword

from .errors import Error, VisibleError

class BaseVariable:
    # This is used as a base for both Variables and policy attributes
    # This is the 'short' name (i.e. no dotted prefix with the dataset name)
    name: str
    errors: List[Error]
    initial: Any

    def __init__(self, name, initial=None, errors=None):
        self.name = name
        self.initial = initial
        self.errors = errors or []
        self._validate_name()

    @property
    def dotted_name(self):
        return self.name

    def __hash__(self):
        # Within a simulation, dotted names are required to be unique
        return hash(self.dotted_name)

    def _validate_name(self):
        """Verify that this has a valid name. If not, update self.errors.
        """
        record_err = lambda msg: self.errors.append(
                [self.dotted_name, 'short_name', 'Illegal name. ' + msg]
        )
        if self.dotted_name == 't':
            record_err('"t" is reserved')
        elif keyword.iskeyword(self.dotted_name):
            record_err(f'{self.name} is a Python keyword.')
        elif not self.name.isidentifier():
            record_err('Not a valid Python identifier.')

    def _parse_eqn(self, eqn: str) -> ast.AST:
        # Allow '^' as alias for exponentiation, and replace newlines with spaces.
        eqn = eqn.replace('^', '**').replace('\n', ' ')
        return ast.parse(eqn, mode='eval')

    def __repr__(self):
        r = f"{self.__class__.__name__} {self.name}"
        if self.initial is not None:
            r += f" (initial={self.initial!r})"
        return r

    def record_exception(self, err: VisibleError):
        err_list = [self.dotted_name, err.error_type, err.msg]
        # Don't record precisely the same error multiple times. For example,
        # without this check, we would record several 'need initial value' errors
        # for a variable defined like x = x[t-5]
        if err_list not in self.errors:
            self.errors.append(err_list)

    def set_poisoned(self, policy_index: Optional[int]):
        """If we detect that a formula has an irredeemable issue (e.g. a syntax error,
        a reference to a non-existent variable, a function call with the wrong type or
        number of arguments), we say that it is 'poisoned'. The Simulator will not try
        to evaluate poisoned trees.
        """
        raise NotImplementedError

    def is_poisoned(self, policy_index: Optional[int]):
        raise NotImplementedError

class Variable(BaseVariable):
    tree: Optional[ast.AST]
    _poisoned: bool

    def __init__(self, name, formula, initial=None, errors=None):
        super().__init__(name, initial, errors)
        self.name = name
        self.formula = formula
        self.initial = initial
        self.errors = []
        self._poisoned = False

        try:
            self.tree = self._parse_eqn(formula)
        except SyntaxError as err:
            self.errors.append([name, "equation", err.msg])
            self.tree = None


    def set_poisoned(self, policy_index: Optional[int]):
        self._poisoned = True

    def is_poisoned(self, policy_index: Optional[int]):
        return self._poisoned or (self.tree is None)




