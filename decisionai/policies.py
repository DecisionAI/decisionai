import ast
from typing import List, Optional, Dict

from .variables import BaseVariable

"""A specification of a policy, as received by the API.
Must have a 'policy_name' key, mapping to the name of the policy.
All other key, value pairs correspond to attribute names and formulas, respectively.
"""
PolicyDefinition = Dict[str, str]

class PolicyAttribute(BaseVariable):
    trees: List[Optional[ast.AST]]
    # True if an equation is specified for every policy (even if that equation
    # has a syntax error or other issue). False otherwise.
    complete: bool
    # flags set per tree
    _poison_flags: List[bool]

    def __init__(self, name, eqns, initial=None, errors=None):
        super().__init__(name, initial, errors)
        self.complete = True
        self.trees = []
        self._poison_flags = []
        for eqn in eqns:
            if eqn == '':
                tree = None
                self.complete = False
            else:
                try:
                    tree = self._parse_eqn(eqn)
                except SyntaxError as e:
                    self.errors.append([self.name, "equation", e.msg])
                    tree = None
            self.trees.append(tree)
            poison = (tree is None)
            self._poison_flags.append(poison)

    def set_poisoned(self, policy_index: Optional[int]):
        if policy_index is None:
            indices = range(len(self.trees))
        else:
            indices = [policy_index]
        for ix in indices:
            self._poison_flags[ix] = True

    def is_poisoned(self, policy_index: Optional[int]):
        if policy_index is None:
            indices = range(len(self.trees))
        else:
            indices = [policy_index]
        return all(
            self._poison_flags[ix] or (self.trees[ix] is None)
            for ix in indices
        )

def policies_to_attributes(user_policies: List[PolicyDefinition],
) -> List[PolicyAttribute]:
    if len(user_policies) == 0:
        return []
    pol1 = user_policies[0]
    attr_names = pol1.keys() - {'policy_name'}
    name_to_eqns = {
            name: [pol.get(name, '') for pol in user_policies] 
            for name in attr_names
    }
    return [
        PolicyAttribute(name, eqns)
        for (name, eqns) in name_to_eqns.items()
    ]


def get_num_policies(attributes) -> int:
    if attributes:
        attribute_tree_lengths = [len(a.trees) for a in attributes]
        assert max(attribute_tree_lengths) == min(attribute_tree_lengths)
        return attribute_tree_lengths[0]
    else:
        return 1
