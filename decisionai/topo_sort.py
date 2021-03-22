from typing import Dict, Set, Tuple, List
from toposort import toposort_flatten, CircularDependencyError


def topo_sort_removing_errors(
    graph: Dict[str, Set[str]],
) -> Tuple[List[str], Set[str]]:
    """
    graph should be dict with values as sets:  {'b': set('a'), 'c': set('b'), 'a': set()}
    That says b depends on a, and c depends on b, and a depends on nothing.
    The returned value is a tuple of form (['a', 'b', 'c'], set()) where the first item
    is the order to evaluate, and the second item is the list of bad dependencies.
    """
    # toposort doesn't find self-dependencies, but they're a problem for us
    self_deps = {key for key, val in graph.items() if key in val}
    graph = {key: val for key, val in graph.items() if key not in self_deps}

    for maybe_bad_key, deps in graph.items():
        if any(v not in graph for v in deps if v != "t"):  # dependencies on t are OK
            return topo_sort_removing_errors(
                {key: val for key, val in graph.items() if key != maybe_bad_key}
            )

    try:
        return toposort_flatten(graph), set()
    except CircularDependencyError as e:
        circle = set(e.data.keys())
        graph = {key: val for key, val in graph.items() if key not in circle}
        return toposort_flatten(graph), set(circle)
