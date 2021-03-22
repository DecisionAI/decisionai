from decisionai.topo_sort import topo_sort_removing_errors

def test_topo_sort_missing():
    assert topo_sort_removing_errors({"b": set("a"), "c": set("b")}) == ([], set())


def test_topo_sort_self():
    # Self deps are identified in evalsim, and thus not reported in topo_sort_removing_errors
    assert topo_sort_removing_errors({"b": set("b")}) == ([], set())


def test_topo_sort_circular():
    assert topo_sort_removing_errors({"b": set("a"), "a": set("b")}) == ([], {"a", "b"})
