import numpy as np


class SamplingException(Exception):
    pass


def to_float(wrapped_fn):
    return lambda *args: wrapped_fn(*args).astype(float)


RANDOM_SAMPLING_FNS = {
    # As written the arguments to random functions can either be scalars OR arrays OR mixed:
    # uniform(1,2) or uniform(x,y) or uniform(1,y)
    # seems like this works without any special handling due to numpy broadcast rules, but we may need to be careful
    # Continuous Distributions
    "beta": np.random.beta,
    "chisquare": np.random.chisquare,
    "dirichlet": np.random.dirichlet,
    "exponential": np.random.exponential,
    "gamma": np.random.gamma,
    "geometric": np.random.geometric,
    "logistic": np.random.logistic,
    "lognormal": np.random.lognormal,
    "normal": np.random.normal,
    "uniform": np.random.uniform,
    "weibull": np.random.weibull,
    "zipf": np.random.zipf,
    # Discrete Distributions
    "poisson": to_float(np.random.poisson),
    "binomial": to_float(np.random.binomial),
    "randint": to_float(np.random.randint),
}


def check_randomness_shared_across_policies(sampling_fn_args):
    """
    test whether, for all items in sampling_fn_args, they are identical when comparing across 
    first axis (policies)
    
    sampling_fn_args is a list. It's length is number of args for relevant sampling_fn
    Items in list are arrays of shape (n_policies, max(1, n_sims))
    """
    similarity_by_arg = [
        np.isclose(s_arg_arr, s_arg_arr[0, :]).all() for s_arg_arr in sampling_fn_args
    ]
    return all(similarity_by_arg)


def sample(sampling_fn, sampling_fn_args):
    """
    sampling_fn: a function from among the values of RANDOM_SAMPLING_FNS above
    sample_fn_args: list of ndarrays to be passed as arguments to sampling_fn.
        arrays should have shape (npols, nsims[, nrows])
    """
    if any(np.isnan(arg).any() for arg in sampling_fn_args):
        # Numpy random sampling breaks on nans, so stop trying to evaluate
        raise SamplingException
    if check_randomness_shared_across_policies(sampling_fn_args):
        representative_sampling_args = [arg_arr[0, :] for arg_arr in sampling_fn_args]
        representative_draw = sampling_fn(*representative_sampling_args)
        n_policies = sampling_fn_args[0].shape[0]
        tilecounts = (n_policies,) + tuple(1 for _ in sampling_fn_args[0].shape[1:])
        out = np.tile(representative_draw, tilecounts)
        return out
    else:
        return sampling_fn(*sampling_fn_args)
