from typing import Any

# placeholder type alias for errors encoded as len-3 lists/tuples
# [ name, error_type, error_detail ]
# Error types: equation, short_name, initial, datasets, empty string (used in external_models.py).
# These are used on the frontend to determine which column to highlight.
# name: this will generally be a (dotted) BaseVariable name. May also be a dataset
# label, for errors when trying to load a dataset.
Error = Any

class VisibleError(Exception):
    """An exception that should be made visible to the user, by passing it 
    through as an "Error" object.
    """
    # TODO: consider using enum for these
    error_type = ''

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class EquationError(VisibleError):
    error_type = 'equation'

class NeedInitialValueError(VisibleError):
    error_type = 'initial'

    def __init__(self, var):
        try:
            varname = var.dotted_name
        except:
            varname = var
        self.msg = f"Initial value required for variable {varname}"

class SilentError(Exception):
    """An error during evaluation for which we don't want to surface a 
    user-visible error message (for example, if the exceptional state is
    the result of a problem that would have previously been reported)
    """
    pass

class CriticalError(Exception):
    """An error condition that we believe shouldn't be possible - i.e. indicative of
    a programming error. These do not get surfaced to the user, but do get logged
    to rollbar.
    """
    pass

def visibly_wrapped_exception(exception, activity, wrapper_cls=VisibleError):
    """Create a VisibleError which reports the occurrence of the given exception
    to the user in the course of the described activity.
    activity should be a string like "loading model X", "calculating value of y at t=0",
    etc.
    """
    # Just calling .capitalize() has the annoying effect of lowercasing everything
    # after the first character, which can mess up variable names etc.
    activity = activity and (activity[0].upper() + activity[1:])
    msg = f"{activity} triggered the following exception: {str(exception)!r}"
    return wrapper_cls(msg)

