from decisionai.errors import VisibleError

# For now, treating these as fatal.
class PreprocessingError(VisibleError):
    error_type = 'equation'

class MissingEntityError(PreprocessingError):
    entity_type = None

    def __init__(self, name):
        self.msg = f"No {self.entity_type} named {name}"

class MissingDatasetError(MissingEntityError):
    entity_type = 'dataset'

class MissingVariableError(MissingEntityError):
    entity_type = 'variable'

class EvaluationError(VisibleError):
    error_type = 'equation'
