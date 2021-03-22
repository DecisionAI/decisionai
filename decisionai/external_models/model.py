from typing import List, Any
import numpy as np

from decisionai.errors import EquationError, visibly_wrapped_exception
from .internal_model import InternalModel, InternalClassifier, InternalRegressor

class Model:
    """The interface to external models used outside this subpackage (i.e. in
    Simulator and TreeEvaluator). Its main responsibility is handling the evaluation
    of expressions of the form "model.predict(...)" or "model.predict_probs(...)"
    when evaluating user-provided formulas.

    This is an abstract base class and should not be instantiated directly.
    """
    name: str
    _model: InternalModel

    def __init__(self, name, model):
        self.name = name
        self._model = model

    def _reshape_inputs(self, *args):
        """Given inputs to a user formula expression of the form
        MY_MODEL.predict(foo, bar, baz...) munges those arguments into a single
        array of the form expected by our InternalModel.
        """
        num_predictors = len(args)
        if self._model.n_features and self._model.n_features != num_predictors:
            raise EquationError(
                f"{self.name}.predict takes {self._model.n_features} features, but was"
                f" called with {num_predictors}"
            )
        one_variable_vals = args[0]
        output_size = one_variable_vals.size
        return np.stack(args, axis=-1).reshape([output_size, num_predictors])

class RegressionModel(Model):
    _model: InternalRegressor

    def predict(self, *args):
        model_input = self._reshape_inputs(*args)
        output_shape = args[0].shape
        if np.isnan(model_input).any():
            out = np.empty(output_shape)
            out[:] = np.nan
        else:
            try:
                predictions = self._model.predict(model_input)
            except Exception as e:
                raise visibly_wrapped_exception(e, 
                        f"calling {self.name}.predict_prob",
                        EquationError,
                )
            out = predictions.reshape(output_shape)
        return out

class ClassifierModel(Model):
    _model: InternalClassifier

    def predict(self, *args):
        """Returns predicted 0/1 labels *sampled* according to the probabilities
        given by self.predict_proba. (Note: we are not just selecting the most
        probable class for each instance.)
        """
        probs = self.predict_proba(*args)
        randomness = np.random.random_sample(probs.shape)
        return (randomness < probs).astype(int)

    def predict_proba(self, *args):
        model_input = self._reshape_inputs(*args)
        output_shape = args[0].shape
        # We're going to allow string variables in some cases, in which case
        # model_input will have dtype string or object (in case of mixed types)
        if (model_input.dtype.type not in (np.object_, np.str_)) and np.isnan(model_input).any():
            out = np.empty(output_shape)
            out[:] = np.nan
        else:
            try:
                probs = self._model.predict_proba(model_input)
            except Exception as e:
                raise visibly_wrapped_exception(e, 
                        f"calling {self.name}.predict_proba",
                        EquationError,
                )
            assert probs.shape[-1] == 2, "Expected 2 classes"
            # By convention we'll return the probability of the second class
            positive_probs = probs[:,1]
            out = positive_probs.reshape(output_shape)
        return out

