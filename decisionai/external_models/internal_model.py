from typing import Any

class InternalModel:
    """A wrapper providing a standard interface for interacting with user-provided models.
    """
    model: Any

    def __init__(self, model):
        self.model = model

    @property
    def n_features(self):
        """The number of features this model expects as input for prediction.
        May return None if we're not able to determine this by introspection
        of the model object.
        """
        return None

class InternalRegressor(InternalModel):
    def predict(self, X):
        """Takes data of shape (n_samples, n_features), and returns array of
        shape (n_samples,) or (n_samples, 1)
        """
        return self.model.predict(X)

class InternalClassifier:
    def predict_proba(self, X):
        """Takes data of shape (n_samples, n_features), and returns array of
        shape (n_samples, 2), with floats representing probabilities.
        """
        return self.model.predict_proba(X)

class TfModel(InternalModel):
    model: 'tf.keras.Model'

    @property
    def n_features(self):
        try:
            return self.model.input_shape[-1]
        except AttributeError:
            return None

class TfRegressor(TfModel, InternalRegressor):
    pass

class TfClassifier(TfModel, InternalClassifier):
    def predict_proba(self):
        # tf.keras models *do* have a predict_proba method, but it does the same
        # thing as .predict, and is deprecated
        return self.model.predict(X)

class SklearnModel(InternalModel):
    model: 'sklearn.base.BaseEstimator'

    @property
    def n_features(self):
        return getattr(self.model, 'n_features_', None)

class SklearnRegressor(SklearnModel, InternalRegressor):
    pass

class SklearnClassifier(SklearnModel, InternalClassifier):
    pass

