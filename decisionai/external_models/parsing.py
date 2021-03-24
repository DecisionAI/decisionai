import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict
from typing import List
import pickle
import os
import tempfile

# We lazily import these modules to reduce our cold start time when firing up a new server instance
tfkerasmodels = None
sklearnbase = None
sklearnpipeline = None

from decisionai.errors import VisibleError, visibly_wrapped_exception
from .model import Model, ClassifierModel, RegressionModel
from .internal_model import TfClassifier, TfRegressor, SklearnClassifier, SklearnRegressor

_PICKLE_EXTENSIONS = {'.pickle', '.pkl'}
_TF_EXTENSIONS = {'.h5'}
_ALL_EXTENSIONS = _PICKLE_EXTENSIONS | _TF_EXTENSIONS

class ExternalModelDefinition(TypedDict, total=False):
    label: str
    localPath: str

class ModelError(VisibleError):
    """For errors with model specification (e.g. inappropriate file type)
    """
    error_type = ''

def parse_model_defn(model_info: ExternalModelDefinition) -> Model:
    label = model_info['label']
    if 'localPath' in model_info:
        saved_fname = model_info['localPath']
        with open(saved_fname, 'rb') as f:
            raw = f.read()
        orig_fname = saved_fname
    else:
        NotImplementedError('Try to access non-local path')
    _, ext = os.path.splitext(orig_fname)
    if ext in _PICKLE_EXTENSIONS:
        return _load_pickled_model(raw, label)
    elif ext in _TF_EXTENSIONS:
        # We require this extra layer of indirection because tf.keras.models.load_model
        # requires a file name. There's no equivalent fn that we can call with raw bytes
        # (or a StringIO object)
        with tempfile.NamedTemporaryFile(buffering=0) as f:
            f.write(raw)
            return _load_tf_model(f.name, label)
    else:
        raise ModelError(
            "Uploaded model should be either a pickled sklearn estimator"
            f" (allowed extesions: {', '.join(_PICKLE_EXTENSIONS)})"
            ", or a tf.keras model saved as an HDF5 file (allowed extensions:"
            f" {', '.join(_TF_EXTENSIONS)}). Got extension {ext!r} instead."
        )

def _load_pickled_model(raw_bytes, label):
    global sklearnbase, sklearnpipeline
    if sklearnbase is None:
        import sklearn.base as sklearnbase
        import sklearn.pipeline as sklearnpipeline
    try:
        model = pickle.loads(raw_bytes)
    except Exception as e:
        raise visibly_wrapped_exception(e,
                f"unpickling {label} model file",
                ModelError,
        )
    relevant_model = model
    if isinstance(model, sklearnpipeline.Pipeline):
        # If this is a pipeline, then use the last step of the pipeline to
        # determine whether the overall model is a classifier or regressor.
        name, relevant_model = model.steps[-1]

    if isinstance(relevant_model, sklearnbase.ClassifierMixin):
        cls, inner_cls = ClassifierModel, SklearnClassifier
        if hasattr(relevant_model, 'n_classes_'):
            n_classes = relevant_model.n_classes_
        elif hasattr(relevant_model, 'classes_'):
            n_classes = len(relevant_model.classes_)
        else:
            # Uh oh, can't figure out how many classes this model has. Crossing our fingers...
            n_classes = 2
        if n_classes != 2:
            raise ModelError("Only binary classifiers supported. Got classifier"
                    f" with {relevant_model.n_classes_} classes.")
    elif isinstance(relevant_model, sklearnbase.RegressorMixin):
        cls, inner_cls = RegressionModel, SklearnRegressor
    else:
        if isinstance(model, sklearnpipeline.Pipeline):
            raise ModelError("Last step of pipeline must be an sklearn classifier"
                    " or regressor, but was of class: " + str(relevant_model.__class__))
        else:
            raise ModelError("Serialized model must be an sklearn classifier"
                    " or regressor, but was of class: " + str(model.__class__))
    inner = inner_cls(model)
    return cls(label, inner)

def _load_tf_model(local_path, label):
    global tfkerasmodels
    if tfkerasmodels is None:
        import tensorflow.keras.models as tfkerasmodels
    try:
        model = tfkerasmodels.load_model(local_path, compile=False)
    except Exception as e:
        raise visibly_wrapped_exception(e,
                f"loading {label} model file",
                ModelError,
        )
    cls, inner_cls = RegressionModel, TfRegressor
    if hasattr(model, 'output_shape'):
        n_outputs = model.output_shape[-1]
        if n_outputs == 2:
            cls, inner_cls = ClassifierModel, TfClassifier
        elif n_outputs > 2:
            raise ModelError("Only binary classifiers supported. Got classifier"
                    f" with {n_outputs} classes.")

    inner = inner_cls(model)
    return cls(label, inner)

