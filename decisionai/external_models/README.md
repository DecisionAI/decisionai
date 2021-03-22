`parse_model_defn` is the main entry point, and returns a `Model` object from a
user-provided json specification.
Model is the interface with which other parts of the code (`Simulator`,
and `TreeEvaluator`) interact. It handles the evaluation of expressions
of the form "MODEL.predict(foo, bar)" in user-defined formulas.

Under the hood, Models do not interact directly with the deserialized model
objects that we load from pickle/hdf5 files, but instead do so via `InternalModel`,
a layer that provides a uniform interface for all the model objects we support
(currently sklearn estimators and tf.keras models - maybe others later).

