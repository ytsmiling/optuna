import optuna
from optuna._experimental import experimental
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

try:
    import mlflow

    _available = True
except ImportError as e:
    _import_error = e
    _available = False
    mlflow = object


def _check_mlflow_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "MLflow is not available. Please install MLflow to use this feature. It can be "
            "installed by executing `$ pip install mlflow`. For further information, please "
            "refer to the installation guide of MLflow. (The actual import error is as "
            "follows: " + str(_import_error) + ")"
        )


@experimental("1.5.0")
class MLflowCallback(object):
    """Callback to track optuna trials with MLflow.

    This callback adds relevant information that is tracked by Optuna to MLflow.

    Example:

        Add MLflow callback to optuna optimization.

        .. testsetup::

            import pathlib
            import tempfile
            tempdir = tempfile.mkdtemp()
            YOUR_TRACKING_URI = pathlib.Path(tempdir).as_uri()

        .. testcode::

            import optuna
            from optuna.integration.mlflow import MLflowCallback

            def objective(trial):
                x = trial.suggest_uniform('x', -10, 10)
                return (x - 2) ** 2

            mlflc = MLflowCallback(
                tracking_uri=YOUR_TRACKING_URI,
                metric_name='my metric score',
            )

            study = optuna.create_study(study_name='my_study')
            study.optimize(objective, n_trials=10, callbacks=[mlflc])

        .. testcleanup::

            import shutil
            shutil.rmtree(tempdir)

        .. testoutput::
            :hide:
            :options: +NORMALIZE_WHITESPACE

            INFO: 'my_study' does not exist. Creating a new experiment

    Args:
        tracking_uri:
            The tracking server URI of MLflow. See the reference of `mlflow.set_tracking_uri
            <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri>`_
            for more details.

        experiment:
            Name of MLflow experiment to be activated. If not set ``study.study_name`` will be
            taken.
    """

    def __init__(self, tracking_uri=None, experiment=None):
        # type: (Optional[str], Optional[str]) -> None

        _check_mlflow_availability()

        self._tracking_uri = tracking_uri
        self._experiment = experiment

    def __call__(self, study, trial):
        # type: (optuna.study.Study, optuna.structs.FrozenTrial) -> None

        # This sets the tracking_uri for MLflow.
        if self._tracking_uri is not None:
            mlflow.set_tracking_uri(self._tracking_uri)

        # This sets the experiment of MLflow.
        if self._experiment is not None:
            mlflow.set_experiment(self._experiment)
        else:
            mlflow.set_experiment(study.study_name)

        with mlflow.start_run(run_name=trial.number):

            # This sets the metric for MLflow.
            trial_value = trial.value if trial.value is not None else float("nan")
            mlflow.log_metric('value', trial_value)

            # This sets the params for MLflow.
            mlflow.log_params(trial.params)

            # This sets the tags for MLflow.
            tags = {}  # type: Dict[str, str]
            tags["number"] = str(trial.number)
            tags["datetime_start"] = str(trial.datetime_start)
            tags["datetime_complete"] = str(trial.datetime_complete)
            # todo: change state name to human readable one
            tags["state"] = str(trial.state)
            tags["direction"] = str(study.direction)
            tags.update(trial.user_attrs)
            # todo: change distribution name to human readable one
            distributions = {
                (k + "_distribution"): str(v) for (k, v) in trial.distributions.items()
            }
            tags.update(distributions)

            # todo: set user attribute

            mlflow.set_tags(tags)
