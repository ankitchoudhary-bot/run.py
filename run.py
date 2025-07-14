import argparse
import json
import os
from kubernetes import client, config
from objective_fn import objective
import kubeflow.katib as katib

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--best_hyperparams", type=str, required=True)
args = parser.parse_args()

config.load_incluster_config()
# config.load_kube_config()

model_type = args.model_type.lower()
# model_type="pytorch"

search_spaces = {
    "sklearn": [
        {"name": "max_depth", "parameterType": "int", "feasibleSpace": {"min": "3", "max": "10"}},
        {"name": "n_estimators", "parameterType": "int", "feasibleSpace": {"min": "50", "max": "150"}}
    ],
    "xgboost": [
        {"name": "max_depth", "parameterType": "int", "feasibleSpace": {"min": "3", "max": "12"}},
        {"name": "learning_rate", "parameterType": "double", "feasibleSpace": {"min": "0.01", "max": "0.3"}}
    ],
    "pytorch": [
        {"name": "lr", "parameterType": "double", "feasibleSpace": {"min": "0.0001", "max": "0.1"}},
        {"name": "threshold", "parameterType": "double", "feasibleSpace": {"min": "0.5", "max": "4"}}
    ]
}

if model_type not in search_spaces:
    raise ValueError(f"Unsupported model type: {model_type}")

hyper_parameters = search_spaces[model_type]


parameters = {
    p["name"]: katib.search.double(
        min=p["feasibleSpace"]["min"],
        max=p["feasibleSpace"]["max"]
    )
    for p in hyper_parameters
}


katib_client = katib.KatibClient(namespace="admin")


name = "tune-experiment-ff"
katib_client.tune(
    name=name,
    objective=objective,
    parameters=parameters,
    objective_metric_name="accuracy",
    max_trial_count=12,
    parallel_trial_count=3,
    resources_per_trial={"cpu": "4","memory": "4Gi"},
    algorithm_name="tpe",
)

# [4] Wait until Katib Experiment is complete
katib_client.wait_for_experiment_condition(name=name)

# [5] Get the best hyperparameters.
best = katib_client.get_optimal_hyperparameters(name=name)
params = best.parameter_assignments
hp_dict = {p.name: float(p.value) for p in params}
print(hp_dict)
dir_path = os.path.dirname(args.best_hyperparams)
if dir_path:
    os.makedirs(dir_path, exist_ok=True)

with open(args.best_hyperparams, "w") as f:
    json.dump(hp_dict, f,indent=2)


