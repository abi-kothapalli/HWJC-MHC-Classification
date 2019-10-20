import json
import time
from statistics import median
import numpy as np
import joblib


def save_checkpoint(config, auc_scores, accuracies, predictions, true_classes):
    checkpoint = {}

    checkpoint["experiment_name"] = config.experiment_name
    checkpoint["completed"] = len(auc_scores)
    checkpoint["runtime"] = time.time() - config.start_time + config.runtime
    checkpoint["data_file"] = config.data_file
    checkpoint["model"] = config.model
    checkpoint["target_feature"] = config.target_feature
    checkpoint["total_runs"] = config.runs
    checkpoint["test_size"] = config.test_size
    checkpoint["folds"] = config.folds
    checkpoint["n_jobs"] = config.n_jobs
    checkpoint["checkpoint_save_frequency"] = config.checkpoint_save_frequency

    checkpoint["auc_scores"] = auc_scores
    checkpoint["accuracies"] = accuracies
    checkpoint["prediction_array"] = predictions
    checkpoint["true_class_array"] = true_classes

    json_txt = json.dumps(checkpoint, indent=4)

    with open("checkpoints/checkpoint.json", "w") as file:
        file.write(json_txt)


def load_checkpoint(config):

    with open("checkpoints/checkpoint.json", "r") as file:
        checkpoint = json.load(file)

    config.experiment_name = checkpoint["experiment_name"]
    config.runtime = checkpoint["runtime"]
    config.data_file = checkpoint["data_file"]
    config.model = checkpoint["model"]
    config.target_feature = checkpoint["target_feature"]
    config.runs = checkpoint["total_runs"]
    config.test_size = checkpoint["test_size"]
    config.folds = checkpoint["folds"]
    config.n_jobs = checkpoint["n_jobs"]
    config.checkpoint_save_frequency = checkpoint["checkpoint_save_frequency"]
    config.save_checkpoints = True
    config.save_model = False

    return checkpoint["auc_scores"], checkpoint["accuracies"], checkpoint["prediction_array"], checkpoint["true_class_array"], checkpoint["completed"]


def save_model(config, model, features):

    name = f"{config.experiment_name}_{config.model}"
    file = f"models/trained/{name}.joblib"
    joblib.dump(model, file)

    info = {}

    info["experiment_name"] = config.experiment_name
    info["model_file"] = f"{name}.joblib"
    info["data_file"] = config.data_file
    info["features"] = features
    info["evaluation_mode"] = config.evaluation_mode
    info["model"] = config.model
    info["target_feature"] = config.target_feature
    info["total_runs"] = config.runs
    info["test_size"] = config.test_size
    info["folds"] = config.folds
    info["n_jobs"] = config.n_jobs

    json_txt = json.dumps(info, indent=4)

    with open(f"models/trained/{name}.json", "w") as file:
        file.write(json_txt)


def load_model(config):

    with open(f"models/trained/{config.load_model}", "r") as file:
        info = json.load(file)

    config.experiment_name = info["experiment_name"]
    config.model = info["model"]

    features = info["features"]
    target_feature = info["target_feature"]
    model = joblib.load(f"models/trained/{info['model_file']}")

    return model, features, target_feature


def save_output(config, prob_predictions, true_classes, auc_scores):

    file = f"output/{config.experiment_name}_{config.model}"

    if config.evaluation_mode == "max":
        predictions = prob_predictions[auc_scores.index(max(auc_scores))]
        true_values = np.asarray(true_classes[auc_scores.index(max(auc_scores))])
        np.savetxt(f"{file}_predictions.csv", predictions, delimiter=",", fmt="%f")
        np.savetxt(f"{file}_true_classes.csv", true_values, delimiter=",", fmt="%s")

    else:
        if config.runs % 2 == 0:
            newAUC = auc_scores.copy()
            newAUC.sort()
            approxAUC = newAUC[int(config.runs / 2)]
            predictions = prob_predictions[auc_scores.index(approxAUC)]
            true_values = np.asarray(true_classes[auc_scores.index(approxAUC)])
            np.savetxt(f"{file}_predictions.csv", predictions, delimiter=",", fmt="%f")
            np.savetxt(f"{file}_true_classes.csv", true_values, delimiter=",", fmt="%s")
        else:
            predictions = prob_predictions[auc_scores.index(median(auc_scores))]
            true_values = np.asarray(true_classes[auc_scores.index(median(auc_scores))])
            np.savetxt(f"{file}_predictions.csv", predictions, delimiter=",", fmt="%f")
            np.savetxt(f"{file}_true_classes.csv", true_values, delimiter=",", fmt="%s")