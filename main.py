import numpy as np
from utils.arg_parser import get_args
from utils.display import display_runtime
from data.data_loader import DataLoader
from evaluator.evaluator import Evaluator
from trainer.trainer import Trainer
from utils.save_load import load_checkpoint, load_model
import sys
import warnings
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

np.set_printoptions(formatter={"float_kind": "{:f}".format})


def main():

    config = get_args()

    if config.load_model is not None:
        model, features, target_feature = load_model(config)
        data_loader = DataLoader(config, split=False, pretrained=True)
        data_loader.setup(features, target_feature)
        evaluator = Evaluator(config)

        evaluator.evaluate_pretrianed(model, data_loader, target_feature)
        
        exit(0)

    if config.load_checkpoint:
        auc, acc, pred, classes, completed = load_checkpoint(config)

    data_loader = DataLoader(config, split=not config.active_features)
    evaluator = Evaluator(config)
    trainer = Trainer(config, data_loader, evaluator)

    if config.load_checkpoint:
        evaluator.set_checkpoint(auc, acc, pred, classes)
        trainer.set_completed(completed)

    trainer.train()

    if not config.active_features:
        print(f"AUC ({config.evaluation_mode}): {evaluator.get_auc()}")
        print(f"Accuracy ({config.evaluation_mode}): {evaluator.get_accuracy()}")

        evaluator.save(data_loader.getFeatures())

    display_runtime(config)


if __name__ == "__main__":
    main()