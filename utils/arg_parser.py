import argparse
import time

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-en", "--experiment_name", type=str, default="classify", help="Experiment name")
    parser.add_argument("-df", "--data_file", type=str, default="cell_data.csv", help="Path to csv data file")

    parser.add_argument("-af", "--active_features", action="store_true", help="Get active features")

    parser.add_argument("-m", "--model", type=get_model, help="Algorithm to be used to develop model")
    parser.add_argument("-tf", "--target_feature", type=str, default="Class", help="Name of target feature if not \"Class\"")
    parser.add_argument("-r", "--runs", type=int, default=1000, help="Number of iterations to train model")
    parser.add_argument("-ts", "--test_size", type=float, default=0.2, help="Proportion of data to be used as test data")
    parser.add_argument("-f", "--folds", type=int, default=5, help="Number of folds to be used for cross validation in grid search algorithm")
    parser.add_argument("-nj", "--n_jobs", type=int, default=-1, help="Number of jobs to run in parallel during training - set to -1 to use all processors")
    parser.add_argument("-em", "--evaluation_mode", type=get_mode, default="median", help="Statistic used to display performance and save model: either \"median\" or \"max\"")

    parser.add_argument("-sc", "--save_checkpoints", action="store_true", help="Periodically save model checkpoints")
    parser.add_argument("-sf", "--checkpoint_save_frequency", type=int, default=20, help="Frequency of saving checkpoints, if saving model checkpoints")
    parser.add_argument("-lc", "--load_checkpoint", action="store_true", default=None, help="Load checkpoint")

    parser.add_argument("-sm", "--save_model", action="store_true", help="Save model upon completion")
    parser.add_argument("-lm", "--load_model", type=str, default=None, help="Name of json file to load model for evaluation on data file specified")

    args = parser.parse_args()

    args.start_time = time.time()
    args.runtime = 0

    if args.save_model and args.save_checkpoints:
        print("Warning: cannot save checkpoints if saving model - overriding save checkpoints option")
        args.save_checkpoints = False

    if args.load_checkpoint and args.load_model:
        print("Warning: load checkpoint and load model arguments received - program will only execute load checkpoint")
        print()
        args.load_model = False

    if args.load_checkpoint or args.load_model:
        return args

    if not args.active_features and args.model is None:
        if attempt:
            parser.error("Invalid model type received: for help please check the \"README.txt\" file")
        else:
            parser.error("Please enter a model type: for help please check the \"README.txt\" file")

    return args


attempt = False


def get_model(input):
    global attempt
    attempt = True
    input = input.lower()
    if input in ("knn", "l1", "l2", "mlp", "rf", "svm", "vc"):
        return input


def get_mode(input):
    input = input.lower()
    if input in("median", "max"):
        return input
    else:
        print("Warning: invalid evaluation mode received: will default to median")
        return "median"
