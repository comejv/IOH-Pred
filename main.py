from argparse import ArgumentParser
from sys import argv
from os.path import join, exists
from sktime.utils import mlflow_sktime

from utils import *

parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--download",
    action="store_true",
    help="download [-n] cases from VitalDB that have all tracks listed in env.json.",
)
parser.add_argument(
    "-p",
    "--preprocess",
    action="store_true",
    help=f"preprocess [-n] cases from {env.DATA_FOLDER}vital and pickle them to {env.DATA_FOLDER}preprocessed/",
)
parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="don't skip already preprocessed data, process it again",
)
parser.add_argument(
    "-t",
    "--transpose",
    action="store_true",
    help="transpose [-n] cases from the preprocessed data into the format required by Rocket and Sklearn (windows of env.WINDOW_SIZE seconds)"
    "and pickle them to {env.DATA_FOLDER}ready/",
)
parser.add_argument(
    "-l",
    "--label",
    action="store_true",
    help=f"create label dataframes shifted by env.PRED_WINDOW minutes to train the model and pickle them to {env.DATA_FOLDER}ready/labels/",
)
parser.add_argument(
    "-s",
    "--train_sgd",
    action="store_true",
    help=f"train the SGD model on [-n] cases from {env.DATA_FOLDER}ready/test",
)
parser.add_argument(
    "-e",
    "--epochs",
    action="store",
    type=int,
    default=4,
    help="number of epochs to train the model",
)
parser.add_argument(
    "-T",
    "--test_sgd",
    action="store_true",
    help="test the model on [-n] cases from {env.DATA_FOLDER}test/",
)
parser.add_argument(
    "-n",
    "--max-number",
    action="store",
    type=int,
    choices=range(-1, 6388),
    default=10,
    help="max number of cases to apply the chosen functions to, -1 for all cases, defaults to 10",
    metavar="-1..6388",
)

args = parser.parse_args()
if len(argv) == 1:
    parser.print_help()
    exit(0)

try:
    import create_dataset as cd
    import feature_extraction as fe
    import plotting as pl
    import sktime_rocket as skr
except ImportError as e:
    perror("Import error(s) :", e.msg)
    solve_imports = binput("Would you like to solve the import errors? (y/n)")
    if solve_imports:
        init_venv_pip()


if args.download:
    cases = cd.find_cases(env.TRACKS)
    cd.download_cases(env.TRACKS, cases, max_cases=args.max_number)
if args.preprocess:
    cd.folder_preprocessing(
        ifolder=join(env.DATA_FOLDER, "vital"),
        ofolder=join(env.DATA_FOLDER, "preprocessed"),
        force=args.force,
        N=args.max_number,
    )
if args.transpose:
    fe.multithreaded_transpose(
        ifolder=join(env.DATA_FOLDER, "preprocessed"),
        ofolder=join(env.DATA_FOLDER, "ready", "cases"),
        tf=env.WINDOW_SIZE,
        n_files=args.max_number,
    )
if args.label:
    fe.multithreaded_label_events(
        ifolder=join(env.DATA_FOLDER, "ready", "cases"),
        ofolder=join(env.DATA_FOLDER, "ready", "labels"),
    )
model = None
if args.train_sgd:
    pbold("Model name ? ", end="")
    model = input()

    if not exists(f"models/model_{model}/"):
        print("No model found, creating one now...")
        pipe, classifier = skr.train_sgd(
            ifolder=join(env.DATA_FOLDER, "ready"),
            epochs=args.epochs,
            n_training_cases=args.max_number,
        )
        mlflow_sktime.save_model(pipe, f"models/model_{model}/pipeline/")
        mlflow_sktime.save_model(classifier, f"models/model_{model}/classifier/")
    else:
        pwarn("Model already exists, skipping training...")
if args.test_sgd:
    if not model:
        pbold("Model name ? ", end="")
        model = input()
    if not exists(f"models/model_{model}/"):
        perror("Model does not exist. Train it first with the --train_sgd flag.")
        exit(1)

    print("Loading pipeline from models folder...")
    pipe = mlflow_sktime.load_model(f"models/model_{model}/pipeline/")
    print("Loading model from models folder...")
    classifier = mlflow_sktime.load_model(f"models/model_{model}/classifier/")
    print("Done.")

    Y_test, Y_scores = skr.test_model_multi(
        join(env.DATA_FOLDER, "ready"), pipe, classifier, n_files=args.max_number
    )

    stats = skr.model_stats(Y_test, Y_scores)

    print_table(
        [model, stats.roc_auc, max(stats.f1_scores), max(stats.gmean)],
        ["model", "roc_auc", "best f1", "best gmean"],
    )
