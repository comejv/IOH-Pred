from argparse import ArgumentParser
from sys import argv
from os.path import join

from utils import *

parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--download",
    action="store_true",
    help="download cases from VitalDB that have all tracks listed in env.json.",
)
parser.add_argument(
    "-p",
    "--preprocess",
    action="store",
    default=None,
    help=f"preprocess the data from {env.DATA_FOLDER}vital and pickle it to {env.DATA_FOLDER}preprocessed/",
)
parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    default=False,
    help="don't skip already preprocessed data, process it again",
)
parser.add_argument(
    "-t",
    "--transpose",
    action="store",
    default=None,
    help="transpose cases from the preprocessed data into the format required by Rocket and Sklearn (windows of env.WINDOW_SIZE seconds)"
    "and pickle them to {env.DATA_FOLDER}ready/",
)
parser.add_argument(
    "-l",
    "--label",
    action="store",
    default=None,
    help=f"create label dataframes shifted by env.PRED_WINDOW minutes to train the model and pickle them to {env.DATA_FOLDER}ready/labels/",
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
    cd.download_cases(
        env.TRACKS, cases, max_cases=args.max_number
    )
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
