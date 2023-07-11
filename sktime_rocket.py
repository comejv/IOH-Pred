# %% IMPORT
from utils import *

verbose("Importing required modules...")
from concurrent.futures import ThreadPoolExecutor
from os import listdir, makedirs
from os.path import basename, exists, join

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import SGDClassifier  # RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sktime.datatypes import check_raise
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.utils import mlflow_sktime
from progress.bar import ChargingBar
from progress.spinner import Spinner


def train_sgd(
    ifolder,
    epochs=4,
    n_training_cases=-1,
    init_case=111,
    rocket_threads=-1,
    fit_each=True,
    expected_type="pd-multiindex",
) -> (Pipeline, SGDClassifier):
    """Train the SGD classifier on the given training data after minirocket transformations.

    Args:
        ifolder (_type_): Path to a folder containing `train` and `labels` subfolder.
        epochs (int, optional): Number of epochs to train for. Defaults to 4.
        n_training_cases (int, optional): Number of training cases to use. Defaults to -1 for all cases.
        init_case (int, optional): Initial case to fit rocket on. Defaults to 111.
        rocket_threads (int, optional): Number of threads to use for minirocket. Defaults to -1 for all available.
        expected_type (str, optional): Type of input data for `sktime.datatypes.check_raise`. Defaults to "pd-multiindex".

    Returns:
        (Pipeline, SGDClassifier): Pipeline with minirocket and scaler, and trained classifier.
    """
    verbose("Loading training data")
    # LOAD TRAIN DATA
    X_train = pd.read_pickle(join(ifolder, "train", f"{init_case}.gz"))
    Y_train = pd.read_pickle(join(ifolder, "labels", f"{init_case}_labels.gz"))

    check_raise(X_train, mtype=expected_type)

    pipe = Pipeline(
        [
            ("rocket", MiniRocketMultivariate(n_jobs=rocket_threads)),
            ("scaler", StandardScaler()),
        ]
    )

    if not fit_each:
        verbose("Fitting pipeline")
        pipe.fit(X_train)

    # TRAIN MODEL
    class_weights = compute_class_weight(
        class_weight="balanced", classes=[False, True], y=Y_train
    )
    classifier = SGDClassifier(
        class_weight={False: class_weights[0], True: class_weights[1]}
    )

    files = listdir(join(ifolder, "train"))
    verbose("Starting training...")
    for epoch in range(epochs):
        n = 0
        for file in ChargingBar(
            f"Fitting epoch {epoch+1}/{epochs}", suffix="%(percent).1f%% - ETA %(eta)ds"
        ).iter(files):
            if file.endswith(".gz"):
                n += 1
                X_train = pd.read_pickle(join(ifolder, "train", file))
                Y_train = pd.read_pickle(
                    join(ifolder, "labels", file[:-3] + "_labels.gz")
                )

                check_raise(X_train, mtype=expected_type)

                if fit_each:
                    X_train_transform = pipe.fit_transform(X_train)
                else:
                    X_train_transform = pipe.transform(X_train)

                classifier.partial_fit(
                    X_train_transform, Y_train, classes=[False, True]
                )
                if n == n_training_cases:
                    break
    return pipe, classifier


# %% TRAIN

model = input("Model name: ")

if not exists(f"models/model_{model}/"):
    verbose("No model found, creating one now...")

    pipe, classifier = train_sgd(join(env.DATA_FOLDER, "ready"), epochs=3)

    mlflow_sktime.save_model(pipe, f"models/model_{model}/pipeline/")
    mlflow_sktime.save_model(classifier, f"models/model_{model}/classifier/")
else:
    verbose("Loading pipeline from models folder...")
    pipe = mlflow_sktime.load_model(f"models/model_{model}/pipeline/")
    verbose("Loading model from models folder...")
    classifier = mlflow_sktime.load_model(f"models/model_{model}/classifier/")
    verbose("Done.")


# %% TEST


def test_model(
    ifolder: str, n_files: int = 5, fit_each: bool = True
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run the model on the test data and return the predictions.
    ifolder is path to a folder where there are two subfolders :
        - cases
        - labels

    Args:
        ifolder (str): Path to the test data
        n_files (int, optional): Number of test files. Defaults to 5.
        fit_each (bool, optional): Whether to fit the pipeline on each file. Defaults to True.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: Labels and predictions
    """
    Y_test_l = []
    Y_scores_l = []
    n = 0
    with ChargingBar("Testing", max=n_files, suffix="%(percent).1f%% - ETA %(eta)ds") as bar:
        for file in listdir(join(ifolder, "test")):
            if file.endswith(".gz"):
                xpath = join(ifolder, "test", file)
                ypath = join(ifolder, "labels", file[:-3] + "_labels.gz")
                case = basename(xpath)
                X_test = pd.read_pickle(xpath)
                Y_test = pd.read_pickle(ypath)

                if fit_each:
                    X_test_transform = pipe.fit_transform(X_test)
                else:
                    X_test_transform = pipe.transform(X_test)

                Y_test_l.append(Y_test)
                Y_scores_l.append(classifier.decision_function(X_test_transform))
                n += 1
                bar.next()
                if n == n_files:
                    break
    bar.finish()

    Y_test = pd.concat(Y_test_l).reset_index(drop=True)
    Y_scores = np.concatenate(Y_scores_l)

    return Y_test, Y_scores


n_test = int(input("Number of test files: "))
Y_test, Y_scores = test_model(join(env.DATA_FOLDER, "ready"), n_files=n_test)


verbose("Computing model performances...")
# ROC Curve
fpr, tpr, auc_thresholds = roc_curve(Y_test, Y_scores)
roc_auc = auc(fpr, tpr)

# gmean
gmean = np.sqrt(tpr * (1 - fpr))
index = np.argmax(gmean)
gmean_threshold = auc_thresholds[index]
Y_pred_gmean = Y_scores > gmean_threshold
# f1 score
# f1_thresholds = np.arange(auc_thresholds[-1], auc_thresholds[0], 0.01)
f1_thresholds = auc_thresholds
f1_scores = []
for t in f1_thresholds:
    y_pred = Y_scores > t
    f1 = f1_score(Y_test, y_pred)
    f1_scores.append(f1)
f1_threshold = f1_thresholds[f1_scores.index(max(f1_scores))]

# Confusion matrix
cm = confusion_matrix(Y_test, Y_pred_gmean)
cm_norm = confusion_matrix(Y_test, Y_pred_gmean, normalize="true")
if cm.shape != (2, 2):
    cm = np.pad(
        cm,
        pad_width=((0, 1), (0, 1)),
        mode="constant",
        constant_values=0,
    )
    cm_norm = np.pad(
        cm_norm,
        pad_width=((0, 1), (0, 1)),
        mode="constant",
        constant_values=0,
    )

# %% PLOT
verbose("Plotting test results.")
# Create dataframe for the confusion matrices
df_cm = pd.DataFrame(cm, index=[False, True], columns=[False, True])
df_cm_norm = pd.DataFrame(cm_norm, index=[False, True], columns=[False, True])

# Create subplots
fig = plt.figure(layout="constrained", figsize=(15, 15))
gs = GridSpec(3, 2, figure=fig)
ax1l = fig.add_subplot(gs[0, 0])
ax1r = fig.add_subplot(gs[0, 1])
ax2l = fig.add_subplot(gs[1, 0])
ax2r = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[2, :])

# Create heatmap for raw counts
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", ax=ax1l)
ax1l.set_title("Confusion Matrix - Counts")
ax1l.set_xlabel("Predicted label")
ax1l.set_ylabel("True label")

# Create heatmap for percentages
sns.heatmap(df_cm_norm, annot=True, fmt=".2%", cmap="Greens", ax=ax1r)
ax1r.set_title("Confusion Matrix - Percentages")
ax1r.set_xlabel("Predicted label")
ax1r.set_ylabel("True label")

# Plot ROC and best threshold point
ax2l.plot(fpr, tpr, color="blue", label="ROC curve (AUC = %0.2f)" % roc_auc)
ax2l.plot([0, 1], [0, 1], color="navy", linestyle="--")

ax2l.set_xlabel("1 - Specificity")
ax2l.set_ylabel("Sensitivity")
ax2l.set_title("Receiver Operating Characteristic (ROC)")
ax2l.legend(loc="lower right")

# f1 score and gmean by threshold
ax2r.plot(f1_thresholds, f1_scores, label="F1 score", color="green")
ax2r.axvline(
    x=f1_threshold,
    color="lightgreen",
    label="Best f1 threshold %.2f : %.2f" % (f1_threshold, max(f1_scores)),
)
ax2r.plot(auc_thresholds, gmean, label="Gmean", color="blue")
ax2r.axvline(
    x=gmean_threshold,
    color="lightblue",
    label="Best gmean threshold %.2f : %.2f" % (gmean_threshold, max(gmean)),
)
ax2r.legend()
ax2r.set_xlabel("Threshold")
ax2r.set_ylabel("F1 score and gmean")
ax2r.set_xlim(
    [
        -1.5 * abs(min(f1_threshold, gmean_threshold)),
        1.5 * abs(max(f1_threshold, gmean_threshold)),
    ]
)
ax2r.set_title("F1 score and gmean by threshold")

# Plot predictions with gmean threshold
Y_test = Y_test.astype(int)
ax3.scatter(
    Y_test.index,
    Y_test,
    color="navy",
    label="True labels",
)
ax3.scatter(
    Y_test.index,
    Y_pred_gmean,
    color="cyan",
    label="Predictions with best gmean threshold",
)
ax3.fill_between(
    Y_test.index,
    0,
    Y_test,
    where=(Y_test == 1) & (Y_pred_gmean == 1),
    color="green",
    alpha=0.5,
)
true_positive_zone = mpatches.Patch(color="green", alpha=0.5)
current_handles, current_labels = ax3.get_legend_handles_labels()
ax3.legend(
    handles=current_handles.append(true_positive_zone),
    labels=current_labels.append("True positive"),
)
ax3.set_title(
    f"Predictions with gmean threshold : {Y_test.index[-1] * 20 / 3600:.2f}hrs"
)

plt.show()
