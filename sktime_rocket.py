from concurrent.futures import ThreadPoolExecutor
from os import listdir, makedirs
from os.path import exists, join

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sktime.datatypes import check_raise
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.utils import mlflow_sktime

from utils import *

model = input("Model name: ")

if not exists(f"models/model{model}/"):
    verbose("No model found, creating one now...")
    verbose("Loading training data")
    # LOAD TRAIN DATA
    X_train = (
        pd.read_pickle(join(env.DATA_FOLDER, "ready", "cases", "111.gz"))
        # .groupby("window")
        # .filter(lambda group: len(group) == 2000)
    )
    Y_train = pd.read_pickle(join(env.DATA_FOLDER, "ready", "labels", "111_labels.gz"))
    # remaining_windows = X_train.index.get_level_values("window").unique()
    # Y_train = Y_train[Y_train.index.isin(remaining_windows)]

    try:
        check_raise(X_train, mtype="pd-multiindex")
    except Exception as e:
        print("Training data has the wrong type (expected multi-index dataframe).")
        raise e

    verbose("Fitting rocket transformer...")
    minirocket_multi = MiniRocketMultivariate(n_jobs=4)
    minirocket_multi.fit(X_train)
    mlflow_sktime.save_model(minirocket_multi, f"models/model{model}/rocket/")
    verbose("Transforming data...")
    X_train_transform = minirocket_multi.transform(X_train)

    scaler = StandardScaler(with_mean=False)
    X_train_scaled_transform = scaler.fit_transform(X_train_transform)
    mlflow_sktime.save_model(scaler, f"models/model{model}/scaler/")

    # TRAIN MODEL
    verbose("Training model...")
    # sgdclass = SGDClassifier()
    # sgdclass.partial_fit()
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_scaled_transform, Y_train)

    # classifier.score(X_test_scaled_transform, Y_test)
    makedirs("models", exist_ok=True)
    mlflow_sktime.save_model(classifier, f"models/model{model}/classifier/")
    verbose("Done. Model, rocket and scaler saved in models folder.")
else:
    verbose("Loading model from models folder...")
    classifier = mlflow_sktime.load_model(f"models/model{model}/classifier/")
    minirocket_multi = mlflow_sktime.load_model(f"models/model{model}/rocket/")
    scaler = mlflow_sktime.load_model(f"models/model{model}/scaler")
    verbose("Done.")


def test_model(x_test_path: str, y_test_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Run the model on the test data and return the predictions.

    Args:
        x_test_path (str): Path to the test data

    Returns:
        tuple[pd.DataFrame, np.ndarray]: Labels and predictions
    """
    verbose("Loading test data...")
    X_test = pd.read_pickle(x_test_path)
    Y_test = pd.read_pickle(y_test_path)

    verbose("Transforming test data with rocket...")
    X_test_transform = minirocket_multi.transform(X_test)
    X_test_scaled_transform = scaler.transform(X_test_transform)

    verbose("Classifying test data...")
    return Y_test, classifier.decision_function(X_test_scaled_transform)


def multithread_test_model(
    ifolder: str, n_files: int = 5
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run the model on the test data and return the predictions.
    ifolder is path to a folder where there are two subfolders :
        - cases
        - labels

    Args:
        ifolder (str): Path to the test data

    Returns:
        tuple[pd.DataFrame, np.ndarray]: Labels and predictions
    """
    xfiles = []
    yfiles = []
    n = 0
    for file in listdir(join(ifolder, "cases")):
        if file.endswith(".gz"):
            xfiles.append(join(ifolder, "cases", file))
            yfiles.append(join(ifolder, "labels", file[:-3] + "_labels.gz"))
            n += 1
            if n == n_files:
                break

    with ThreadPoolExecutor(max_workers=env.CORES) as executor:
        futures = executor.map(test_model, xfiles, yfiles)
    Y_test_l, Y_scores_l = zip(*futures)
    Y_test = pd.concat(Y_test_l).reset_index(drop=True)
    Y_scores = np.concatenate(Y_scores_l)

    return Y_test, Y_scores


Y_test, Y_scores = multithread_test_model(join(env.DATA_FOLDER, "ready"))

# ROC Curve
fpr, tpr, auc_thresholds = roc_curve(Y_test, Y_scores)
roc_auc = auc(fpr, tpr)

# gmean
gmean = np.sqrt(tpr * (1 - fpr))
index = np.argmax(gmean)
gmean_threshold = auc_thresholds[index]
Y_pred_gmean = Y_scores > gmean_threshold
# f1 score
f1_thresholds = np.arange(auc_thresholds[-1], auc_thresholds[0], 0.01)
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
verbose("Plotting.")
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
ax2r.plot(auc_thresholds, gmean, label="Gmean", color="blue")
ax2r.axvline(
    x=f1_threshold,
    color="lightgreen",
    label="Best f1 threshold (%.2f)" % f1_threshold,
)
ax2r.axvline(
    x=gmean_threshold,
    color="lightblue",
    label="Best gmean threshold (%.2f)" % gmean_threshold,
)
ax2r.legend()
ax2r.set_xlabel("Threshold")
ax2r.set_ylabel("F1 score and gmean")
ax2r.set_xlim([-2, 2])
ax2r.set_title("F1 score and gmean by threshold")

# Plot predictions with gmean threshold
Y_test = Y_test.astype(int)
ax3.scatter(
    Y_test.index,
    Y_pred_gmean,
    color="blue",
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
ax3.set_title("Predictions with gmean threshold")

plt.show()

# %%
