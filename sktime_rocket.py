from os import makedirs
from os.path import join, exists
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sktime.transformations.panel.rocket import (
    MiniRocketMultivariate,
)
from sktime.datatypes import check_raise
from sklearn.metrics import confusion_matrix
from sktime.utils import mlflow_sktime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import *


if not exists("models/model1/"):
    # LOAD TRAIN DATA
    X_train = (
        pd.read_pickle(join(env.DATA_FOLDER, "ready", "cases", "111.gz"))
        .groupby("window")
        .filter(lambda group: len(group) == 2000)
    )
    Y_train = pd.read_pickle(join(env.DATA_FOLDER, "ready", "labels", "111_labels.gz"))
    remaining_windows = X_train.index.get_level_values("window").unique()
    Y_train = Y_train[Y_train.index.isin(remaining_windows)]
    check_raise(X_train, mtype="pd-multiindex")

    minirocket_multi = MiniRocketMultivariate(n_jobs=4)
    minirocket_multi.fit(X_train)
    minirocket_multi.save("models/model1/estimator")
    X_train_transform = minirocket_multi.transform(X_train)

    scaler = StandardScaler(with_mean=False)
    X_train_scaled_transform = scaler.fit_transform(X_train_transform)

    # TRAIN MODEL
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_scaled_transform, Y_train)

    # classifier.score(X_test_scaled_transform, Y_test)
    makedirs("models", exist_ok=True)
    mlflow_sktime.save_model(classifier, "models/model1/")
else:
    classifier = mlflow_sktime.load_model("models/model1/")
    minirocket_multi = MiniRocketMultivariate()
    minirocket_multi.load_from_path("models/model1.zip")
    scaler = StandardScaler(with_mean=False)


# LOAD TEST DATA
X_test = (
    pd.read_pickle(join(env.DATA_FOLDER, "ready", "cases", "118.gz"))
    .groupby("window")
    .filter(lambda group: len(group) == 2000)
)
Y_test = pd.read_pickle(join(env.DATA_FOLDER, "ready", "labels", "118_labels.gz"))
remaining_windows = X_test.index.get_level_values("window").unique()
Y_test = Y_test[Y_test.index.isin(remaining_windows)]
X_test_transform = minirocket_multi.transform(X_test)
X_test_scaled_transform = scaler.fit_transform(X_test_transform)

# PREDICTION
Y_pred = classifier.predict(X_test_scaled_transform)
Y_scores = classifier.decision_function(X_test_scaled_transform)
cm = confusion_matrix(Y_test, Y_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]


# PLOT
# Create dataframe for the confusion matrices
df_cm = pd.DataFrame(cm, index=[False, True], columns=[False, True])
df_cm_norm = pd.DataFrame(cm_norm, index=[False, True], columns=[False, True])

# Create subplots
fig = plt.figure(layout="constrained")
gs = GridSpec(2, 2, figure=fig)
axtl = fig.add_subplot(gs[0, 0])
axtr = fig.add_subplot(gs[0, 1])
axb = fig.add_subplot(gs[1, :])

# Create heatmap for raw counts
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", ax=axtl)
axtl.set_title("Confusion Matrix - Counts")
axtl.set_xlabel("Predicted label")
axtl.set_ylabel("True label")

# Create heatmap for percentages
sns.heatmap(df_cm_norm, annot=True, fmt=".2%", cmap="Greens", ax=axtr)
axtr.set_title("Confusion Matrix - Percentages")
axtr.set_xlabel("Predicted label")
axtr.set_ylabel("True label")

axb.plot(Y_test, color="blue")
# axb.fill_between(Y_test.indices(), Y_test.values, color='blue', alpha=0.2)
axb.plot(Y_pred, color="red")
# axb.fill_between(Y_pred.indices(), Y_pred.values, color='red', alpha=0.2)
axb.set_title("Predictions")

plt.show()
