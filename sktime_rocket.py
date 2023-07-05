from os.path import join
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sktime.transformations.panel.rocket import (
    MiniRocketMultivariate,
)
from sktime.datatypes import check_raise, convert
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

X_train = pd.read_pickle(join(env.DATA_FOLDER, "ready", "cases", "111.gz")).groupby('window').filter(lambda group: len(group) == 2000)
Y_train = pd.read_pickle(join(env.DATA_FOLDER, "ready", "labels", "111_labels.gz"))
remaining_windows = X_train.index.get_level_values('window').unique()
Y_train = Y_train[Y_train.index.isin(remaining_windows)]
check_raise(X_train, mtype="pd-multiindex")

minirocket_multi = MiniRocketMultivariate()
minirocket_multi.fit(X_train)
X_train_transform = minirocket_multi.transform(X_train)

scaler = StandardScaler(with_mean=False)
X_train_scaled_transform = scaler.fit_transform(X_train_transform)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_scaled_transform, Y_train)

X_test = pd.read_pickle(join(env.DATA_FOLDER, "ready", "cases", "118.gz")).groupby('window').filter(lambda group: len(group) == 2000)
Y_test = pd.read_pickle(join(env.DATA_FOLDER, "ready", "labels", "118_labels.gz"))
remaining_windows = X_test.index.get_level_values('window').unique()
Y_test = Y_test[Y_test.index.isin(remaining_windows)]

X_test_transform = minirocket_multi.transform(X_test)

X_test_scaled_transform = scaler.transform(X_test_transform)
#classifier.score(X_test_scaled_transform, Y_test)

# PREDICTION
Y_pred = classifier.predict(X_test_scaled_transform)
cm = confusion_matrix(Y_test, Y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create dataframe for the confusion matrices
df_cm = pd.DataFrame(cm, index=[False, True], columns=[False, True])
df_cm_norm = pd.DataFrame(cm_norm, index=[False, True], columns=[False, True])

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Create heatmap for raw counts
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix - Counts')
ax[0].set_xlabel('Predicted label')
ax[0].set_ylabel('True label')

# Create heatmap for percentages
sns.heatmap(df_cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix - Percentages')
ax[1].set_xlabel('Predicted label')
ax[1].set_ylabel('True label')

plt.tight_layout()
plt.show()