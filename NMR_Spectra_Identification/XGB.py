import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, average_precision_score
import matplotlib.pyplot as plt

# load in the data
path = 'processed_nmr_with_labels.csv'
df = pd.read_csv(path)
df = df.sample(frac=1).reset_index(drop=True)
df.replace(['Unknown'], np.nan, inplace=True)
print(len(df))
# create tensors
train_df, test_df = train_test_split(df, test_size=0.2)
X_train = train_df.iloc[:, 2:27].fillna(0).astype(float).values
X_test = test_df.iloc[:, 2:27].fillna(0).astype(float).values
y_train = train_df.iloc[:, 27:40].fillna(0).astype(float).values
y_test = test_df.iloc[:, 27:40].fillna(0).astype(float).values
labels = df.columns[27:40]

# initiate XGB from packages
XGB_model = xgb.XGBClassifier(objective = 'binary:logistic')
XGB_model = MultiOutputClassifier(XGB_model)
XGB_model.fit(X_train, y_train)
y_pred_proba = XGB_model.predict_proba(X_test)
y_pred = np.array([prob[:, 1] for prob in y_pred_proba]).T 
y_pred = (y_pred > .5).astype(int)

# evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
hammingLoss = hamming_loss(y_test, y_pred)
jaccard = jaccard_score(y_test, y_pred, average = 'samples', zero_division = 0)
yPredPosProba = np.array([prob[:, 1] for prob in y_pred_proba]).T
averagePrecision = average_precision_score(y_test, yPredPosProba, average = 'macro')
print('Accuracy on testing: {:.1f}%'.format(accuracy * 100))
print('Hamming Loss on testing: {:.3f}'.format(hammingLoss))
print('Jaccard Index on testing: {:.3f}'.format(jaccard))
print('Average Precision on testing: {:.3f}'.format(averagePrecision))

# generate confusion matrices for each label and then print them out
confusionMats = []
for label in range(len(labels)): 
    confusionMat = metrics.confusion_matrix(y_test[:, label], y_pred[:, label])
    confusionMats.append(confusionMat)

fig, axes = plt.subplots(4, 3, figsize = (15, 10))
axes = axes.ravel()

# print out each matrix
for i, mat in enumerate(confusionMats[:12]):
    output = metrics.ConfusionMatrixDisplay(confusion_matrix = mat, display_labels = [0,1])
    output.plot(ax=axes[i], values_format = 'd', cmap = 'Blues')
    axes[i].set_title('Confusion Matrix for {}'.format(labels[i]), fontsize=10)
plt.tight_layout()
plt.show()

