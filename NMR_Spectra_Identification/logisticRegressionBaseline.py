import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import hamming_loss, accuracy_score, classification_report, jaccard_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

# baseline model
# read in the data and get it into tensor format, removing any unknown information
path = 'processed_nmr_with_labels.csv'
df = pd.read_csv(path)
df = df.sample(frac=1).reset_index(drop=True)
df.replace(['Unknown'], np.nan, inplace=True)
train_df, test_df = train_test_split(df, test_size=0.2)
X_train = train_df.iloc[:, 2:27].fillna(0).astype(float).values
X_test = test_df.iloc[:, 2:27].fillna(0).astype(float).values
y_train = train_df.iloc[:, 27:40].fillna(0).astype(float).values
y_test = test_df.iloc[:, 27:40].fillna(0).astype(float).values
titles = df.columns[27:40]
# throw into tensors for training
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

X_test = test_df.iloc[:, 2:27].fillna(0).astype(float).values
y_test = test_df.iloc[:, 27:40].fillna(0).astype(float).values
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)

# create a class for logistic regression to throw the tensors into
class MultiLabelLogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    # standard sigmoid function
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
# create an instance of the model
model = MultiLabelLogisticRegression(input_dim, output_dim)

# binary cross entropy loss was chosen here
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# learning steps
num_epochs = 170
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# print out all the relevant evaluation metrics
def evaluate_model(model, X_test, y_test):
    model.eval()  
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)

    predictions_binary = (predictions > 0.5).float()
    predictions_binary_numpy = predictions_binary.numpy()
    y_test_numpy = y_test.numpy()
    ham_loss = hamming_loss(y_test_numpy, predictions_binary_numpy)
    exact_match = accuracy_score(y_test_numpy, predictions_binary_numpy)
    jaccard = jaccard_score(y_test_numpy, predictions_binary_numpy, average='samples')
    average_precision = average_precision_score(y_test_numpy, predictions_binary_numpy, average='macro')
    report = classification_report(y_test_numpy, predictions_binary_numpy, target_names=[f'Label {i}' for i in range(y_test_numpy.shape[1])], zero_division=0)

    # check for overfitting
    print(f'Binary Cross Entropy Loss: {test_loss}')
    # all the evaluation
    print(f'Hamming Loss: {ham_loss}')    
    print(f'Exact Match Ratio (Subset Accuracy): {exact_match}')    
    print(f'Jaccard Index: {jaccard}')    
    print(f'Average Precision: {average_precision}')    
    print('Precision, Recall, F1-Score per label and averages:\n', report)

    # print out all the confusion matrices
    confusionMatrices = []
    for label in range(12):
        confusionMat = metrics.confusion_matrix(y_test_numpy[:, label], predictions_binary_numpy[:, label])
        confusionMatrices.append(confusionMat)
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, cm in enumerate(confusionMatrices[:12]):
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
        cm_display.plot(ax = axes[i], values_format = 'd', cmap = 'Blues')
        axes[i].set_title('Confusion Matrix for {}'.format(titles[i]), fontsize = 10)
    plt.tight_layout()
    plt.show()

evaluate_model(model, X_test_tensor, y_test_tensor)
