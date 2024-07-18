import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from sklearn import metrics
from sklearn.metrics import hamming_loss, accuracy_score, classification_report, jaccard_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# load in the dataset and convert everything to tensors
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
# throw them into tensors for training
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# initialize the tensor datasets to be used in the neural network
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(dataset = train_dataset, batch_size = len(train_dataset), shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = False)

# note this class is a slight modification since this is for multilabel
class MultiLabelNN(nn.Module):
    def __init__(self, i, h, o):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(i, h)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, o)
    # input is passed through a relu after each layer
    # then after the last layer is passed through a sigmoid before output to give value between 0 and 1
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) 
        return x

inputDim = X_train.shape[1]
hiddenDim = 128
outputDim = y_train.shape[1]
model = MultiLabelNN(inputDim, hiddenDim, outputDim)

# binary cross entropy loss function again (explained in paper)
criterion = nn.BCELoss()
# use the LBFGS optimization (quasi-newtonian approximation) which we discuss in paper
optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    def closure():
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        return loss
    optimizer.step(closure)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {closure().item():.4f}')

# generate evaluation metrics based on testing data
def evaluate_model(model, data_loader):
    model.eval() 
    with torch.no_grad():
        finalPredictions = []
        finalLabels = []
        for inputs, labels in data_loader:
            predictions = model(inputs)
            finalPredictions.append(predictions)
            finalLabels.append(labels)
    
    predictions = torch.cat(finalPredictions, dim = 0).cpu().numpy()
    labels = torch.cat(finalLabels, dim = 0).cpu().numpy()
    predictions_binary = (predictions > 0.5).astype(int)
    
    # print all evaluation metrics
    print(f'Hamming Loss: {hamming_loss(labels, predictions_binary)}')
    print(f'Accuracy: {accuracy_score(labels, predictions_binary)}')
    print(f'Jaccard Score: {jaccard_score(labels, predictions_binary, average = "samples")}')
    print(f'Average Precision: {average_precision_score(labels, predictions_binary, average="macro")}')
    print(classification_report(labels, predictions_binary, zero_division=0))

    # generate confusion matrices for each label and then print them out on one plot
    confusionMats = [metrics.confusion_matrix(labels[:, i], predictions_binary[:, i]) for i in range(predictions_binary.shape[1])]

    fig, axes = plt.subplots(4, 3, figsize = (15, 10))
    axes = axes.ravel()

    for i, mat in enumerate(confusionMats[:12]):
        output = metrics.ConfusionMatrixDisplay(confusion_matrix = mat, display_labels = [0,1])
        output.plot(ax=axes[i], values_format = 'd', cmap = 'Blues')
        axes[i].set_title(f'Confusion Matrix for Label {titles[i]}', fontsize = 10)
    plt.tight_layout()
    plt.show()

evaluate_model(model, test_loader)
