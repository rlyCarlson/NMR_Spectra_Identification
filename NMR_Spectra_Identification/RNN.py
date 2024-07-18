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

# process all the data as a DF and then convert it to tensor, replacing all problematic cells with 0
path = 'processed_nmr_with_labels.csv'
df = pd.read_csv(path)
df = df.sample(frac=1).reset_index(drop=True)
df.replace(['Unknown'], np.nan, inplace=True)
train_df, test_df = train_test_split(df, test_size=0.2)
# consider the features that are not part of NMR spectra (temperature etc.)
X_train_non_seq = train_df.iloc[:, 2:7].fillna(0).astype(float).values
X_test_non_seq = test_df.iloc[:, 2:7].fillna(0).astype(float).values
print("done select 1")
y_train = train_df.iloc[:, 27:40].fillna(0).astype(float).values
print("done select 2")
titles = df.columns[27:40]
# consider the features that correspond to the NMR spectra (considered as sequential)
X_train_seq = train_df.iloc[:, 7:27].fillna(0).astype(float).values
X_test_seq = test_df.iloc[:, 7:27].fillna(0).astype(float).values
print("done select 3")
y_test = test_df.iloc[:, 27:40].fillna(0).astype(float).values
print("done select 4")
# throw into tensors, to be used in training
X_train_non_seq_tensor = torch.tensor(X_train_non_seq, dtype=torch.float32)
X_train_seq_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_non_seq_tensor = torch.tensor(X_test_non_seq, dtype=torch.float32)
X_test_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# create a dataset that accomodates both sequential and non sequential data
class CustomDataset(Dataset):
    def __init__(self, non_seq_data, seq_data, labels):
        self.non_seq_data = non_seq_data
        self.seq_data = seq_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        non_seq_sample = self.non_seq_data[i]
        seq_sample = self.seq_data[i]
        label = self.labels[i]
        return non_seq_sample, seq_sample, label

# initialize all the datasets
train_dataset = CustomDataset(X_train_non_seq_tensor, X_train_seq_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_non_seq_tensor, X_test_seq_tensor, y_test_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size = len(train_dataset), shuffle = True)
test_loader = DataLoader(dataset=test_dataset, batch_size = len(test_dataset), shuffle = False)

# this neural net is a combination of RNN-LSTM for the NMR spectra (treated as a sequence)
# and of a fully connected neural network for the nonsequential data (the atributes such as temperature etc.)
class CombinedModel(nn.Module):
    def __init__(self, FNNDim, RNNDim, hidden, output, layers):
        super(CombinedModel, self).__init__()
        self.numLayers = layers
        self.hiddenDim = hidden
        # your average fully connected net
        self.FNN = nn.Linear(FNNDim, 50) 
        # an RNN-LSTM
        self.lstm = nn.LSTM(RNNDim, hidden, layers, batch_first = True)        
        self.fnnFinal = nn.Linear(hidden + 50, output)
    
    def forward(self, nonsequential, sequential):
        fnnOut = torch.relu(self.FNN(nonsequential))
        h0 = torch.zeros(self.numLayers, sequential.size(0), self.hiddenDim).to(sequential.device)
        c0 = torch.zeros(self.numLayers, sequential.size(0), self.hiddenDim).to(sequential.device)
        
        RNNOut, states = self.lstm(sequential, (h0, c0))
        RNNOut = RNNOut[:, -1, :]
        
        totalOutput = torch.cat((fnnOut, RNNOut), dim=1)
        
        out = self.fnnFinal(totalOutput)
        return torch.sigmoid(out)

fnnInputDim = 5
RNNInputDim = 1
hidDim = 128
outDim = 12
numLayers = 2

# call to actually use model
model = CombinedModel(fnnInputDim, RNNInputDim, hidDim, outDim, numLayers)

# binary cross entropy again
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# training loop
model.train()

num_epochs = 16
for epoch in range(num_epochs):
    def step():
        optimizer.zero_grad()
        non_seq_data, seq_data, labels = next(iter(train_loader))
        outputs = model(non_seq_data, seq_data.unsqueeze(-1))
        loss = criterion(outputs, labels)
        loss.backward()
        return loss
    optimizer.step(step)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {step().item():.4f}')

# evaluation metrics contained here
def evaluate_model(model, data_loader):
    model.eval()
    totalPreds = []
    totalLabels = []
    
    with torch.no_grad():
        for non_seq_data, seq_data, labels in data_loader:
            predictions = model(non_seq_data, seq_data.unsqueeze(-1))
            # discretize to 0-1
            binaryPreds = (predictions > 0.5).float()
            totalPreds.append(binaryPreds)
            totalLabels.append(labels)
    
    totalPreds = torch.cat(totalPreds, dim=0).cpu().numpy()
    totalLabels = torch.cat(totalLabels, dim=0).cpu().numpy()
    # go through and calculate every metric and print it out
    print(f'Hamming Loss: {hamming_loss(totalLabels, totalPreds)}')
    print(f'Exact Match Ratio: {accuracy_score(totalLabels, totalPreds)}')
    print(f'Jaccard Index: {jaccard_score(totalLabels, totalPreds, average = "samples")}')
    print(f'Average Precision: {average_precision_score(totalLabels, totalPreds, average = "macro")}')
    
    report = classification_report(totalLabels, totalPreds, target_names=[f'Label {i}' for i in range(totalLabels.shape[1])], zero_division=0)
    print(report)
    # find confusion matrices and print out each time
    confusionMats = []
    for label in range(totalLabels.shape[1]):
        cMat = metrics.confusion_matrix(totalLabels[:, label], totalPreds[:, label])
        confusionMats.append(cMat)

    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    axes = axes.ravel()
    # combine all matrices and output them on one page
    for i, mat in enumerate(confusionMats[:12]):
        cm_display = ConfusionMatrixDisplay(confusion_matrix = mat, display_labels=[0, 1])
        cm_display.plot(ax = axes[i], values_format = 'd', cmap = 'Blues')
        axes[i].set_title('Confusion Matrix for {}'.format(titles[i]), fontsize=10)
    
    plt.tight_layout()
    plt.show()

evaluate_model(model, test_loader)

