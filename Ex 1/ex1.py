import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Define the list of amino acids
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# One-hot encoding function for a peptide
def one_hot_vector(peptide):
    one_hot_vector = np.zeros(20)
    for aa in peptide:
        one_hot_vector[amino_acids.index(aa)] += 1
    return one_hot_vector

# Preprocessing function
def preprocess():
    neg_data = np.loadtxt('neg_A0201.txt', dtype=str)
    pos_data = np.loadtxt('pos_A0201.txt', dtype=str)

    neg_data_onehot = np.array([one_hot_vector(peptide) for peptide in neg_data])
    pos_data_onehot = np.array([one_hot_vector(peptide) for peptide in pos_data])
    pos_data_onehot = resample(pos_data_onehot, n_samples=len(neg_data_onehot), random_state=123)

    neg_data_encoded = torch.tensor(neg_data_onehot)
    pos_data_encoded = torch.tensor(pos_data_onehot)
    neg_torch = torch.cat((neg_data_encoded, torch.zeros(neg_data_encoded.size()[0], 1)), 1)
    pos_torch = torch.cat((pos_data_encoded, torch.ones(pos_data_encoded.size()[0], 1)), 1)

    data = torch.cat((neg_torch, pos_torch), 0)
    data = data[torch.randperm(data.size()[0])]
    train_data = data[:int(data.size()[0] * 0.9), :]
    test_data = data[int(data.size()[0] * 0.1):, :]

    return train_data, test_data

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        output = self.sigmoid(x)
        return output

class Linear_MLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128):
        super(Linear_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = self.sigmoid(x)
        return output

# Training function with evaluation
def train_and_evaluate_model(model, train_loader, test_data, criterion, optimizer, device, batch_size, epochs=50):
    train_loss = []
    eval_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    model.train()
    test_loss = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_test_loss = 0.0
        for i in range(0, len(train_loader), batch_size):
            inputs = train_loader[i:i + batch_size, :-1].float().to(device)
            labels = train_loader[i:i + batch_size, -1].unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / (len(train_loader) // batch_size)
        train_loss.append(avg_epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}')
        _, tloss= eval_model(model, test_data, device,criterion)

        test_loss.append(tloss)
    return model, train_loss, eval_metrics,test_loss

# Evaluation function
def eval_model(model, test_data, device,criterion):
    model.eval()
    predicted_labels = 0
    y_true = []
    y_pred = []
    test_loss = []
    with torch.no_grad():
        test_loss = 0.0
        for i in range(len(test_data)):
            
            inputs = test_data[i, :-1].float().to(device)
            labels = test_data[i, -1].unsqueeze(0).to(device)
            outputs = model(inputs)
            predicted = 1 if outputs >= 0.5 else 0
            y_true.append(labels)
            y_pred.append(predicted)
            loss = criterion(outputs, labels.float())
            if(predicted == labels):
                predicted_labels += 1
            
            test_loss += loss.item()
        test_loss = test_loss / len(test_data)
        print(f'Test Loss: {test_loss:.4f}')
        accuracy = predicted_labels / len(test_data)

    print(f'Accuracy: {accuracy*100:.4f}')

    return accuracy,test_loss

# Function to extract 9-mers from a given protein sequence
def extract_9mers(protein_sequence):
    return [protein_sequence[i:i + 9] for i in range(len(protein_sequence) - 8)]

# Function to predict peptides from the Spike protein of SARS-CoV-2
def predict_peptides(model, device, spike_protein_seq):
    model.eval()
    peptides = extract_9mers(spike_protein_seq)
    results = []
    with torch.no_grad():
        for peptide in peptides:
            one_hot_peptide = one_hot_vector(peptide)
            input_tensor = torch.tensor(one_hot_peptide).float().unsqueeze(0).to(device)
            output = model(input_tensor)
            results.append((peptide, output.item()))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:3]

# Main function
def main():
    #Q1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, test_data = preprocess()
    batch_size = 32
    #
    model1=MLP(20,20).to(device)
    
    model2 = MLP().to(device)
    
    criterion1 = nn.BCELoss()
    criterion2 = nn.BCELoss()
    criterion3 = nn.BCELoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    model1, train_loss1, eval_metrics1,test_loss1 = train_and_evaluate_model(model1, train_data, test_data, criterion1, optimizer1, device, batch_size, epochs=40)
    model2, train_loss2, eval_metrics2,test_loss2 = train_and_evaluate_model(model2, train_data, test_data, criterion2, optimizer2, device, batch_size, epochs=40)

    model3 = Linear_MLP().to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=0.001)
    model3, train_loss3, eval_metrics3,test_loss3 = train_and_evaluate_model(model3, train_data, test_data, criterion3, optimizer3, device, batch_size, epochs=40)


    #Q2
    plt.plot(train_loss1, label='Train Loss')
    plt.plot(test_loss1, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MLP with hidden size 20')
    plt.legend()
    plt.show()
    
    #Q3
    plt.plot(train_loss2, label='Train Loss')
    plt.plot(test_loss2, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('MLP with hidden size 128')
    plt.show()

    #Q4
    plt.plot(train_loss3, label='Train Loss')
    plt.plot(test_loss3, label='Test Loss')
    plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    plt.legend()
    plt.title('Linear MLP with hidden size 128')
    plt.show()


    
    file = open('spike_data.txt', 'r')
    spike_data = file.read()
    spike_protein_seq = "".join(spike_data.split())

    top_peptides = predict_peptides(model2, device, spike_protein_seq)
    print(f'Three most detectable peptides: {top_peptides}')


if __name__ == "__main__":
    main()
