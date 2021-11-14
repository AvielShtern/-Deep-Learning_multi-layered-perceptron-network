import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchmetrics.functional import accuracy, precision_recall
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

""" Constants"""

BATCH_SIZE = 1000
LEARNING_RATE = 0.05
NUM_OF_EPOCHS = 500
DROPOUT_RATE = 0.5
REGULARIZATION_LAMBDA = 10 ** (-3.5)
PATH = ["/content/drive/MyDrive/dl/ex1/data/pos_A0201.txt",
        "/content/drive/MyDrive/dl/ex1/data/neg_A0201.txt"]

# Amino acid letters
LETTERS = ('A',
           'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')

"""
Data loading
"""
pos = np.genfromtxt(PATH[0], dtype=np.str)
neg = np.genfromtxt(PATH[1], dtype=np.str)
pos2neg_ratio = pos.shape[0] / neg.shape[0]
translation = {c: v for c, v in
               zip(LETTERS, np.split(np.identity(20).astype(np.float32), 20))}
translate = lambda strings: torch.tensor(
    [np.concatenate([translation[c] for c in s])
     for s in strings])
pos_x = translate(pos)
neg_x = translate(neg)
pos_y = torch.ones((pos_x.shape[0], 1), dtype=torch.float)
neg_y = torch.zeros((neg_x.shape[0], 1), dtype=torch.float)
all_data = torch.cat((pos_x, neg_x))
all_labels = torch.cat((pos_y, neg_y))

"""Split to train and test making sure there is the same ratio
 between the classes in the train and test."""

x_train_pos, x_test_pos, y_train_pos, y_test_pos = train_test_split(pos_x,
                                                                    pos_y,
                                                                    test_size=0.1)
x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(neg_x,
                                                                    neg_y,
                                                                    test_size=0.1)
x_test = torch.cat((x_test_pos, x_test_neg))
y_test = torch.cat((y_test_pos, y_test_neg))
train_size = x_train_pos.shape[0] + x_train_neg.shape[0]

# create two dataloaders, for the negative and positive to make
# the batches balanced
pos_batch_size = int(np.round(pos2neg_ratio * BATCH_SIZE))
pos_loader = DataLoader(
    TensorDataset(x_train_pos, y_train_pos), shuffle=True,
    batch_size=pos_batch_size
)
neg_loader = DataLoader(
    TensorDataset(x_train_neg, y_train_neg), shuffle=True,
    batch_size=BATCH_SIZE - pos_batch_size
)

"""Network"""


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(20 * 9, 250),
                nn.ReLU(),
                nn.Dropout(p=DROPOUT_RATE),
                nn.Linear(250, 250),
                nn.Dropout(p=DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(250, 1),
                nn.Sigmoid()
            ])

    def forward(self, x):
        x = torch.flatten(x, 1)
        for layer in self.layers:
            x = layer(x)
        return x


"""
Training
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
x_test, y_test = x_test.to(device), y_test.to(device)
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                           lr_lambda=lambda
                                               epoch: 1 if epoch < NUM_OF_EPOCHS * 0.6 else 0.5)
# Find positive and negative samples in the test set for the weights in loss function
test_is_pos = (y_test == 1)
num_of_pos_test_samples = test_is_pos.sum()
weight_pos_test = (y_test.shape[
                       0] - num_of_pos_test_samples) / num_of_pos_test_samples
weights_test = torch.where(test_is_pos, weight_pos_test,
                           torch.tensor(1.0, dtype=torch.float).to(device))

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
train_recalls = []
test_recalls = []
# training loop.
for epoch in range(NUM_OF_EPOCHS):
    net.train()  # make sure we are in "train" mode for dropout.
    running_loss = 0.0
    running_accuracy = 0.0
    running_recall = 0.0
    for batch_pos, batch_neg in zip(pos_loader, neg_loader):
        batch_pos = [item.to(device) for item in batch_pos]
        batch_neg = [item.to(device) for item in batch_neg]
        pos_samples, pos_labels = batch_pos
        neg_samples, neg_labels = batch_neg
        data, labels = torch.cat((pos_samples, neg_samples)), torch.cat(
            (pos_labels, neg_labels))
        shuffled_indices = torch.randperm(data.shape[0])
        data, labels = data[shuffled_indices], labels[shuffled_indices]

        # find weights for loss based on batch.
        pos_labels_ind = (labels == 1)
        num_pos = pos_labels_ind.sum()
        weight_pos = (labels.shape[0] - num_pos) / num_pos
        weights = torch.where(pos_labels_ind, weight_pos,
                              torch.tensor(1.0).to(device))

        optimizer.zero_grad()
        output = net(data)
        loss_fn = nn.BCELoss(weight=weights)
        train_loss = loss_fn(output, labels)
        l1_regularization = torch.linalg.vector_norm(
            nn.utils.parameters_to_vector(net.parameters()), ord=1)
        regularized_train_loss = train_loss + REGULARIZATION_LAMBDA * l1_regularization
        regularized_train_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        running_loss += train_loss.item()
        running_accuracy += accuracy(output, labels.int()).item()
        recall = precision_recall(output.double(), labels.int())[1].item()
        running_recall += recall
    train_losses.append(running_loss / np.ceil(train_size / BATCH_SIZE))
    train_accuracies.append(
        running_accuracy / np.ceil(train_size / BATCH_SIZE))
    train_recalls.append(running_recall / np.ceil(train_size / BATCH_SIZE))
    net.eval()
    with torch.no_grad():
        test_output = net(x_test)
        test_losses.append(
            F.binary_cross_entropy(test_output, y_test, weight=weights_test))
        test_accuracies.append(accuracy(test_output, y_test.int()).item())
        precision, recall = precision_recall(test_output, y_test.int())
        test_recalls.append(recall.item())

fig, ax = plt.subplots(1, 3)
fig.set_size_inches((25, 10))
ax[0].plot(train_losses, label="train error")
ax[0].plot(test_losses, label="test error")
ax[0].legend()
ax[0].title.set_text("CELoss")
ax[0].set_ylim([0, 1.5])

ax[1].plot(train_accuracies, label="train")
ax[1].plot(test_accuracies, label="test")
ax[1].legend()
ax[1].title.set_text("Accuracy")
ax[1].set_ylim([0, 1])

ax[2].plot(train_recalls, label="train")
ax[2].plot(test_recalls, label="test")
ax[2].legend()
ax[2].title.set_text("Recall")
ax[2].set_ylim([0, 1])

fig.suptitle(
    f"Learning rate = {LEARNING_RATE}, Num of epochs = {NUM_OF_EPOCHS}")

spike_seq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLH" \
            "STQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKS" \
            "NIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHK" \
            "NNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKN" \
            "IDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALH" \
            "RSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALD" \
            "PLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFN" \
            "ATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCF" \
            "TNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNL" \
            "DSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYF" \
            "PLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCV" \
            "NFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPC" \
            "SFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNV" \
            "FQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYT" \
            "MSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLL" \
            "QYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSK" \
            "RSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALL" \
            "AGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDS" \
            "LSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQID" \
            "RLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHL" \
            "MSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQR" \
            "NFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDV" \
            "DLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGL" \
            "IAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT "


def find_top_5(net, seq):
    """
    Find the 5 most detectable subsequences in a given sequence according to the given net
    :param net: a neural network which is a subclass of torch.nn.Module
    :param seq: Protein sequence.
    :return:
    """
    sub_seq = np.array(
        [seq[i: i + 9] for i in range(len(seq) - 8)])
    data = translate(sub_seq)
    output = net(data)
    top_5_indices = torch.topk(output, 5, dim=0)[1]
    top_5_seq = sub_seq[top_5_indices]
    return top_5_seq


def optimize_sequence(net, iters):
    """
    Perform optimization and find the most detectable sequence according to net
    :param net:
    :param iters: number of GD iterations
    :return: A sequence and a list of probabilities from GD iterations
    """
    net.eval()
    rand_indices = torch.randint(0, 20, (9,))
    w = torch.eye(20)[rand_indices]
    w.requires_grad = True
    optimizer = optim.SGD([w], lr=0.1)
    probabilities = []
    for i in range(iters):
        output = -net(torch.unsqueeze(w, 0))
        probabilities.append(-output)
        output.backward()
        optimizer.step()
    seq = ''.join(np.array(LETTERS)[w.max(dim=1).indices]) #translate back to string
    return seq, probabilities

