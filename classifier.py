import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_loader import DataLoader


# ==============
# Build Model
# ==============

class CNN(nn.Module):
    """1D Convolutional Neural Network"""

    def __init__(self, vocab_size, embed_dim, class_num, kernel_num, kernel_sizes, drop_out):
        super(CNN, self).__init__()
        V = vocab_size
        D = embed_dim
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes

        self.embedding = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


class BiRNN(nn.Module):
    """Bi-directional recurrent neural network."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, bidirectional):
        super(BiRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # input is concatenated forward and backward hidden state

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_output, hidden = self.rnn(embeds)
        # concatenate forward and backward hidden before FC
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        fc_output = self.fc(hidden.squeeze(0))

        return fc_output


class BiLSTM(nn.Module):
    """Bi-directional long-short term memory."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, bidirectional):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # input is concatenated forward and backward hidden state

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_output, (hidden, memory) = self.lstm(embeds)
        # concatenate forward and backward hidden before FC
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        fc_output = self.fc(hidden.squeeze(0))

        return fc_output


class SentimentClassifier:
    """Sentiment classifier that takes input data and model."""

    def __init__(self, train_iter, valid_iter, model, device=None):
        """
        :param train_iter: training data iterator
        :param valid_iter: validation data iterator
        :param model: model to be trained
        :param device: GPU device if available
        """
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
        self.loss_function = self.loss_function.to(device)  # place it to GPU (if available)

    def accuracy(self, pred, y):
        """Define metric for evaluation."""
        pred = torch.round(torch.sigmoid(pred))
        acc = torch.sum((pred == y)).float() / len(y)
        return acc

    def train_model(self):
        """Train one epoch of inputs and update weights.

        :return: average loss, average accuracy.
        """
        epoch_loss = []
        epoch_acc = []
        self.model.train()

        for batch_data in self.train_iter:
            self.optimizer.zero_grad()  # clear out gradient
            if type(self.model) is CNN:
                pred = self.model(batch_data.text.t_()).squeeze(1)
            else:
                pred = self.model(batch_data.text).squeeze(1)
            y = (batch_data.label.squeeze(0) >= 3).float()  # neg:2, pos:3 -> convert them to 0 and 1
            loss = self.loss_function(pred, y)
            acc = self.accuracy(pred, y)

            # backprob and update gradient
            loss.backward()
            self.optimizer.step()

            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())

        return np.mean(epoch_loss), np.mean(epoch_acc)

    def evaluate_model(self):
        """Evaluate one epoch of inputs.

        :return: average loss, average accuracy.
        """
        epoch_loss = []
        epoch_acc = []
        self.model.eval()

        with torch.no_grad():
            for batch_data in self.valid_iter:
                if type(self.model) is CNN:
                    pred = self.model(batch_data.text.t_()).squeeze(1)
                else:
                    pred = self.model(batch_data.text).squeeze(1)
                y = (batch_data.label.squeeze(0) >= 3).float()
                loss = self.loss_function(pred, y)
                acc = self.accuracy(pred, y)

                epoch_loss.append(loss.item())
                epoch_acc.append(acc.item())

        return np.mean(epoch_loss), np.mean(epoch_acc)

    def run_epochs(self, num_epochs=10, eval_each=1):
        """Run # epochs and evaluate the model.

        :return: average loss and accuracy per epoch for training and validation set.
        """
        train_epoch_metrics, valid_epoch_metrics = [], []

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_model()
            valid_loss, valid_acc = self.evaluate_model()
            train_epoch_metrics.append((train_loss, train_acc))
            valid_epoch_metrics.append((valid_loss, valid_acc))

            if (epoch + 1) % eval_each == 0:
                print('Epoch %d | Train Loss: %.2f | Train Acc: %.2f | Test Loss: %.2f | Test Acc: %.2f'
                      % (epoch, train_loss, train_acc, valid_loss, valid_acc))

        return train_epoch_metrics, valid_epoch_metrics

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


# ==============
# Train Model
# ==============

def train_classifier(alg='BiLSTM', data_loader=None, small_subsets=False, model_config=None, outfile=None, device=None):
    """Train classifier and return the trained classifier and its metrics.

    :param alg: algorithm to train {'BiRNN', 'BiLSTM', 'CNN'}
    :param data_loader: DataLoader object
    :param small_subsets: boolean, whether to use smaller subsets of data
    :param model_config: dictionary of model configuration
    :param outfile: outfile to save the trained model
    :param device: GPU device if available
    :return: trained classifier, train and validation metrics per epoch
    """

    # load data for training and evaluation
    if not data_loader:
        data_loader = DataLoader()
    if small_subsets:
        train_iter, valid_iter = data_loader.small_train_valid_iter()
    else:
        train_iter, valid_iter = data_loader.large_train_valid_iter()

    # model configuration
    if not model_config:
        model_config = dict()
        model_config['VOCAB_SIZE'], model_config['EMBEDDING_DIM'] = data_loader.TEXT.vocab.vectors.shape
        model_config['HIDDEN_DIM'] = 32
        model_config['OUTPUT_DIM'] = 1
        model_config['BIDRECTIONAL'] = True
        model_config['KERNEL_NUM'] = 100
        model_config['KERNEL_SIZES'] = [3, 4, 5]
        model_config['DROP_OUT'] = 0.5
        model_config['NUM_EPOCH'] = 10
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize model
    if alg == 'BiRNN':
        model = BiRNN(model_config['VOCAB_SIZE'], model_config['EMBEDDING_DIM'],
                      model_config['HIDDEN_DIM'], model_config['OUTPUT_DIM'], model_config['BIDRECTIONAL']).to(device)
    elif alg == 'BiLSTM':
        model = BiLSTM(model_config['VOCAB_SIZE'], model_config['EMBEDDING_DIM'],
                       model_config['HIDDEN_DIM'], model_config['OUTPUT_DIM'], model_config['BIDRECTIONAL']).to(device)
    elif alg == 'CNN':
        model = CNN(model_config['VOCAB_SIZE'], model_config['EMBEDDING_DIM'], model_config['OUTPUT_DIM'],
                    model_config['KERNEL_NUM'], model_config['KERNEL_SIZES'], model_config['DROP_OUT']).to(device)
    else:
        raise ValueError('Unknown model: %s.' % alg)

    # replace initial weights of embedding layer with pre-trained embedding
    pretrained_embeddings = data_loader.TEXT.vocab.vectors
    model.embedding.weight.from_pretrained(pretrained_embeddings)
    #model.embedding.weight.data.copy_(pretrained_embeddings)

    # train classifier
    classifier = SentimentClassifier(train_iter, valid_iter, model)
    print('start training: %s' %alg)
    print('model config:', model_config.items())
    train_epoch_metrics, valid_epoch_metrics = classifier.run_epochs(model_config['NUM_EPOCH'])

    # save trained model
    if outfile:
        classifier.save_model('%s.pt' % outfile)
        print("Saved model's state_dict:")
        for param_tensor in classifier.model.state_dict():
            print(param_tensor, "\t", classifier.model.state_dict()[param_tensor].size())

    return classifier, train_epoch_metrics, valid_epoch_metrics


# ===================
# Load Saved Model
# ===================

def load_saved_model(alg, path, data_loader=None, small_subsets=False, model_config=None):
    """Load saved model from file.

    :param alg: algorithm to initialize, should match the saved model {'BiRNN', 'BiLSTM', 'CNN'}
    :param path: saved model file to load
    :param small_subsets: boolean, whether to use a smaller subsets of data
    :param data_loader: DataLoader object
    :param model_config: dictionary of model configuration
    :return: loaded model
    """

    # load data and build vocabulary
    if not data_loader:
        data_loader = DataLoader()
    if small_subsets:
        data_loader.small_train_valid()
    else:
        data_loader.large_train_valid()

    if not model_config:
        model_config = dict()
        model_config['VOCAB_SIZE'], model_config['EMBEDDING_DIM'] = data_loader.TEXT.vocab.vectors.shape
        model_config['HIDDEN_DIM'] = 32
        model_config['OUTPUT_DIM'] = 1
        model_config['BIDRECTIONAL'] = True
        model_config['KERNEL_NUM'] = 100
        model_config['KERNEL_SIZES'] = [3, 4, 5]
        model_config['DROP_OUT'] = 0.5
        model_config['NUM_EPOCH'] = 10

    # initialize model
    if alg == 'BiRNN':
        model = BiRNN(model_config['VOCAB_SIZE'], model_config['EMBEDDING_DIM'],
                      model_config['HIDDEN_DIM'], model_config['OUTPUT_DIM'], model_config['BIDRECTIONAL'])
    elif alg == 'BiLSTM':
        model = BiLSTM(model_config['VOCAB_SIZE'], model_config['EMBEDDING_DIM'],
                       model_config['HIDDEN_DIM'], model_config['OUTPUT_DIM'], model_config['BIDRECTIONAL'])
    elif alg == 'CNN':
        model = CNN(model_config['VOCAB_SIZE'], model_config['EMBEDDING_DIM'], model_config['OUTPUT_DIM'],
                    model_config['KERNEL_NUM'], model_config['KERNEL_SIZES'], model_config['DROP_OUT'])
    else:
        raise ValueError('Unknown model: %s.' % alg)

    # load model
    model.load_state_dict(torch.load(path))
    print(model.eval())

    return model
