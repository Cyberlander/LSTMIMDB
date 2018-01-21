from keras.datasets import imdb
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse

class LSTMIMDB(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super(LSTMIMDB, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding(FLAGS.num_words, embedding_dim)
        self.lstm_layer = nn.LSTM( embedding_dim, hidden_dim )
        self.linear_layer = nn.Linear( hidden_dim, 2)
    def forward(self,inputs, hidden):
        x = self.embedding_layer(inputs).view(len(inputs),1,-1)
        lstm_out, lstm_hidden = self.lstm_layer(x, hidden)
        lstm_last_out = lstm_out[-1]
        hidden2linear = self.linear_layer(lstm_last_out)
        predicted = F.log_softmax(hidden2linear)
        return predicted, lstm_hidden
    def init_hidden(self):
        return ( autograd.Variable(torch.zeros(1,1,self.hidden_dim)),
                 autograd.Variable(torch.zeros(1,1,self.hidden_dim)))

def accuracy_on_test_set( x_test, y_test ):
    right_answers = 0
    for index, data_entry in enumerate(x_test):
        data_entry = np.array(data_entry)
        data_entry = data_entry.astype(np.int32)
        data_entry = [ a.item() for a in data_entry ]
        target_data = y_test[index].item()
        input_data = autograd.Variable( torch.LongTensor(data_entry) )
        target_data = autograd.Variable( torch.LongTensor(target_data))
        hidden = lstm_imdb.init_hidden()
        y_pred, _ = lstm_imdb(input_data, hidden)
        value, predicted_index = torch.max( y_pred, 1 )
        predicted_value = predicted_index.data.numpy()[0]
        if predicted_value = target_value:
            right_answers += 1
    accuracy = right_answers / len(x_test)
    print( "Accuracy on test set: ", accuracy)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--epochs", type=int, default=1)
    parser.add_argument( "--num_words", type=int, default=5000)
    parser.add_argument( "--max_sequence_len", type=int, default=300)
    FLAGS, unparsed = parser.parse_known_args()

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=FLAGS.num_words,
                                                          maxlen=FLAGS.max_sequence_len)

    lstm_imdb = LSTMIMDB(100,50)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam( lstm_imdb.parameters(), lr=1e-3)

    epochs = FLAGS.epochs

    for i in range(epochs):
        loss_average = 0.0
        for index, data_entry in enumerate(X_train):
            data_entry = np.array( data_entry )
            data_entry = data_entry.astype(np.int32)
            data_entry = [ a.item() for a in data_entry ]
            y = y_train[index].item()

            input_sequence = autograd.Variable(torch.LongTensor(data_entry))
            y = autograd.Variable(torch.LongTensor([y]))
            hidden = lstm_imdb.init_hidden()
            y_pred, _ = lstm_imdb(input_sequence, hidden)
            lstm_imdb.zero_grad()
            loss = loss_function(y_pred, y)
            loss_average += loss.data[0]
            if index % 100 == 0:
                print( "epoch: %d iteration: %d loss: %g" %(i, index, loss.data[0]))
            loss.backward()
            optimizer.step()
        print( "Average loss: ", (loss_average/len(X_train)))
