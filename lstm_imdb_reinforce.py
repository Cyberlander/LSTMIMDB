from keras.datasets import imdb
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import argparse

class Policy(nn.Module):
    def __init__( self, lstm_hidden_size, max_jump_size ):
        super(Policy, self).__init__()
        #max_jump_size += 1
        policy_hidden_size = int(( lstm_hidden_size + max_jump_size ) / 2)
        self.linear_layer_1 = nn.Linear( lstm_hidden_size, policy_hidden_size )
        self.linear_layer_2 = nn.Linear( policy_hidden_size, max_jump_size )

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.linear_layer_1( x )
        x = F.relu( x )
        action_scores = self.linear_layer_2( x )
        return F.softmax( action_scores, dim=1 )

def choose_action( state ):
    probabilities = policy( state )
    categorical_distribution = Categorical( probabilities.view( 1, -1) )
    action = categorical_distribution.sample()
    policy.saved_log_probs.append( categorical_distribution.log_prob( action ) )
    return action

def finish_episode():
    R = 0
    rewards = []
    policy_loss = []
    for r in policy.rewards[::-1]:
        R = r + 0.99 * R
        rewards.insert(0,R)
    rewards = torch.Tensor( rewards )
    # computing the z-score
    rewards = ( rewards - rewards.mean() ) / ( rewards.std() + np.finfo(np.float32).eps)
    for reward, log_prob in zip( rewards, policy.saved_log_probs ):
        policy_loss.append( -log_prob * reward )

    optimizer.zero_grad()
    policy_loss = torch.cat( policy_loss ).sum()
    torch.nn.utils.clip_grad_norm( policy.parameters(), True)
    policy_loss.backward(retain_graph=True)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

class LSTMIMDB(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super(LSTMIMDB, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding(FLAGS.num_words, embedding_dim)
        self.lstm_layer = nn.LSTM( embedding_dim, hidden_dim )
        self.linear_layer = nn.Linear( hidden_dim, 2)
    def forward(self,inputs, hidden, label ):
        inputs_rest = inputs
        loops = 0
        for i in range( NUMBER_OF_JUMPS_ALLOWED ):
            # if there are enough words to read to make a jump decision
            # NUMBER_OF_TOKENS_BETWEEN_JUMPS is the number of tokens
            # the lstm reads in before a jump decision
            if len( inputs_rest ) > NUMBER_OF_TOKENS_BETWEEN_JUMPS:
                loops += 1
                tokens_to_read = inputs_rest[:NUMBER_OF_JUMPS_ALLOWED]
                x = self.embedding_layer( tokens_to_read ).view(len(tokens_to_read),1,-1)
                lstm_out, lstm_hidden = self.lstm_layer( x, hidden )
                last_hidden_state = lstm_hidden[-1]
                action = choose_action( last_hidden_state )
                action_int = int(action.data.numpy()[0])
                # is the position still in the sentence after the jump
                if NUMBER_OF_TOKENS_BETWEEN_JUMPS + action_int < len( inputs_rest ):
                    inputs_rest = inputs_rest[NUMBER_OF_TOKENS_BETWEEN_JUMPS+action_int-1:]
                else:
                    break
            # there was already a jump and there a just a few words left
            elif len ( inputs_rest ) < NUMBER_OF_TOKENS_BETWEEN_JUMPS and loops > 0:
                break;
                #x = self.embedding_layer( inputs_rest ).view(len( inputs_rest ),1,-1)
                #lstm_out, lstm_hidden = self.lstm_layer( x, hidden )
            else:
                x = self.embedding_layer(inputs_rest).view(len(inputs_rest),1,-1)
                lstm_out, lstm_hidden = self.lstm_layer(x, hidden)

        #print( label )
        lstm_last_out = lstm_out[-1]
        hidden2linear = self.linear_layer(lstm_last_out)
        predicted = F.log_softmax(hidden2linear)
        value, index = torch.max( predicted, 1 )
        predicted_int = int(index.data.numpy()[0])
        if predicted_int == int(label):
            reward = 1
        else:
            reward = -1
        # compute reward for every step
        rewards = [reward] * len(policy.saved_log_probs)
        policy.rewards.extend( rewards )
        #print( rewards )
        finish_episode()

        return predicted, lstm_hidden
    def init_hidden(self):
        return ( autograd.Variable(torch.zeros(1,1,self.hidden_dim)),
                 autograd.Variable(torch.zeros(1,1,self.hidden_dim)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--epochs", type=int, default=10)
    parser.add_argument( "--num_words", type=int, default=5000)
    parser.add_argument( "--max_sequence_len", type=int, default=300)
    FLAGS, unparsed = parser.parse_known_args()

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=FLAGS.num_words,
                                                          maxlen=FLAGS.max_sequence_len)

    LSTM_HIDDEN_DIM = 100
    NUMBER_OF_JUMPS_ALLOWED = 2
    NUMBER_OF_TOKENS_BETWEEN_JUMPS = 3
    MAX_JUMP_SIZE = 5

    lstm_imdb = LSTMIMDB(LSTM_HIDDEN_DIM,50)

    policy = Policy( LSTM_HIDDEN_DIM, MAX_JUMP_SIZE )

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam( lstm_imdb.parameters(), lr=1e-5)

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
            y_pred, _ = lstm_imdb(input_sequence, hidden, y_train[index])
            lstm_imdb.zero_grad()
            loss = loss_function(y_pred, y)


            #if np.isnan( loss.data.numpy()[0]):
                #fill_index = autograd.Variable( torch.LongTensor([0]) )
                #loss.index_fill_(0,fill_index,0.01 )

            loss_average += loss.data[0]
            if index % 100 == 0:
                print( "epoch: %d iteration: %d loss: %g" %(i, index, loss.data[0]))

            loss.backward()
            torch.nn.utils.clip_grad_norm( lstm_imdb.parameters(), True)
            optimizer.step()
        print( "Average loss: ", (loss_average/len(X_train)))
