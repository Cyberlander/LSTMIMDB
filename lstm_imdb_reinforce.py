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

    def forward(self, x):
        x = self.linear_layer_1( x )
        x = F.relu( x )
        action_scores = self.linear_layer_2( x )
        return F.softmax( action_scores, dim=1 )

    def reset(self):
        del self.saved_log_probs[:]

def choose_action( state ):
    probabilities = policy( state )
    categorical_distribution = Categorical( probabilities )
    action = categorical_distribution.sample()
    policy.saved_log_probs.append( categorical_distribution.log_prob( action ) )
    return action

def get_policy_loss(reward):
    """
    Returns policy loss

    :return:
    """
    policy_loss = []
    #we do not have any reward discounting factor
    #in learnign to skim paper the same reward is used for all actions
    rewards = torch.Tensor( [reward] * len(policy.saved_log_probs) )

    for reward, log_prob in zip( rewards, policy.saved_log_probs ):
        policy_loss.append( -log_prob * reward )

    policy_loss = torch.cat( policy_loss ).sum()
    return policy_loss



class LSTMIMDB(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super(LSTMIMDB, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding(FLAGS.num_words, embedding_dim)
        self.lstm_layer = nn.LSTM( embedding_dim, hidden_dim )
        self.linear_layer = nn.Linear( hidden_dim, 2)

    def forward(self, inputs, hidden ):
        inputs_rest = inputs
        loops = 0
        for i in range( NUMBER_OF_JUMPS_ALLOWED ):
            # if there are enough words to read to make a jump decision
            # NUMBER_OF_TOKENS_BETWEEN_JUMPS is the number of tokens
            # the lstm reads in before a jump decision
            if len( inputs_rest ) > NUMBER_OF_TOKENS_BETWEEN_JUMPS:
                loops += 1
                #we read NUMBER_OF_TOKENS_BETWEEN_JUMPS before next decision
                tokens_to_read = inputs_rest[:NUMBER_OF_TOKENS_BETWEEN_JUMPS]
                x = self.embedding_layer( tokens_to_read ).view(len(tokens_to_read),1,-1)
                lstm_out, lstm_hidden = self.lstm_layer( x, hidden )
                #we take output features from LSTM to predict action
                action = choose_action( lstm_out[-1] )
                action_int = int(action.data.numpy()[0])
                # is the position still in the sentence after the jump
                if action_int == 0:
                    #0 action should terminate right away
                    #the same as in paper
                    break

                if NUMBER_OF_TOKENS_BETWEEN_JUMPS + action_int < len( inputs_rest ):
                    inputs_rest = inputs_rest[NUMBER_OF_TOKENS_BETWEEN_JUMPS+action_int-1:]
                else:
                    break
            # there was already a jump and there a just a few words left
            elif len ( inputs_rest ) < NUMBER_OF_TOKENS_BETWEEN_JUMPS and loops > 0:
                break
                #x = self.embedding_layer( inputs_rest ).view(len( inputs_rest ),1,-1)
                #lstm_out, lstm_hidden = self.lstm_layer( x, hidden )
            else:
                x = self.embedding_layer(inputs_rest).view(len(inputs_rest),1,-1)
                lstm_out, lstm_hidden = self.lstm_layer(x, hidden)

        #print( label )
        lstm_last_out = lstm_out[-1]
        hidden2linear = self.linear_layer(lstm_last_out)
        predicted = F.softmax(hidden2linear)


        return predicted, lstm_last_out


    def init_hidden(self):
        return ( autograd.Variable(torch.zeros(1,1,self.hidden_dim)),
                 autograd.Variable(torch.zeros(1,1,self.hidden_dim)))


def calculate_reward(y_pred_proba, y_true):
    _, pred_label = torch.max(y_pred_proba, 1)
    pred_label = pred_label.data.numpy()[0]
    return 1.0 if pred_label == y_true else -1.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_words", type=int, default=5000)
    parser.add_argument("--max_sequence_len", type=int, default=300)
    parser.add_argument("--max_grad_norm", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lr", type=float, default=1e-4)

    FLAGS, unparsed = parser.parse_known_args()
    #fixing random seed for the experiment
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=FLAGS.num_words,
                                                          maxlen=FLAGS.max_sequence_len, seed=FLAGS.seed)

    LSTM_HIDDEN_DIM = 100

    NUMBER_OF_JUMPS_ALLOWED = 10
    NUMBER_OF_TOKENS_BETWEEN_JUMPS = 3
    MAX_JUMP_SIZE = 5

    lstm_imdb = LSTMIMDB(LSTM_HIDDEN_DIM,50)

    policy = Policy( LSTM_HIDDEN_DIM, MAX_JUMP_SIZE )

    ce_criterion = nn.CrossEntropyLoss()

    total_parameters = list(lstm_imdb.parameters()) + list(policy.parameters())
    optimizer = optim.Adam(total_parameters , lr=FLAGS.lr)

    epochs = FLAGS.epochs

    for i in range(epochs):
        loss_average = 0.0
        for index, data_entry in enumerate(X_train):
            optimizer.zero_grad()

            data_entry = np.array( data_entry )
            data_entry = data_entry.astype(np.int32)
            data_entry = [ a.item() for a in data_entry ]
            y = y_train[index].item()

            input_sequence = autograd.Variable(torch.LongTensor(data_entry))
            y = autograd.Variable(torch.LongTensor([y]))
            hidden = lstm_imdb.init_hidden()
            y_pred_proba, _ = lstm_imdb(input_sequence, hidden)

            reward = calculate_reward(y_pred_proba, y_train[index])
            policy_loss =  get_policy_loss(reward)

            nll_loss =  ce_criterion(y_pred_proba, y)
            total_loss = nll_loss + policy_loss

            loss_average += total_loss.data[0]
            if index % 100 == 0:
                print( "epoch: %d iteration: %d CE loss: %g Policy Loss: %g" %(i, index,
                                                                                nll_loss.data[0], policy_loss.data[0]))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm( total_parameters, max_norm=FLAGS.max_grad_norm)
            optimizer.step()

            policy.reset()
        print( "Average loss: ", (loss_average/len(X_train)))
