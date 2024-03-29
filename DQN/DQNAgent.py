import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import collections
import random
from keras.models import load_model


class DQNAgent:
    def __init__(self, state_size, action_size):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001
        self.batch_size = 64
        self.train_start = 32
        self.round_count = 0
        self.total_round = 100
        # create replay memory using deque
        self.memory = collections.deque(maxlen=100000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            q_value = np.random.randint(1, 101, self.action_size)
            print("random answer -> epsilon {}".format(self.epsilon))
            return q_value
        else:
            q_value = self.model.predict(np.reshape(state, (1, self.state_size)))
            #print("qvalues =", q_value)
            print("smart answer -> epsilon {}".format(self.epsilon))
            return q_value[0]

        # save sample <s,a,r,s'> to the replay memory

    def get_smart_action(self, state):
        q_value = self.model.predict(np.reshape(state, (1, self.state_size)))
        return q_value[0]

    def load_model(self, path):
        self.model = load_model(path)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []
        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def update_greedy(self):
        self.round_count += 1
        # print("epsilon is now = ", self.epsilon)
        # print("ROUND --------> {}/{} ".format(self.round_count, self.total_round))
        if self.epsilon > self.epsilon_min and len(self.memory) > 31:
            if self.round_count % (self.total_round / 20) == 0:
                self.epsilon = self.epsilon -0.049