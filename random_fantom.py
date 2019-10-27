import socket
import os
import collections
import logging
from logging.handlers import RotatingFileHandler
import json
import keras
from keras.models import Sequential
from keras.layers import Dense
import protocol
import numpy as np
from random import randrange
import random

characters = ['pink', 'blue', 'brown', 'red', 'black', 'white', 'purple', 'grey']
host = "localhost"
port = 12000
# HEADERSIZE = 10

"""
set up fantom logging
"""
fantom_logger = logging.getLogger()
fantom_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s :: %(levelname)s :: %(message)s", "%H:%M:%S")
# file
if os.path.exists("./logs/fantom.log"):
    os.remove("./logs/fantom.log")
file_handler = RotatingFileHandler('./logs/fantom.log', 'a', 1000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
fantom_logger.addHandler(file_handler)
# stream
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
fantom_logger.addHandler(stream_handler)


# class DQN:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = collections.deque(maxlen=2000)
#         self.gamma = 0.95  # discount rate
#         self.epsilon = 1.0  # exploration rate
#         self.epsilon_min = 0.2
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.model = self._build_model()
#         self.round_count = 0
#         self.total_round = 1000
#         self.train_start = 1000
#         self.batch_size = 32
#
#     def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Dense(24, input_dim=32, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse',
#                       optimizer=keras.optimizers.Adam(lr=self.learning_rate))
#         return model
#
#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
#
#     def act(self, state):
#         ret = 0
#         if np.random.rand() <= self.epsilon:
#             ret = random.randrange(self.action_size)
#         else:
#             act_values = self.model.predict(np.reshape(state, (1, 32)))
#             ret = np.argmax(act_values[0])
#         return ret  # returns action
#
#     def replay(self, batch_size):
#         batch_size = min(len(self.memory), batch_size)
#         minibatch = random.sample(self.memory, batch_size)
#         print("epsilon is now = ", self.epsilon)
#         print("ROUND --------> {}/{} ".format(self.round_count, self.total_round))
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = reward + self.gamma * np.amax(self.model.predict(np.reshape(next_state, (1, 32)))[0])
#             target_f = self.model.predict(np.reshape(state, (1, 32)))
#             target_f[0][action] = target
#         self.model.fit(np.reshape(state, (1, 32)), target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min and len(self.memory) > 50:
#             if self.round_count % (self.total_round / 10) == 0:
#                 self.epsilon -= 0.1
#
#     def train_model(self):
#         if len(self.memory) < self.train_start:
#             return
#         batch_size = min(self.batch_size, len(self.memory))
#         mini_batch = random.sample(self.memory, batch_size)
#
#         update_input = np.zeros((batch_size, self.state_size))
#         update_target = np.zeros((batch_size, self.state_size))
#         action, reward, done = [], [], []
#
#         for i in range(self.batch_size):
#             update_input[i] = mini_batch[i][0]
#             action.append(mini_batch[i][1])
#             reward.append(mini_batch[i][2])
#             update_target[i] = mini_batch[i][3]
#             done.append(mini_batch[i][4])
#
#         target = self.model.predict(update_input)
#         target_val = self.target_model.predict(update_target)
#
#         for i in range(self.batch_size):
#             # Q Learning: get maximum Q value at s' from target model
#             if done[i]:
#                 target[i][action[i]] = reward[i]
#             else:
#                 target[i][action[i]] = reward[i] + self.discount_factor * (
#                     np.amax(target_val[i]))
#
#         # and do the model fit!
#         self.model.fit(update_input, target, batch_size=self.batch_size,
#                        epochs=1, verbose=0)

class DQN:

    def __init__(self, state_size, action_size):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 500
        self.round_count = 0
        self.total_round = 10000
        # create replay memory using deque
        self.memory = collections.deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        # if self.load_model:
        #     self.model.load_weights("./save_model/cartpole_dqn.h5")

        # approximate Q function using Neural Network
        # state is input and Q Value of each action is output of network

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
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
            print("random")
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(np.reshape(state, (1, 32)))
            print("smart")
            return np.argmax(q_value[0])

        # save sample <s,a,r,s'> to the replay memory


    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


        # pick samples randomly from replay memory (with batch_size)


    def train_model(self):
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []
        print("minibatch shape = ", mini_batch[0][0])
        print("minibatch shape = ", update_input)
        print("batchsize = ", batch_size)
        print("len memory = ", len(self.memory))
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
        print("epsilon is now = ", self.epsilon)
        print("ROUND --------> {}/{} ".format(self.round_count, self.total_round))
        if self.epsilon > self.epsilon_min and len(self.memory) > 500:
            if self.round_count % (self.total_round / 10) == 0:
                self.epsilon -= 0.1


class EnvManager:
    def __init__(self):
        self.isFirstAction = True
        self.env_size = 32
        self.reward = 0
        self.carlotta_pos = [-1, -1]
        self.suspect_nbr = [8, 0]
        self.env = [[], []]

    def process_env(self, env):
        ret = [0] * self.env_size
        if env["question type"] == "select character" or env["question type"] == "Reset":
            self.set_env_info(env)
            for val in env["game state"]["characters"]:
                if val in env["data"]:
                    # if the character is in the question["data"] array it mean its a playable character thus 1
                    ret[characters.index(val["color"]) * 4] = 1
                # Set the is suspect value to int 0 = false, 1 = true
                ret[(characters.index(val["color"]) * 4) + 1] = int(val["suspect"])
                self.suspect_nbr[1] += int(val["suspect"]);
                # Set the character position
                ret[(characters.index(val["color"]) * 4) + 2] = val["position"]
                # Set the power value to int 0 = false, 1 = true
                ret[(characters.index(val["color"]) * 4) + 3] = int(val["power"])
            self.env[1] = ret
            self.calculate_reward()
        return ret

    def set_env_info(self, env):
        if self.carlotta_pos[0] == -1:
            self.carlotta_pos[0] = env["game state"]["position_carlotta"]
            self.carlotta_pos[1] = self.carlotta_pos[0]
        else:
            self.isFirstAction = False
            self.suspect_nbr[0] = self.suspect_nbr[1]
            self.suspect_nbr[1] = 0
            self.carlotta_pos[0] = self.carlotta_pos[1]
            self.carlotta_pos[1] = env["game state"]["position_carlotta"]
            self.env[0] = self.env[1]

    def calculate_reward(self):
        self.reward = 0
        # print("old pos = ", self.carlotta_pos[0])
        # print("new pos =", self.carlotta_pos[1])
        # print("suspects =", self.suspect_nbr)
        # print("environments =", self.env)
        # When the game start the carlotta position is - 1 before the first server message
        # pos[0] >=0 means this is not the first action so we can calculate the reward for the action played
        if self.carlotta_pos[0] >= 0:
            self.reward = ((self.carlotta_pos[1] - self.carlotta_pos[0]) - (3 * (self.suspect_nbr[0] - self.suspect_nbr[1])))
        # print("reward = ", self.reward)

    def reset(self):
        self.carlotta_pos = [-1, -1]
        self.suspect_nbr = [8, 0]
        self.env = [[], []]
        self.isFirstAction = True


class Player:
    def __init__(self):
        self.end = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.envManager = EnvManager()
        self.dqnAgent = DQN(self.envManager.env_size, 8)
        self.answerIdx = 0
        self.gameIteration = 0

    def connect(self):
        self.socket.connect((host, port))

    def reset(self):
        self.socket.close()

    def answer(self, question):
        # work
        data = question["data"]
        game_state = question["game state"]
        response_index = random.randint(0, len(data)-1)
        # log
        fantom_logger.debug("|\n|")
        fantom_logger.debug("fantom answers")
        fantom_logger.debug(f"question type ----- {question['question type']}")
        fantom_logger.debug(f"data -------------- {data}")
        fantom_logger.debug(f"response index ---- {response_index}")
        fantom_logger.debug(f"response ---------- {data[response_index]}")
        return response_index

    def smart_answer(self, data):
        if data["question type"] != "select character" and data["question type"] != "Reset":
            return self.answer(data)
        elif data["question type"] == "select character":
            if not self.envManager.isFirstAction:
                self.dqnAgent.append_sample(np.array(self.envManager.env[0]), self.answerIdx, self.envManager.reward, np.array(self.envManager.env[1]), self.end)
                self.dqnAgent.train_model()
                print("carlotta position = ", self.envManager.carlotta_pos)
                print("suspect nbr = ", self.envManager.suspect_nbr)
                print("reward = ", self.envManager.reward)
            self.answerIdx = self.dqnAgent.get_action(np.array(self.envManager.env[1]))
            while not self.validate_answer(self.answerIdx):
                print("learning ...")
                self.envManager.process_env(data)
                self.envManager.reward = -15
                self.dqnAgent.append_sample(np.array(self.envManager.env[0]), self.answerIdx, self.envManager.reward,
                                       np.array(self.envManager.env[1]), self.end)
                self.answerIdx = self.dqnAgent.get_action(np.array(self.envManager.env[1]))
            response = self.dqn2server_answer(data["data"], self.answerIdx)
            print("character to chose from", data["data"])
            print("character chose", data["data"][response])
            return response
        elif data["question type"] == "Reset":
            self.envManager.process_env(data)
            self.dqnAgent.append_sample(np.array(self.envManager.env[0]), self.answerIdx, self.envManager.reward,
                                  np.array(self.envManager.env[1]), self.end)
            #self.dqnAgent.replay(32)
            self.envManager.reset()
            self.dqnAgent.update_greedy()
            print("RESET ------------------------------------------------------------------------------ RESET")
            return 0



    def validate_answer(self, idx):
        return self.envManager.env[1][idx * 4] == 1


    def dqn2server_answer(self, question, idx):
        i = 0
        for val in question:
            if val["color"] == characters[idx]:
                return i
            i += 1

    def handle_json(self, data):
        data = json.loads(data)
        self.envManager.process_env(data)
        response = self.smart_answer(data)
        # send back to server
        bytes_data = json.dumps(response).encode("utf-8")
        protocol.send_json(self.socket, bytes_data)

    def run(self):

        self.connect()

        while self.end is not True:
            received_message = protocol.receive_json(self.socket)
            if received_message:
                self.handle_json(received_message)
            else:
                print("no message, finished learning")
                self.dqnAgent.model.save("model.h5")
                self.end = True


p = Player()
p.run()
