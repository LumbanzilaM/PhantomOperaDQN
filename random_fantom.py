import socket
import os
import logging
from logging.handlers import RotatingFileHandler
import json
import protocol
import numpy as np
import random
from DQNAgents import DQNAgent
import EnvManagers

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


class Player:
    def __init__(self):
        self.end = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.envManager = EnvManagers.CharacterEnvManager()
        self.dqnAgent = DQNAgent(self.envManager.env_size, 8)
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
        return self.envManager.env[1][idx * 5] == 1


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
