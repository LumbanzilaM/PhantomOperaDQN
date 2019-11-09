import socket
import os
import logging
from logging.handlers import RotatingFileHandler
import json
import protocol
import numpy as np
import random
from DQNAgents import DQNAgent
import EnvManager
import utils as u
from collections import namedtuple

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
        self.answerIdx = 0
        self.gameIteration = 0
        self.envManagers = self.init_dictionnary()
        self.last_env = None

    def init_dictionnary(self):

        dic = {u.CHAR_SELECT: EnvManager.CharacterEnvManagerV2(),
               u.POS_SELECT: EnvManager.PositionEnvManagerV2()}
        return dic

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
        question = data[u.QUESTION]
        #print("question = ", question)
        if question != u.CHAR_SELECT and question != u.POS_SELECT and question != u.RESET:
            #print("Answer to chose from", data["data"])
            #print("Answer chose", data["data"][0])
            return 1
        elif question == u.RESET:
            for key, envManager in self.envManagers.items():
                envManager.process_env(data)
                envManager._append_sample(True)
                #print("last carlotta position = ", envManager.carlotta_pos)
                #print("last suspect nbr = ", envManager.suspect_nbr)
                # self.dqnAgent.replay(32)
                envManager.reset()
                envManager.dqnAgent.update_greedy()
            #print("RESET ------------------------------------------------------------------------------ RESET")
            return 0
        else:
            envManager = self.envManagers[question]
            envManager.previous_env = self.last_env
            envManager.process_env(data)
            if not envManager.isFirstAction:
                envManager._append_sample(False)
                envManager.dqnAgent.train_model()
                #print("carlotta position = ", envManager.carlotta_pos)
                #print("suspect nbr = ", envManager.suspect_nbr)
                #print("reward = ", envManager.reward)
            envManager.get_action(np.array(envManager.env[1]))
            self.last_env = envManager
            response = envManager._dqn2server_answer(data["data"])
            #print("Answer to chose from", data["data"])
            #print("Answer chose", data["data"][response])
            return response

    def handle_json(self, data):
        data = json.loads(data)
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
                #print("no message, finished learning")
                for key, envManager in self.envManagers.items():
                    envManager.save_training()
                self.end = True


p = Player()
p.run()
