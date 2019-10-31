import socket
import os
import logging
from logging.handlers import RotatingFileHandler
import json
import protocol
import numpy as np
import random
import utils as u
import EnvManager.Phantom as phantomManager

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


class Phantom:
    def __init__(self):
        self.end = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.answerIdx = 0
        self.gameIteration = 0
        self.envManagers = self.init_dictionnary()
        self.last_env = None

    def init_dictionnary(self):

        dic = {
               u.POS_SELECT: phantomManager.PositionEnvManager()}
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
        print("question = ", question)
        if question == u.RESET:
            self.last_env.game_end(data)
            self.last_env = None
            print("RESET ------------------------------------------------------------------------------ RESET")
            return 0
        elif question not in self.envManagers:
            return 0
        else:
            manager = self.envManagers[question]
            if self.last_env is not None:
                self.last_env.learn(data, False)
                manager.get_info_from_previous_env(self.last_env)
            response = manager.get_action(data)
            self.last_env = manager
            print("Answer to chose from", data["data"])
            print("Answer chose", data["data"][response])
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
                print("no message, finished learning")
                for key, envManager in self.envManagers.items():
                    envManager.save_training()
                self.end = True


p = Phantom()
p.run()
