from abc import ABC, abstractmethod
import DQNAgents
import utils as u
import numpy as np
import collections

characters = ['pink', 'blue', 'brown', 'red', 'black', 'white', 'purple', 'grey']

class AEnvManager(ABC):
    def __init__(self):
        self.isFirstAction = True
        self.env_size = 43
        self.output_size = 8
        self.reward = 0
        self.carlotta_pos = [-1, -1]
        self.suspect_nbr = [8, 0]
        self.env = [[], []]
        self.dqnAgent = DQNAgents.DQNAgent(self.env_size, self.output_size)
        self.phantom_color = -1
        self.answerIdx = -1
        self.previous_env = None
        self.question = None

    @abstractmethod
    def process_env(self, env):
        pass

    @abstractmethod
    def set_env_info(self, env):
        self.question = env[u.QUESTION]
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

    @abstractmethod
    def validate_answer(self):
        pass

    @abstractmethod
    def save_training(self):
        pass

    @abstractmethod
    def dqn2server_answer(self, question):
        pass

    def calculate_reward(self, data):
        if self.carlotta_pos[0] == self.carlotta_pos[1] and self.suspect_nbr[0] == self.suspect_nbr[1]:
            self.count_suspect()
        self.reward = 0
        if not self.question == u.RESET:
            if self.carlotta_pos[0] >= 0:
                self.reward = ((self.carlotta_pos[1] - self.carlotta_pos[0]) - (3 * (self.suspect_nbr[0] - self.suspect_nbr[1])))
        else:
            self.reward = data[0]
            print("Last reward = ", self.reward)

    def count_suspect(self):
        ret = 0
        for i in range(8):
            ret = ret + self.env[1][i * 5]
        self.suspect_nbr[1] = ret

    def reset(self):
        print()
        self.carlotta_pos = [-1, -1]
        self.suspect_nbr = [8, 0]
        self.env = [[], []]
        self.isFirstAction = True

    def get_action(self, env, smart=False):
        if not smart:
            self.answerIdx = self.dqnAgent.get_action(env)
        else:
            self.answerIdx = self.dqnAgent.get_smart_action(env)

    def append_sample(self, end):
        self.dqnAgent.append_sample(self.env[0], self.answerIdx, self.reward, self.env[1], end)

    def wrong_answer(self):
        self.reward = -100

class CharacterEnvManager(AEnvManager):
    def __init__(self):
        super().__init__()

    def process_env(self, env):
        ret = [0] * self.env_size
        if env[u.QUESTION] == u.CHAR_SELECT or env[u.QUESTION] == u.RESET:
            self.set_env_info(env)
            for val in env[u.GAME_STATE][u.CHARACTERS]:
                if env[u.QUESTION] == u.CHAR_SELECT and val in env[u.DATA]:
                    # if the character is in the question["data"] array it mean its a playable character thus 1
                    ret[characters.index(val[u.COLOR]) * 5] = 1
                # Set the is suspect value to int 0 = false, 1 = true
                ret[(characters.index(val[u.COLOR]) * 5) + 1] = int(val[u.SUSPECT])
                self.suspect_nbr[1] += int(val[u.SUSPECT])
                # Set the character position
                ret[(characters.index(val[u.COLOR]) * 5) + 2] = val[u.POSITION]
                # Set the power value to int 0 = false, 1 = true
                ret[(characters.index(val[u.COLOR]) * 5) + 3] = int(val[u.POWER])
            if self.phantom_color == -1:
                self.phantom_color = env[u.GAME_STATE][u.FANTOM]
            ret[(characters.index(self.phantom_color) * 5) + 4] = -1
            ret[self.env_size - 3] = env[u.GAME_STATE][u.SHADOW]
            ret[self.env_size - 2] = env[u.GAME_STATE][u.BLOCKED][0]
            ret[self.env_size - 1] = env[u.GAME_STATE][u.BLOCKED][1]
            print("phantom is  = ", self.phantom_color)
            self.env[1] = ret
            self.calculate_reward(env[u.DATA])
        return ret

    def validate_answer(self):
        return self.env[1][self.answerIdx * 5] == 1

    def set_env_info(self, env):
        super().set_env_info(env)

    def save_training(self):
        self.dqnAgent.model.save("tmp_models/character_picker.h5")

    def dqn2server_answer(self, question):
        i = 0
        for val in question:
            if val["color"] == characters[self.answerIdx]:
                return i
            i += 1


class PositionEnvManager(AEnvManager):
    def __init__(self):
        super().__init__()
        self.env_size = 53
        self.output_size = 10
        self.dqnAgent = DQNAgents.DQNAgent(self.env_size, self.output_size)

    def process_env(self, env):
        ret = [0] * self.env_size
        if env[u.QUESTION] == u.POS_SELECT or env[u.QUESTION] == u.RESET:
            self.set_env_info(env)
            ret[characters.index(characters[self.previous_env.answerIdx]) * 5] = 1
            for val in env[u.GAME_STATE][u.CHARACTERS]:
                # Set the is suspect value to int 0 = false, 1 = true
                ret[(characters.index(val[u.COLOR]) * 5) + 1] = int(val[u.SUSPECT])
                self.suspect_nbr[1] += int(val[u.SUSPECT])
                # Set the character position
                ret[(characters.index(val[u.COLOR]) * 5) + 2] = val[u.POSITION]
                # Set the power value to int 0 = false, 1 = true
                ret[(characters.index(val[u.COLOR]) * 5) + 3] = int(val[u.POWER])
            print(u.GAME_STATE, env[u.GAME_STATE][u.SHADOW])
            if env[u.QUESTION] == u.POS_SELECT:
                for val in env[u.DATA]:
                    ret[40 + val] = -2
            if self.phantom_color == -1:
                self.phantom_color = env[u.GAME_STATE][u.FANTOM]
            ret[(characters.index(self.phantom_color) * 5) + 4] = -1
            ret[self.env_size - 3] = env[u.GAME_STATE][u.SHADOW]
            ret[self.env_size - 2] = env[u.GAME_STATE][u.BLOCKED][0]
            ret[self.env_size - 1] = env[u.GAME_STATE][u.BLOCKED][1]
            print("pos ret =", ret)
            print("selected character =", characters[self.previous_env.answerIdx])
            print("possible pos = ", env[u.DATA])
            self.env[1] = ret
            self.calculate_reward(env[u.DATA])
        return ret

    def validate_answer(self):
        return self.env[1][40 + self.answerIdx] == -2

    def set_env_info(self, env):
        super().set_env_info(env)

    def save_training(self):
        self.dqnAgent.model.save("tmp_models/position_picker.h5")

    def dqn2server_answer(self, question):
        i = 0
        for val in question:
            if val == self.answerIdx:
                return i
            i += 1


class CharacterEnvManagerV2(AEnvManager):
    def __init__(self):
        super().__init__()
        self.dqnAgent = DQNAgents.DQNAgentV2(self.env_size, self.output_size)

    def process_env(self, env):
        ret = [0] * self.env_size
        if env[u.QUESTION] == u.CHAR_SELECT or env[u.QUESTION] == u.RESET:
            self.set_env_info(env)
            for val in env[u.GAME_STATE][u.CHARACTERS]:
                if env[u.QUESTION] == u.CHAR_SELECT and val in env[u.DATA]:
                    # if the character is in the question["data"] array it mean its a playable character thus 1
                    ret[characters.index(val[u.COLOR]) * 5] = 1
                # Set the is suspect value to int 0 = false, 1 = true
                ret[(characters.index(val[u.COLOR]) * 5) + 1] = int(val[u.SUSPECT])
                self.suspect_nbr[1] += int(val[u.SUSPECT])
                # Set the character position
                ret[(characters.index(val[u.COLOR]) * 5) + 2] = val[u.POSITION]
                # Set the power value to int 0 = false, 1 = true
                ret[(characters.index(val[u.COLOR]) * 5) + 3] = int(val[u.POWER])
            if self.phantom_color == -1:
                self.phantom_color = env[u.GAME_STATE][u.FANTOM]
            ret[(characters.index(self.phantom_color) * 5) + 4] = -1
            ret[self.env_size - 3] = env[u.GAME_STATE][u.SHADOW]
            ret[self.env_size - 2] = env[u.GAME_STATE][u.BLOCKED][0]
            ret[self.env_size - 1] = env[u.GAME_STATE][u.BLOCKED][1]
            print("phantom is  = ", self.phantom_color)
            self.env[1] = ret
            self.calculate_reward(env[u.DATA])
        return ret

    def validate_answer(self):
        return self.env[1][self.answerIdx * 5] == 1

    def set_env_info(self, env):
        super().set_env_info(env)

    def save_training(self):
        self.dqnAgent.model.save("tmp_models/character_picker.h5")

    def get_action(self, env, smart=False):
        values = np.array(self.dqnAgent.get_action(env))
        #values = [0, 2, 3, 5]
        for i in range(self.output_size):
            self.answerIdx = np.argmax(values)
            if not self.validate_answer():
                values[self.answerIdx] = -10000

    def dqn2server_answer(self, question):
        i = 0
        for val in question:
            if val["color"] == characters[self.answerIdx]:
                return i
            i += 1



class PositionEnvManagerV2(AEnvManager):
    def __init__(self):
        super().__init__()
        self.env_size = 53
        self.output_size = 10
        self.dqnAgent = DQNAgents.DQNAgentV2(self.env_size, self.output_size)

    def process_env(self, env):
        ret = [0] * self.env_size
        if env[u.QUESTION] == u.POS_SELECT or env[u.QUESTION] == u.RESET:
            self.set_env_info(env)
            print(" previous answer idx=", characters[self.previous_env.answerIdx])
            ret[characters.index(characters[self.previous_env.answerIdx]) * 5] = 1
            for val in env[u.GAME_STATE][u.CHARACTERS]:
                # Set the is suspect value to int 0 = false, 1 = true
                ret[(characters.index(val[u.COLOR]) * 5) + 1] = int(val[u.SUSPECT])
                self.suspect_nbr[1] += int(val[u.SUSPECT])
                # Set the character position
                ret[(characters.index(val[u.COLOR]) * 5) + 2] = val[u.POSITION]
                # Set the power value to int 0 = false, 1 = true
                ret[(characters.index(val[u.COLOR]) * 5) + 3] = int(val[u.POWER])
            print(u.GAME_STATE, env[u.GAME_STATE][u.SHADOW])
            if env[u.QUESTION] == u.POS_SELECT:
                for val in env[u.DATA]:
                    ret[40 + val] = -2
            if self.phantom_color == -1:
                self.phantom_color = env[u.GAME_STATE][u.FANTOM]
            ret[(characters.index(self.phantom_color) * 5) + 4] = -1
            ret[self.env_size - 3] = env[u.GAME_STATE][u.SHADOW]
            ret[self.env_size - 2] = env[u.GAME_STATE][u.BLOCKED][0]
            ret[self.env_size - 1] = env[u.GAME_STATE][u.BLOCKED][1]
            print("pos ret =", ret)
            print("selected character =", characters[self.previous_env.answerIdx])
            print("possible pos = ", env[u.DATA])
            self.env[1] = ret
            self.calculate_reward(env[u.DATA])
        return ret

    def validate_answer(self):
        return self.env[1][40 + self.answerIdx] == -2

    def set_env_info(self, env):
        super().set_env_info(env)

    def save_training(self):
        self.dqnAgent.model.save("tmp_models/position_picker.h5")

    def dqn2server_answer(self, question):
        i = 0
        for val in question:
            if val == self.answerIdx:
                return i
            i += 1

    def get_action(self, env, smart=False):
        values = np.array(self.dqnAgent.get_action(env))
        for i in range(self.output_size):
            self.answerIdx = np.argmax(values)
            if not self.validate_answer():
                values[self.answerIdx] = -10000
