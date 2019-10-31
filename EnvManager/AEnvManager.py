from abc import ABC, abstractmethod
import DQN.DQNAgent as Dqn
import utils as u
import numpy as np


class AEnvManager(ABC):
    def __init__(self):
        self.carlotta_pos = [22, 0]
        self.suspect_nbr = [8, 0]
        self.reward = 0
        self.answerIdx = 0
        self.phantom_color = u.characters[0]
        self.selected_character = u.characters[0]
        self.previous_env = None
        self.question = None
        self.env_size, self.output_size = self._set_model_scope()
        self.model_name = self._set_model_name()
        self.env = [[0] * self.env_size, [0] * self.env_size]
        self.dqnAgent = Dqn.DQNAgent(self.env_size, self.output_size)

    @abstractmethod
    def _set_model_name(self):
        pass

    @abstractmethod
    def _validate_answer(self, answers):
        pass

    @abstractmethod
    def _dqn2server_answer(self, answers):
        pass

    @abstractmethod
    def _set_model_scope(self):
        pass

    @abstractmethod
    def _format_env2dqn(self, env):
        ret = [0] * self.env_size
        for val in env[u.GAME_STATE][u.CHARACTERS]:
            # Set the is suspect value to int 0 = false, 1 = true
            ret[(u.characters.index(val[u.COLOR]) * 5) + 1] = int(val[u.SUSPECT])
            # Set the character position
            ret[(u.characters.index(val[u.COLOR]) * 5) + 2] = val[u.POSITION]
            # Set the power value to int 0 = false, 1 = true
            ret[(u.characters.index(val[u.COLOR]) * 5) + 3] = int(val[u.POWER])
        if u.FANTOM in env[u.GAME_STATE]:
            self.phantom_color = env[u.GAME_STATE][u.FANTOM]
            ret[(u.characters.index(self.phantom_color) * 5) + 4] = -1
        ret[self.env_size - 3] = env[u.GAME_STATE][u.SHADOW]
        ret[self.env_size - 2] = env[u.GAME_STATE][u.BLOCKED][0]
        ret[self.env_size - 1] = env[u.GAME_STATE][u.BLOCKED][1]
        # print("phantom is  = ", self.phantom_color)
        return ret

    def _set_starting_env(self, env):
        self.env[0] = self._format_env2dqn(env)
        self._set_env_info(env, 0)
        print("env sta", self.env[0])

    def _set_ending_env(self, env):
        self.env[1] = self._format_env2dqn(env)
        self._set_env_info(env, 1)
        print("env end", self.env[1])
        self._calculate_reward(env[u.DATA])

    def _count_suspect(self, idx):
        ret = 0
        for i in range(len(u.characters)):
            ret = ret + self.env[idx][(i * 5) + 1]
        return ret

    def _set_env_info(self, env, idx):
        self.question = env[u.QUESTION]
        self.carlotta_pos[idx] = env[u.GAME_STATE][u.CARLOTTA_POS]
        self.suspect_nbr[idx] = self._count_suspect(idx)

    def _calculate_reward(self, data):
        if not self.question == u.RESET:
            self.reward = 10 - (10 * (self.suspect_nbr[0] - self.suspect_nbr[1]))
        else:
            self.reward = data[0]
            # print("Last reward = ", self.reward)

    def _append_sample(self, end):
        print("reward at append =", self.reward)
        self.dqnAgent.append_sample(self.env[0], self.answerIdx, self.reward, self.env[1], end)

    def _wrong_answer(self):
        self.reward = -100

    def get_action(self, env):
        while True:
            self._set_starting_env(env)
            self.answerIdx = self.dqnAgent.get_action(self.env[0])
            if self._validate_answer(env[u.ANSWER]):
                break
            print("learning ... ", self.model_name)
            self._set_ending_env(env)
            self._wrong_answer()
            self._append_sample(False)
            self.dqnAgent.train_model()
        return self._dqn2server_answer(env[u.ANSWER])

    def get_smart_action(self, env):
        self.answerIdx = self.dqnAgent.get_smart_action(self.env[0])
        return self._dqn2server_answer(env[u.ANSWER])

    def learn(self, env, is_end):
        self._set_ending_env(env)
        self._append_sample(is_end)
        self.dqnAgent.train_model()

    def save_training(self):
        self.dqnAgent.model.save(self.model_name)

    def game_end(self, env):
        self._set_ending_env(env)
        self.reward = env[u.DATA][0]
        self._append_sample(True)
        self.dqnAgent.update_greedy()

    @abstractmethod
    def get_info_from_previous_env(self, previous_env):
        pass
