from abc import ABC, abstractmethod
import DQN.DQNAgent as Dqn
import utils as u
import numpy as np


class AEnvManager(ABC):
    def __init__(self, is_smart=False):
        self.smart = is_smart
        self.phase_nbr = 0
        self.carlotta_pos = [22, 0]
        self.suspect_nbr = [8, 0]
        self.reward = 0
        self.answerIdx = -1
        self.phantom_color = u.characters[0]
        self.phantom_position = 0
        self.selected_character = u.characters[0]
        self.previous_env = None
        self.question = None
        self.env_size, self.output_size = self._set_model_scope()
        self.model_name = self._set_model_name()
        self.env = [[0] * self.env_size, [0] * self.env_size]
        self.dqnAgent = Dqn.DQNAgent(self.env_size, self.output_size)
        if self.smart:
            self.dqnAgent.load_model(self.model_name)

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
            self.phantom_position = ret[(u.characters.index(self.phantom_color) * 5) + 2]
            ret[(u.characters.index(self.phantom_color) * 5) + 4] = -1
        ret[self.env_size - 3] = env[u.GAME_STATE][u.SHADOW]
        ret[self.env_size - 2] = env[u.GAME_STATE][u.BLOCKED][0]
        ret[self.env_size - 1] = env[u.GAME_STATE][u.BLOCKED][1]
        print("phantom is  = ", self.phantom_color)
        return ret

    def _set_starting_env(self, env):
        print("Set {} starting env".format(self.model_name))
        self.env[0] = self._format_env2dqn(env)
        self._set_env_info(env, 0)

    def _set_ending_env(self, env):
        self.env[1] = self._format_env2dqn(env)
        self._set_env_info(env, 1)
        self._calculate_reward(env)
        print("Set {} ending env with reward {}".format(self.model_name, self.reward))

    def _count_suspects(self, env):
        pos_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for val in env[u.GAME_STATE][u.CHARACTERS]:
            pos_dict[val[u.POSITION]] = pos_dict[val[u.POSITION]] + 1
        do_scream = pos_dict[self.phantom_position] == 1 or self.phantom_position == env[u.GAME_STATE][u.SHADOW]
        suspect = 0
        if do_scream:
            for val in env[u.GAME_STATE][u.CHARACTERS]:
                if val[u.SUSPECT] == 1:
                    if pos_dict[val[u.POSITION]] == 1 or val[u.POSITION] == self.phantom_position \
                            or val[u.POSITION] == env[u.GAME_STATE][u.SHADOW]:
                        suspect = suspect + 1
        else:
            for val in env[u.GAME_STATE][u.CHARACTERS]:
                if val[u.SUSPECT] == 1:
                    if pos_dict[val[u.POSITION]] > 1 and val[u.POSITION] != env[u.GAME_STATE][u.SHADOW]:
                        suspect = suspect + 1
        # print("game state =", env[u.GAME_STATE])
        # print("phantom is {} pos {}".format(self.phantom_color, self.phantom_position))
        # print("dict =", dict)
        # print("suspect =", suspect)
        return suspect

    def _set_env_info(self, env, idx):
        self.question = env[u.QUESTION]
        self.carlotta_pos[idx] = env[u.GAME_STATE][u.CARLOTTA_POS]
        self.suspect_nbr[idx] = self._count_suspects(env)

    def _calculate_reward(self, env):
        data = env[u.DATA]
        if self.suspect_nbr[1] == 1:
            self.reward = -100
        elif self.suspect_nbr[0] - self.suspect_nbr[1] <= 0:
            self.reward = 50 + (50 * (self.suspect_nbr[1] - self.suspect_nbr[0]))
        else:
            self.reward = -(12.25 * (self.suspect_nbr[0] - self.suspect_nbr[1]))
        #print("phantom is {} pos {}".format(self.phantom_color, self.phantom_position))
        #print("last suspect =", self.suspect_nbr[0])
        #print("new suspect =", self.suspect_nbr[1])
        #print("reward =", self.reward)

    def _append_sample(self, end):
        self.dqnAgent.append_sample(self.env[0], self.answerIdx, self.reward, self.env[1], end)

    def _get_smart_action(self, env):
        self._set_starting_env(env)
        values = np.array(self.dqnAgent.get_smart_action(self.env[0]))
        for i in range(self.output_size):
            self.answerIdx = np.argmax(values)
            if not self._validate_answer(env[u.ANSWER]):
                print("WRONG SMART ANSWER ->", values)
                values[self.answerIdx] = -10000
        print("Learning state ->", values)
        return self._dqn2server_answer(env[u.ANSWER])

    def _get_learning_action(self, env):
        self._set_starting_env(env)
        while True:
            values = self.dqnAgent.get_action(self.env[0])
            self.answerIdx = np.argmax(values)
            if self._validate_answer(env[u.ANSWER]):
                break
            self._set_ending_env(env)
            self._wrong_answer()
            self._append_sample(False)
            self.dqnAgent.train_model()
            print("Manager {} learning wih actual values {}".format(self.model_name, values))
            print("With actual state {} ".format(env))
        print("Learning state ->", values)
        return self._dqn2server_answer(env[u.ANSWER])

    def _wrong_answer(self):
        self.reward = -100

    def get_action(self, env):
        print("Get {} action".format(self.model_name))
        return self._get_smart_action(env) if self.smart else self._get_learning_action(env)

    def learn(self, env, is_end):
        if not self.smart:
            self._set_ending_env(env)
            self._append_sample(is_end)
            self.dqnAgent.train_model()

    def save_training(self):
        if not self.smart:
            self.dqnAgent.model.save(self.model_name)

    def game_end(self, env):
        self._set_ending_env(env)
        self.reward = env[u.DATA][0]
        self._append_sample(True)
        self.dqnAgent.update_greedy()

    def reset(self):
        self.answerIdx = -1

    @abstractmethod
    def get_info_from_previous_env(self, previous_env):
        pass
