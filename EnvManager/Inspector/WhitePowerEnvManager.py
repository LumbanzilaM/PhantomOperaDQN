from EnvManager import AInspectorEnvManager
import numpy as np
import utils as u


class WhiteEnvManager(AInspectorEnvManager):
    def __init__(self, is_smart=False):
        super().__init__(is_smart)

    def _set_model_name(self):
        return "i_white_power.h5"

    def _validate_answer(self, answers):
        #print("Answer idx =", self.answerIdx)
        return self.answerIdx in answers

    def _dqn2server_answer(self, answers):
        i = 0
        for val in answers:
            if val == self.answerIdx:
                return i
            i += 1

    def _format_env2dqn(self, env):
        ret = super()._format_env2dqn(env)
        ret[u.characters.index(self.selected_character) * 5] = 1
        if env[u.QUESTION] == u.WHITE_POWER_USE:
            for val in env[u.DATA]:
                ret[40 + val] = -2
        return ret

    def _set_model_scope(self):
        return 53, 10

    def get_info_from_previous_env(self, previous_env):
        self.selected_character = previous_env.selected_character

    def _format_env2dqn(self, env):
        ret = super()._format_env2dqn(env)
        ret[u.characters.index(self.selected_character) * 5] = 1
        return ret

    def _calculate_reward(self, env):
        data = env[u.DATA]
        if self.suspect_nbr[1] == 1:
            self.reward = -100
        elif self.suspect_nbr[0] == self.suspect_nbr[1]:
            self.reward = 0
        elif self.suspect_nbr[0] - self.suspect_nbr[1] < 0:
            self.reward = 50 + (50 * (self.suspect_nbr[1] - self.suspect_nbr[0]))
        else:
            self.reward = -(12.25 * (self.suspect_nbr[0] - self.suspect_nbr[1]))

    def learn(self, env, is_end):
        if not self.smart:
            self._set_ending_env(env)
            self._append_sample(is_end)
            self.dqnAgent.train_model()
            print("White Power first env ", self.env[0])
            print("White Power last env ", self.env[1])
            print("suspects ", self.suspect_nbr)
            print("White Power reward ", self.reward)