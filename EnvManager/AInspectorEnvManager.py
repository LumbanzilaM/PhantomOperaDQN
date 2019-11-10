from EnvManager import AEnvManager
import numpy as np
import utils as u


class AInspectorEnvManager(AEnvManager):
    def __init__(self, is_smart=False):
        super().__init__(is_smart)

    def _set_model_name(self):
        pass

    def _validate_answer(self, answers):
        pass

    def _dqn2server_answer(self, answers):
        pass

    def _set_model_scope(self):
        pass

    def _format_env2dqn(self, env):
        return super()._format_env2dqn(env)

    def _calculate_reward(self, env):
        data = env[u.DATA]
        if self.suspect_nbr[1] == 1:
            self.reward = 1000
        elif self.suspect_nbr[0] - self.suspect_nbr[1] > 0:
            self.reward = 50 + (50 * (self.suspect_nbr[0] - self.suspect_nbr[1]))
        else:
            self.reward = - 12.5 + (12.5 * (self.suspect_nbr[0] - self.suspect_nbr[1]))
        #print("phantom is {} pos {}".format(self.phantom_color, self.phantom_position))
        #print("last suspect =", self.suspect_nbr[0])
        #print("new suspect =", self.suspect_nbr[1])
        #print("reward =", self.reward)