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
            self.reward = 100
        elif (self.suspect_nbr[0] / 2) - (self.suspect_nbr[0] - self.suspect_nbr[1]) == 0:
            self.reward = 100
        else:
            mult = (self.suspect_nbr[0] / 2) - (self.suspect_nbr[0] - self.suspect_nbr[1])
            self.reward = -mult * 10