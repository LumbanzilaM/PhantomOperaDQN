from EnvManager import AInspectorEnvManager
import numpy as np
import utils as u


class WhiteEnvManager(AInspectorEnvManager):
    def __init__(self, is_smart=False):
        super().__init__(is_smart)

    def _set_model_name(self):
        return "inspector_white_power.h5"

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
