from EnvManager import AInspectorEnvManager
import numpy as np
import utils as u


class GreyEnvManager(AInspectorEnvManager):
    def __init__(self, is_smart=False):
        super().__init__(is_smart)

    def _set_model_name(self):
        return "inspector_grey_power.h5"

    def _validate_answer(self, answers):
        return True

    def _dqn2server_answer(self, answers):
        i = 0
        for val in answers:
            if val == self.answerIdx:
                return i
            i += 1

    def _set_model_scope(self):
        return 43, 10

    def get_info_from_previous_env(self, previous_env):
        self.selected_character = previous_env.selected_character

    def _format_env2dqn(self, env):
        ret = super()._format_env2dqn(env)
        ret[u.characters.index(self.selected_character) * 5] = 1
        return ret
