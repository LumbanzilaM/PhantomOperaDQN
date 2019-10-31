from EnvManager import AEnvManager
import numpy as np
import utils as u


class PositionEnvManager(AEnvManager):
    def __init__(self):
        super().__init__()
        self.save_pos = [0] * 10

    def _set_model_name(self):
        return "pos_picker.h5"

    def _validate_answer(self, answers):
        return self.answerIdx in answers

    def _dqn2server_answer(self, answers):
        i = 0
        for val in answers:
            if val == self.answerIdx:
                return i
            i += 1

    def _set_model_scope(self):
        return 43, 10

    def _format_env2dqn(self, env):
        ret = super()._format_env2dqn(env)
        ret[u.characters.index(self.selected_character) * 5] = 1
        return ret

    def get_info_from_previous_env(self, previous_env):
        self.selected_character = previous_env.selected_character
