from EnvManager import AEnvManager
import numpy as np
import utils as u


class CharacterEnvManager(AEnvManager):
    def __init__(self):
        super().__init__()

    def _set_model_name(self):
        return "char_picker.h5"

    def _validate_answer(self, answers):
        return self.env[0][self.answerIdx * 5] == 1

    def _dqn2server_answer(self, answers):
        i = 0
        for val in answers:
            if val[u.COLOR] == u.characters[self.answerIdx]:
                return i
            i = i + 1

    def _set_model_scope(self):
        return 43, 8

    def _format_env2dqn(self, env):
        ret = super()._format_env2dqn(env)
        for val in env[u.GAME_STATE][u.CHARACTERS]:
            # if the character is in the question["data"] array it mean its a playable character thus 1
            ret[u.characters.index(val[u.COLOR]) * 5] = 1 if val in env[u.DATA] else 0
        return ret

    def get_info_from_previous_env(self, previous_env):
        pass
