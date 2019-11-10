from EnvManager import AEnvManager
import numpy as np
import utils as u


class PowerActivationEnvManager(AEnvManager):
    def __init__(self, is_smart=False):
        super().__init__(is_smart)
        self.isActivated = False
        self.selected_char_moved = False

    def _set_model_name(self):
        return "power_activation_picker.h5"

    def _validate_answer(self, answers):
        return True

    def _dqn2server_answer(self, answers):
        i = 0
        for val in answers:
            if val == self.answerIdx:
                return i
            i += 1

    def _set_model_scope(self):
        return 44, 2

    def _format_env2dqn(self, env):
        ret = super()._format_env2dqn(env)
        ret[self.env_size - 4] = int(self.selected_char_moved)
        print("Selected char =", self.selected_character)
        ret[u.characters.index(self.selected_character) * 5] = 1
        return ret

    def get_info_from_previous_env(self, previous_env):
        self.selected_character = previous_env.selected_character

    def learn(self, env, is_end):
        if self.isActivated and not self.smart:
            self._set_ending_env(env)
            self._append_sample(is_end)
            self.dqnAgent.train_model()
            print("Power Activate first env ", self.env[0])
            print("Power Activate last env ", self.env[1])
            print("suspects ", self.suspect_nbr)
            print("Power Activate reward ", self.reward)
            self.isActivated = False

    def _get_smart_action(self, env):
        if u.RED_POWER_ACTIVATE == env[u.QUESTION] or u.PURPLE_POWER_ACTIVATE == env[u.QUESTION]:
            self.answerIdx = 1
            return 1
        self._set_starting_env(env)
        values = np.array(self.dqnAgent.get_smart_action(self.env[0]))
        for i in range(self.output_size):
            self.answerIdx = np.argmax(values)
            if not self._validate_answer(env[u.ANSWER]):
                print("WRONG SMART ANSWER ->", values)
                values[self.answerIdx] = -10000
        print("Learning state ->", values)
        return self._dqn2server_answer(env[u.ANSWER])

    def get_action(self, env):
        self.isActivated = True
        return super().get_action(env)

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
