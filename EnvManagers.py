from abc import ABC, abstractmethod
import collections

characters = ['pink', 'blue', 'brown', 'red', 'black', 'white', 'purple', 'grey']


class AEnvManager(ABC):
    def __init__(self):
        self.isFirstAction = True
        self.env_size = 40
        self.reward = 0
        self.carlotta_pos = [-1, -1]
        self.suspect_nbr = [8, 0]
        self.env = [[], []]

    @abstractmethod
    def process_env(self, env):
        pass

    @abstractmethod
    def set_env_info(self, env):
        if self.carlotta_pos[0] == -1:
            self.carlotta_pos[0] = env["game state"]["position_carlotta"]
            self.carlotta_pos[1] = self.carlotta_pos[0]
        else:
            self.isFirstAction = False
            self.suspect_nbr[0] = self.suspect_nbr[1]
            self.suspect_nbr[1] = 0
            self.carlotta_pos[0] = self.carlotta_pos[1]
            self.carlotta_pos[1] = env["game state"]["position_carlotta"]
            self.env[0] = self.env[1]

    def calculate_reward(self):
        self.reward = 0
        # print("old pos = ", self.carlotta_pos[0])
        # print("new pos =", self.carlotta_pos[1])
        # print("suspects =", self.suspect_nbr)
        # print("environments =", self.env)
        # When the game start the carlotta position is - 1 before the first server message
        # pos[0] >=0 means this is not the first action so we can calculate the reward for the action played
        if self.carlotta_pos[0] >= 0:
            self.reward = ((self.carlotta_pos[1] - self.carlotta_pos[0]) - (3 * (self.suspect_nbr[0] - self.suspect_nbr[1])))
        # print("reward = ", self.reward)

    def reset(self):
        self.carlotta_pos = [-1, -1]
        self.suspect_nbr = [8, 0]
        self.env = [[], []]
        self.isFirstAction = True


class CharacterEnvManager(AEnvManager):
    def __init__(self):
        super().__init__()
        self.phantom_color = -1

    def process_env(self, env):
        ret = [0] * self.env_size
        if env["question type"] == "select character" or env["question type"] == "Reset":
            self.set_env_info(env)
            for val in env["game state"]["characters"]:
                if val in env["data"]:
                    # if the character is in the question["data"] array it mean its a playable character thus 1
                    ret[characters.index(val["color"]) * 5] = 1
                # Set the is suspect value to int 0 = false, 1 = true
                ret[(characters.index(val["color"]) * 5) + 1] = int(val["suspect"])
                self.suspect_nbr[1] += int(val["suspect"]);
                # Set the character position
                ret[(characters.index(val["color"]) * 5) + 2] = val["position"]
                # Set the power value to int 0 = false, 1 = true
                ret[(characters.index(val["color"]) * 5) + 3] = int(val["power"])
            if (self.phantom_color == -1):
                self.phantom_color = env["game state"]["fantom"]
            ret[(characters.index(self.phantom_color) * 5) + 4] = -1
            print("print phantom is  = ", self.phantom_color)
            print("ret  = ", ret)
            self.env[1] = ret
            self.calculate_reward()
        return ret

    def set_env_info(self, env):
        super().set_env_info(env)


class PositionEnvManager(AEnvManager):
    def process_env(self, env):
        ret = [0] * self.env_size
        if env["question type"] == "select character" or env["question type"] == "Reset":
            self.set_env_info(env)
            for val in env["game state"]["characters"]:
                if val in env["data"]:
                    # if the character is in the question["data"] array it mean its a playable character thus 1
                    ret[characters.index(val["color"]) * 4] = 1
                # Set the is suspect value to int 0 = false, 1 = true
                ret[(characters.index(val["color"]) * 4) + 1] = int(val["suspect"])
                self.suspect_nbr[1] += int(val["suspect"])
                # Set the character position
                ret[(characters.index(val["color"]) * 4) + 2] = val["position"]
                # Set the power value to int 0 = false, 1 = true
                ret[(characters.index(val["color"]) * 4) + 3] = int(val["power"])
                ret[(characters.index(env["game state"]["fantom"]) * 5) + 4] = 1
            self.env[1] = ret
            self.calculate_reward()
        return ret

    def set_env_info(self, env):
        super().set_env_info(env)