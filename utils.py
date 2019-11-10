CHAR_SELECT = "select character"
POS_SELECT = "select position"

# region Powers

GREY_POWER_ACTIVATE = "activate grey power"
GREY_POWER_USE = "grey character power"
# output = 10 from 0 to 9, power before position

BLUE_POWER_ACTIVATE = "activate blue power"
BLUE_POWER_USE = "blue character power room"
# output = 10 from 0 to 9, power before position
BLUE_POWER_USE2 = "blue character power exit"
# output = 10 from 0 to 9 (but just pos adjacent), power before position

WHITE_POWER_ACTIVATE = "activate white power"
WHITE_POWER_USE = "white character power"
# output = 10 from 0 to 9 (but just pos adjacent), power after position

BLACK_POWER_ACTIVATE = "activate black power"

BROWN_POWER_ACTIVATE = "activate brown power"
BROWN_POWER_USE = "brown character power"
# output = 10 from 0 to 9, power before position

PURPLE_POWER_ACTIVATE = "activate purple power"
PURPLE_POWER_USE = "purple character power"
# output = 7 from 0 to 7 select a character color, power before position

RED_POWER_ACTIVATE = "activate red power"

POWER_ACTIVATE = "activate"

# end Region Powers

# region Environment

CARLOTTA_POS = "position_carlotta"
DATA = "data"
ANSWER = "data"
RESET = "Reset"
QUESTION = "question type"
CHARACTERS = "characters"
GAME_STATE = "game state"
COLOR = "color"
SUSPECT = "suspect"
POSITION = "position"
POWER = "power"
FANTOM = "fantom"
SHADOW = "shadow"
BLOCKED = "blocked"
END_PHASE = "end phase"
NUM_TOUR = "num_tour"
# end region Environment

characters = ['pink', 'blue', 'brown', 'red', 'black', 'white', 'purple', 'grey']
passages = [{0, 1}, {0, 4}, {1, 2}, {2, 3}, {3, 7}, {4, 5}, {4, 8}, {4, 5}, {5, 6}, {6, 7}, {7, 9}, {8, 9}]

