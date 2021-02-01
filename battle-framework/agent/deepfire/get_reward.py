import numpy as np

'''
v3: consider death of LX31 and DA of LX21.

v2: consider the DA of COMMAND, with new function:
    def get_reward_COMMAND_DA(self,obs_blue):
    get a more smooth reward.

'''

class GetReward():
    '''
        compare units previous and now 就认为被摧毁了。
    '''

    def __init__(self):
        self.red_units = {
            "15": 16,
            "12": 1,
            "14": 3,
            "13": 1,
            "11": 20,
            "21": 2,
            "32": 1,
        }

        self.blue_units = {
            "15": 8,
            "12": 1,
            "11": 12,
            "21": 1,
            "32": 2,
            "41": 2,
        }

        self.blue_COMMAND_DA = {
            # ID: DA
        }

        self.blue_LX21_DA = {
            # ID: DA
        }

    def assert_units_num(self, obs_red, obs_blue):
        '''
            in the first some obs, the units number of red should keep same with the definition.
        '''
        # red
        red_units = self.get_red_units(obs_red)
        for k in self.red_units:
            assert red_units[k] == self.red_units[k]

        # blue
        blue_units = self.get_blue_units(obs_blue)
        for k in self.blue_units:
            assert blue_units[k] == self.blue_units[k]

    def get_red_units(self, obs_red):
        red_units = {k: 0 for k in self.red_units}

        # add units in airports.
        port = obs_red["airports"][0]
        red_units["11"] += port["AIR"]
        red_units["12"] += port["AWCS"]
        red_units["13"] += port["JAM"]
        red_units["14"] += port["UAV"]
        red_units["15"] += port["BOM"]

        assert port["KY"] == sum(port[k] for k in "AIR, AWCS, JAM, UAV, BOM".split(", "))

        # add units in units.
        for unit in obs_red["units"]:
            LX = unit["LX"]
            for k in "11, 12, 13, 14, 15, 21, 32".split(", "):
                if LX == int(k):
                    red_units[k] += 1

        return red_units

    def get_blue_units(self, obs_blue):
        blue_units = {k: 0 for k in self.blue_units}

        # add units in airports.
        port = obs_blue["airports"][0]
        blue_units["11"] += port["AIR"]
        blue_units["12"] += port["AWCS"]
        blue_units["15"] += port["BOM"]

        assert port["KY"] == sum(port[k] for k in "AIR, AWCS, BOM".split(", "))

        # add units in units.
        for unit in obs_blue["units"]:
            LX = unit["LX"]
            for k in "11, 12, 15, 21, 32, 41".split(", "):
                if LX == int(k):
                    blue_units[k] += 1

        return blue_units

    def step(self, obs_red, obs_blue):
        '''
            compare the units number (with units in airports) to get who are destoryed.
        '''
        reward = 0.

        red_units = self.get_red_units(obs_red)

        blue_units = self.get_blue_units(obs_blue)

        # red
        for k in self.red_units:
            death = self.red_units[k] - red_units[k]
            if death > 0:
                print("red:", k, death)
                if k == "11":
                    reward += -0.05 * death * 0.8
                if k == "12":
                    reward += -1.0 * death
                if k == "13":
                    reward += -1.0 * death
                if k == "14":
                    reward += -0.33 * death
                if k == "15":
                    reward += -0.0625 * death * 1.2
                if k == "21":
                    reward += -0.5 * death
                if k == "32":
                    reward += -1.0 * death
        self.red_units = red_units

        # blue death
        for k in self.blue_units:
            death = self.blue_units[k] - blue_units[k]
            if death > 0:
                print("blue:", k, death)
                if k == "11":
                    reward += 0.05 * death
                if k == "12":
                    reward += 1.0 * death
                if k == "15":
                    reward += 0.0625 * death
                if k == "21":
                    reward += 1.5 * death * 0.34  # need 3 missiles for the death, the last one make death.
                if k == "32":
                    reward += 1.0 * death
                if k == "31":
                    reward += 0.5 * death
                if k == "41":
                    reward += 3.0 * death * 0.25  # need 4 missiles for the death, the last one make death.

        reward += self.get_reward_COMMAND_DA(obs_blue)
        reward += self.get_reward_LX21_DA(obs_blue)

        self.blue_units = blue_units

        return reward

    def get_reward_COMMAND_DA(self,obs_blue):

        reward = 0.

        # blue COMMAND DA
        COMMAND_units = [unit for unit in obs_blue["units"] if unit["LX"] == 41]
        for unit in COMMAND_units:
            if not unit["ID"] in self.blue_COMMAND_DA:
                self.blue_COMMAND_DA[unit["ID"]] = unit["DA"]

            reward += (unit["DA"] - self.blue_COMMAND_DA[unit["ID"]]) * 0.01 * 3.0
            self.blue_COMMAND_DA[unit["ID"]] = unit["DA"]

        return reward

    def get_reward_LX21_DA(self,obs_blue):

        reward = 0.

        # blue LX21 DA
        units = [unit for unit in obs_blue["units"] if unit["LX"] == 21]
        for unit in units:
            if not unit["ID"] in self.blue_LX21_DA:
                self.blue_LX21_DA[unit["ID"]] = unit["DA"]

            reward += (unit["DA"] - self.blue_LX21_DA[unit["ID"]]) * 0.01 * 1.5
            self.blue_LX21_DA[unit["ID"]] = unit["DA"]

        return reward