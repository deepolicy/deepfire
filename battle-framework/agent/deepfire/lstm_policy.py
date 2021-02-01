import tensorflow as tf
import numpy as np
import os

from .decode_cmd import action_nn, decode_cmd_fun
from .get_reward import GetReward
from .encode_obs import lstm_nn, obs_nn, obs_nn2,value_nn
from .params import train_params, data_params, run_params
from .utils import repair_unit_TMID, print_2  # repair when TMID of unit is 0.
from .rms_norm import _rms_norm


def get_reward_id_offset():
    '''
        get reward_id_offset
    '''

    def sort(file_name):
        def get_file_index(file):
            return int(file.split("-")[1].split(".")[0])

        L = [[get_file_index(file), file] for file in file_name]
        L.sort(key=lambda x: x[0])

        file_name = [i[1] for i in L]

        return file_name

    import os
    path = "./agent/deepfire/data/reward/"
    file_name = os.listdir(path)
    # print(file_name)
    if len(file_name) == 0:
        reward_id_offset = 0
    else:
        file_name = sort(file_name)
        reward_id_offset = int(file_name[-1].split("-")[1].split(".")[0]) - 2
    return reward_id_offset

def get_sum_nlogp_by_inc(results, act_include, maintype):
    '''
        nlogp_disc_cont = get_sum_nlogp_by_inc(results, act_include)
    '''
    MT = maintype
    sum_list = []
    for item in ["neglogp_discrete", "neglogp_continuous"]:
        for k in results[item]:
            if k in ["maintype"]:
                sum_list.append(results[item][k])
            else:
                try:

                    if MT in act_include[k] and act_include[k][MT]:
                        sum_list.append(results[item][k][MT])
                except Exception as e:
                    print(e)
                    print(k)
                    assert 0
    return np.sum(sum_list)

class Red_agent_deepfire():
    def __init__(self):


        self.rms = {}
        self.i_rms = 0

        self.units_num = {
            "11": 20,
            "12": 1,
            "13": 1,
            "14": 3,
            "15": 16,
            "21": 2,
            "32": 1,
        }

        self.agent_step = 0

        if not train_params.submit == 1:
            self.reward_ep_list = []
            self.reward_id_offset = get_reward_id_offset()
            self.get_reward = GetReward()
            self.mb_memory = self.init_memory()

        self.build_networks()

        self.present_model_version = ""

        # Create session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.set_nn_model()

    def set_nn_model(self):

        model_path = data_params.data_dir + "./agent/deepfire/model/"

        if tf.train.latest_checkpoint(model_path):
            self.load_model()
        else:
            print_2("None model exists!")
            assert 0

        '''
            ID set in this epsilon for units by LX
        '''
        self.units_id = {k: [] for k in "11, 12, 13, 14, 15, 21, 32".split(", ")}

        '''
            init next_act_delay
        '''
        self.next_act_delay = {}

        if not train_params.submit == 1:
            self.get_reward = GetReward()

        self.lstm_state = self.sess.run(self.lstm_init_state)

    def load_model(self):

        model_path = data_params.data_dir + "./agent/deepfire/model/"
        checkpoint = tf.train.latest_checkpoint(model_path)

        if checkpoint:
            latest_model_version = checkpoint.split("/")[-1].split("-")[-1]
            if not latest_model_version == self.present_model_version:
                self.present_model_version = latest_model_version
                tf.train.Saver().restore(self.sess, checkpoint)
                
                print("")
                print("load model from %s" % checkpoint)
                print("")

    def build_networks(self):

        self.airport_ph, self.teams_ph, self.units_ph, self.qb_ph, self.rockets_ph, airport_ly1, teams_ly1, units_ly1, qb_ly1, rockets_ly1,\
        units_id,teams_pt,teams_mask, units_mask, qb_mask, rockets_mask = obs_nn(units_num=self.units_num)

        obs_concat, teams_key, units_key, qb_key = obs_nn2(airport_ly1, teams_ly1, units_ly1, qb_ly1, rockets_ly1,units_id,teams_pt,
                                                           teams_mask, units_mask, qb_mask, rockets_mask)

        act_fc, self.lstm_init_state, self.lstm_final_state = lstm_nn(obs_concat)

        # 100-164
        self.value = value_nn(tf.slice(act_fc, [0,0], [-1,64]))

        self.op_act, self.op_nlogp, self.op_ent, op_pd, self.param_act, self.param_nlogp, self.param_ent, param_pd = action_nn(tf.slice(act_fc, [0,64], [-1,-1]), teams_key, units_key, qb_key,teams_mask, units_mask, qb_mask, rockets_mask)

    def save_reward(self, i_episode):
        np.save("./agent/deepfire/data/reward/reward_ep_list-"+str(i_episode), self.reward_ep_list)
        self.reward_ep_list = []

    def step(self, obs_red_const, observation, done, i_epsilon, sim_time):

        import copy
        obs_json = copy.deepcopy(obs_red_const)
        self.agent_step += 1

        if not train_params.submit == 1:
            # save reward per epsilon.
            if done and len(self.reward_ep_list) > 0:
                # np.save("./agent/deepfire/data/reward/reward_ep_list-"+str(i_epsilon+self.reward_id_offset), self.reward_ep_list)
                # self.reward_ep_list = []

                # save final observation
                import json
                with open("./agent/deepfire/data/final-observation/observation-"+ str(i_epsilon+self.reward_id_offset) +".json", "w") as f:
                    f.write(json.dumps(observation))

            reward = self.get_reward.step(observation["red"],observation["blue"])

            self.reward_ep_list.append(reward)

            if self.agent_step%50==0:
                self.load_model()

        obs_json = repair_unit_TMID(obs_json)
        obs_encode, airport_id, cov_id_list, self_id_list, target_id_list, team_id = self.encode_obs_fun(obs_json)

        fetches = {
            "act": [self.op_act, self.op_nlogp, self.op_ent, self.param_act, self.param_nlogp, self.param_ent],
            # "act_nodes": self.act_nodes,
            # "neglogp_discrete": self.neglogp_discrete,
            # "neglogp_continuous": self.neglogp_continuous,
            # "neglogp_all_list": self.neglogp_all_list,
            "value": self.value,
            "lstm_final_state": self.lstm_final_state,
        }

        feed_dict = {
            self.airport_ph: [obs_encode["airport"]],
            # self.teams_ph: [obs_encode["teams"]],
            # self.units_ph: [obs_encode["units"]],
            # self.qb_ph: [obs_encode["qb"]],
            self.rockets_ph: [obs_encode["rockets"]],
            self.lstm_init_state: self.lstm_state,

        }

        feed_dict.update( {self.teams_ph[LX]: [obs_encode["teams"][LX]] for LX in "11, 12, 13, 14, 15, 21, 32".split(", ")} )
        feed_dict.update( {self.units_ph[LX]: [obs_encode["units"][LX]] for LX in "11, 12, 13, 14, 15, 21, 32".split(", ")} )
        feed_dict.update( {self.qb_ph[LX]: [obs_encode["qb"][LX]] for LX in obs_encode["qb"]} )

        results = self.sess.run(fetches,feed_dict)

        cmd, nlogp, act_inc, op_act_inc, self.next_act_delay = decode_cmd_fun(results["act"], airport_id, cov_id_list, self_id_list, target_id_list, obs_json, team_id, obs_red_const, sim_time, self.next_act_delay)

        # '''
        #     sum the neglogp of discrete and continuous by act_include.
        # '''
        # nlogp_disc_cont = get_sum_nlogp_by_inc(results, act_include, cmd[0]["maintype"])
        #
        # '''
        #     get neglogp of self_id and etc.
        # '''
        # nlogp_var_mask = 0
        # for i, k in zip([0, 1, 2], ["self_id", "cov_id", "target_id"]):
        #     if act_include[k]:
        #         nlogp_var_mask += results["neglogp_all_list"][i][0][results["act_nodes"][k][0]]
        # nlogp_var_mask = [nlogp_var_mask]
        #
        # '''
        #     get one_hot of self_id and etc.
        # '''
        # one_hot = [np.zeros(len(k)+1) for k in [self_id_list, cov_id_list, target_id_list]]
        # one_hot[0][results["act_nodes"]["self_id"][0]] = 1.
        # one_hot[1][results["act_nodes"]["cov_id"][0]] = 1.
        # one_hot[2][results["act_nodes"]["target_id"][0]] = 1.
        # results["act_nodes"]["self_id"] = [one_hot[0]]
        # results["act_nodes"]["cov_id"] = [one_hot[1]]
        # results["act_nodes"]["target_id"] = [one_hot[2]]

        if not train_params.submit == 1:
            self.add_memory(obs_encode, [results["act"][0], results["act"][3]], [nlogp], act_inc, op_act_inc, reward, results["value"], done, self.lstm_state )

        self.lstm_state = results["lstm_final_state"]

        return cmd

    def encode_obs_fun(self,obs):

        obs_in = {}

        unit_team_id = [{k:[] for k in "11, 12, 13, 14, 15, 21, 32".split(", ")},{k:[] for k in "11, 12, 13, 14, 15, 21, 32".split(", ")}]

        assert len(obs["airports"]) == 1
        obs_in["airport"] = [obs["airports"][0][k] for k in
                             "CA, NM, KY, AIR, AWCS, JAM, UAV, BOM, X, Y, Z, WH, DA".split(", ")]  # without embedding and mask data.
        airport_id = obs["airports"][0]["ID"]

        team_units = {}

        '''
            units
        '''
        obs_in["units"] = {k: [] for k in "11, 12, 13, 14, 15, 21, 32".split(", ")}
        for unit in obs["units"]:
            LX = str(unit["LX"])
            if unit["type"] == "COMMAND":
                print("red has not  COMMAND")
                assert 0

            # deal with WP
            # WP types is 4: 170, 360, 519 and 0, for embedding
            assert len(unit["WP"]) <= 1
            WP = [0, 0]
            if len(unit["WP"]) == 1:
                # print(unit["WP"])
                # e.g. {'519': 36}
                for k in unit["WP"]:
                    if not int(k) in [170, 360, 519]:
                        print(unit["WP"])
                        assert int(k) in [170, 360, 519]
                    WP = [[360, 170, 519].index(int(k)), unit["WP"][k]]  # 2nd element is num of weapon


            # deal with ID
            if not unit["ID"] in self.units_id[LX]:
                self.units_id[LX].append(unit["ID"])
            ID = self.units_id[LX].index(unit["ID"])
            tmp = [0]*self.units_num[LX]  # or the max length by LX

            try:
                tmp[ID] = 1
            except Exception as e:
                print(e)

            ID = tmp

            # deal with its teams.
            if not unit["TMID"] in team_units:
                team_units[unit["TMID"]] = [0]*self.units_num[LX]
            team_units[unit["TMID"]][ self.units_id[LX].index(unit["ID"]) ] = 1

            obs_in["units"][LX].append(
                [unit[k] for k in "X, Y, Z, HX, SP, TM, Locked, WH, DA, Hang, Fuel".split(", ")] + WP[1:] + \
                WP[:1] + [unit['ST']] + ID + [1] )  # rms [:12], embedd [12:14], mask [14:]

            unit_team_id[0][LX].append(unit["ID"])

            # ST
            obs_in["units"][LX][-1][13] = [0, 13, 1, 14, 2, 15, 3, 16, 4, 17, 5, 31, 6, 32, 7, 41, 8, 60, 9, 61, 10, 91, 11,
                                       90, 12, 92].index(obs_in["units"][LX][-1][13])

        for k in obs_in["units"]:
            obs_in["units"][k].append(
                [0 for k in range(14)] + [0]*self.units_num[k] + [0])
            unit_team_id[0][k].append(0)

        '''
            teams
        '''
        obs_in["teams"] = {k: [] for k in "11, 12, 13, 14, 15, 21, 32".split(", ")}
        for team in obs["teams"]:

            if len(team["PT"])==0:
                print("")
                print("remove team when its PT is empty:",team)
                print("")
                continue

            LX = str(team["LX"])

            # deal with PT
            # Repair the PT&TMID problem.
            if not team["TMID"] in team_units:
                '''
                    Filter (remove) the team whose units cannot found in units of observation by ID.
                '''
                continue
            PT = team_units[team["TMID"]]

            obs_in["teams"][LX].append([team[k] for k in "Num, ST".split(", ")] + PT + [1])  # rms [:1], embedd [1:2], mask [2:]

            # cov_id_list.append(team["TMID"])
            unit_team_id[1][LX].append(team["TMID"])

            # ST
            obs_in["teams"][LX][-1][1] = [0, 33, 1, 34, 2, 35, 3, 40, 4, 41, 5, 42, 6, 43, 7, 44, 8, 51, 9, 52, 10, 53, 11,
                                      54, 12, 55, 13, 60, 14, 61, 15, 62, 16, 63, 17, 64, 18, 71, 30, 72, 31, 73, 32,
                                      81].index(obs_in["teams"][LX][-1][1])

        for k in obs_in["teams"]:
            obs_in["teams"][k].append([0 for k in range(2)] + [0]*self.units_num[k] + [
                0])  # its id will not be choosed for its probability is zero after mask operation
            unit_team_id[1][k].append(0)


        '''
            qb
        '''
        obs_in["qb"] = {str(k): [] for k in [11, 12, 15, 21, 18, 28, 41, 42, 31, 32]}  # 32 has not show, but we contains it here to avoid error.

        target_id_list = []

        for qb in obs["qb"]:
            LX = str(qb["LX"])

            if int(LX) in [19, 29]:
                assert qb["JB"] == 3

            # Filter items from qb when JB=3.
            if qb["JB"] == 3:
                continue

            assert LX in obs_in["qb"]

            # obs_in["qb"][LX].append([qb[k] for k in "X, Y, Z, HX, SP, LX, WH, DA, TM".split(", ")] + [1])
            obs_in["qb"][LX].append([qb[k] for k in "X, Y, Z, HX, SP, WH, DA, TM".split(", ")] + [1])  # rms [:8], embedd None, mask [8:]
            target_id_list.append(qb["ID"])

        for k in obs_in["qb"]:
            obs_in["qb"][k].append([0 for k in range(8)] + [0])
            target_id_list.append(0)  # repair and add here in 08-03 10:21

        '''
            rockets
        '''
        obs_in["rockets"] = []
        for rocket in obs["rockets"]:
            obs_in["rockets"].append([rocket[k] for k in "X, Y, Z, FY, HG, HX, WH, TM".split(", ")] + [1])  # rms [:8], embedd None, mask [8:]

        obs_in["rockets"].append([0 for k in range(8)] + [0])

        # collect self_id_list.
        self_id_list = []
        # for LX in unit_team_id[0]:
        for LX in "11, 12, 13, 14, 15, 21, 32".split(", "):
            self_id_list+=unit_team_id[0][LX]
        # for LX in unit_team_id[1]:
        for LX in "11, 12, 13, 14, 15, 21, 32".split(", "):
            self_id_list+=unit_team_id[1][LX]

        cov_id_list = []
        type4cov = ["12", "13", "15"]
        for LX in type4cov:
            cov_id_list+=unit_team_id[1][LX]

        '''
            rms
        '''
        rms, i_rms = self.rms, self.i_rms
        rms_file = 'agent/deepfire/red_rms_data.npy'
        if rms == {}:
            rms = _rms_norm.init(obs_in)
            if os.path.exists(rms_file):
                rms = _rms_norm.load(rms, rms_file)

        if i_rms > 1000:
            _rms_norm.save(rms, rms_file)
            i_rms = 0
        else:
            i_rms += 1

        obs_in = _rms_norm.step(obs_in, rms)

        self.rms, self.i_rms = rms, i_rms



        return obs_in, airport_id, cov_id_list, self_id_list, target_id_list, unit_team_id[1]

    def add_memory(self, obs, action, neglogpac, act_include, op_act_inc, reward, value, done, lstm_state ):
        horizon = train_params.horizon
        lam = 0.95
        gamma = 0.99

        if len(self.mb_memory["mb_obs"])==horizon:
            '''
                deal
            '''
            last_values = value

            mb_rewards = self.mb_memory["mb_rewards"]
            mb_dones = self.mb_memory["mb_dones"]
            mb_values = self.mb_memory["mb_values"]

            # discount/bootstrap off value fn
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(horizon)):
                if t == horizon - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t+1]
                    nextvalues = mb_values[t+1]
                delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

            mb_returns = mb_advs + mb_values

            # save memory
            import time
            self.mb_memory["mb_returns"] = mb_returns
            self.mb_memory["mb_advs"] = mb_advs

            tmp = str(time.time())+"-ip"+run_params.ip+"-env"+run_params.env
            np.save(data_params.data_dir + "./agent/deepfire/data/memory-json/memory-" + tmp + ".npy", self.mb_memory)
            np.save(data_params.data_dir + "./agent/deepfire/data/memory-json-finish/memory-" + tmp + ".npy", [])  # marks that the npy file last step has written finished.
            np.save(data_params.data_dir + "./agent/deepfire/data/files-train/memory-" + tmp + ".npy", [])

            # empty memory
            self.mb_memory = self.init_memory()

        '''
            collect
        '''
        self.mb_memory["mb_obs"].append(obs)
        self.mb_memory["mb_actions"].append(action)

        self.mb_memory["mb_values"].append(value[0][0]) # value shape(1,1), batch_size=1, output_dim=1
        self.mb_memory["mb_neglogpacs"].append(neglogpac[0])
        self.mb_memory["mb_act_include"].append(act_include)
        self.mb_memory["mb_op_act_inc"].append(op_act_inc)
        self.mb_memory["mb_dones"].append(done)
        assert len(lstm_state)==2
        self.mb_memory["mb_states"].append(lstm_state)
        self.mb_memory["mb_rewards"].append(reward)

    def init_memory(self):
        return {
            k:[] for k in "mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_act_include, mb_op_act_inc, mb_states".split(", ")
        }
