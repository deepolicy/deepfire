import tensorflow as tf
import numpy as np
import os
import time

from .decode_cmd import action_nn, decode_cmd_fun
from .decode_cmd import action_nlogp_nn, action_entropy_nn
from .encode_obs import lstm_nn, obs_nn, obs_nn2,value_nn
from .params import train_params, data_params
from .utils import get_cmd_list

class Train_nn():
    def __init__(self):
        self.nbatch = train_params.horizon
        self.minibatch = train_params.mini_batch

        self.units_num = {
            "11": 20,
            "12": 1,
            "13": 1,
            "14": 3,
            "15": 16,
            "21": 2,
            "32": 1,
        }

        self.build_networks()

        self.train_method()

        # Create session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.saver = tf.train.Saver(max_to_keep=5)

        self.set_nn_model()
        self.run_train()

    def run_train(self):
        '''
            Train
        '''
        path = "./agent/deepfire/data/"
        while 1:
            while len(os.listdir(path+"memory-json-finish/")) < 1:
                time.sleep(10)

            file_name = os.listdir(path+"memory-json-finish/")
            np.save(path + 'record-train/' + str(self.i_model) + '-' + str(len(file_name)) + '.start.' + str(time.time()) + '.npy' , [])

            self.step_train([path+"memory-json/"+i for i in file_name])

            # delete files.
            for i in file_name:
                os.remove(path+"memory-json-finish/"+i)
                os.remove(path+"memory-json/"+i)

            self.save_model(self.i_model)
            np.save(path + 'record-train/' + str(self.i_model) + '-' + str(len(file_name)) + '.end.' + str(time.time()) + '.npy' , [])
            self.i_model += 1

    def set_nn_model(self):
        model_path = "./agent/deepfire/model/"
        if tf.train.latest_checkpoint(model_path):
            tf.train.Saver().restore(self.sess, tf.train.latest_checkpoint(model_path))

            latest_model_version = tf.train.latest_checkpoint(model_path).split("/")[-1].split("-")[-1]
            self.i_model = int(latest_model_version)+1

            print("")
            print("load model from %s" % tf.train.latest_checkpoint(model_path))
            print("")

        else:
            self.sess.run(tf.global_variables_initializer())
            self.i_model = 0

            # save a random init. model to let all model used in env is same
            self.save_model(self.i_model)
            self.i_model += 1

            print("")
            print("Init. model and save it, done!")
            print("")

        self.lstm_state = self.sess.run(self.lstm_init_state)

    def save_model(self,i_model):
        model_path = "./agent/deepfire/model/"+"model"
        self.saver.save(self.sess,model_path,global_step=i_model)

    def build_networks(self):

        self.airport_ph, self.teams_ph, self.units_ph, self.qb_ph, self.rockets_ph, airport_ly1, teams_ly1, units_ly1, qb_ly1, rockets_ly1,\
        units_id,teams_pt,teams_mask, units_mask, qb_mask, rockets_mask = obs_nn(units_num=self.units_num)

        obs_concat, teams_key, units_key, qb_key = obs_nn2(airport_ly1, teams_ly1, units_ly1, qb_ly1, rockets_ly1,units_id,teams_pt,
                                                           teams_mask, units_mask, qb_mask, rockets_mask)

        act_fc, self.lstm_init_state, self.lstm_final_state = lstm_nn(obs_concat, batch_size=self.minibatch)

        self.value = value_nn(tf.slice(act_fc, [0,0], [-1,64]))

        self.op_act, self.op_nlogp, self.op_ent, self.op_pd, self.param_act, self.param_nlogp, self.param_ent, self.param_pd = action_nn(
            tf.slice(act_fc, [0, 64], [-1, -1]), teams_key, units_key, qb_key, teams_mask, units_mask, qb_mask,
            rockets_mask)

    def train_method(self):
        '''

        return:
        '''

        ent_coef = 0.01
        vf_coef = 0.5

        # CREATE THE PLACEHOLDERS
        self.A = A = self.get_action_ph()

        self.op_act_inc = {}
        for LX in ["11", "12", "14", "15"]:
            self.op_act_inc[LX] = tf.placeholder(tf.float32, [None, None],name="op_act_inc_"+str(LX))

        self.act_inc = self.get_act_inc_ph()

        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])

        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])

        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None]) # value
        self.LR = LR = tf.placeholder(tf.float32, [])

        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = action_nlogp_nn(A, self.op_pd, self.param_pd, self.act_inc, self.op_act_inc)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(action_entropy_nn(self.op_ent, self.param_ent, self.act_inc, self.op_act_inc))

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = self.value
        vpredclipped = OLDVPRED + tf.clip_by_value(self.value - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables()

        # 2. Build our trainer
        # if comm is not None and comm.Get_size() > 1:
        #     self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        # else:
        self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        max_grad_norm = 0.5

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

    def get_action_ph(self):
        op_act = {}
        param_act = {}
        for LX in ["11", "12", "14", "15"]:
            op_act[LX] = tf.placeholder(tf.int32, [None, None],name="op_act_"+str(LX))
            # op_act_inc[LX] = tf.placeholder(tf.float32, [None, None],name="op_act_inc_"+str(LX))

            param_act[LX] = {}
            MT = "areapatrol"
            n_params = 3
            param_act[LX][MT] = tf.placeholder(tf.float32, [None, None, n_params], name="param_act_" + str(LX))

            if LX == "15":
                MT = "target_wh_hunt"
                param_act[LX][MT] = tf.placeholder(tf.int32, [None, None], name="param_act_" + str(LX)+"_"+str(MT))

                for MT in ["target0_hunt", "target1_hunt", "target2_hunt"]:
                    n_params = 3
                    param_act[LX][MT] = tf.placeholder(tf.float32, [None, None, n_params], name="param_act_" + str(LX)+"_"+str(MT))

                MT = "area_wh_hunt"
                param_act[LX][MT] = tf.placeholder(tf.int32, [None, None], name="param_act_" + str(LX)+"_"+str(MT))

                for MT in ["area0_hunt", "area1_hunt"]:
                    n_params = 3+3
                    param_act[LX][MT] = tf.placeholder(tf.float32, [None, None, n_params], name="param_act_" + str(LX)+"_"+str(MT))

        return [op_act, param_act]

    def get_act_inc_ph(self):
        act_inc_ph = {}  # action include place holder.

        for LX in ["11", "12", "14", "15"]:
            act_inc_ph[LX] = {}
            MT = "areapatrol"
            act_inc_ph[LX][MT] = tf.placeholder(tf.float32, [None, None],name="act_inc_ph_"+str(LX)+"_"+str(MT))

            if LX == "15":
                MT = "target_wh_hunt"
                act_inc_ph[LX][MT] = tf.placeholder(tf.float32, [None, None],
                                                    name="act_inc_ph_" + str(LX) + "_" + str(MT))
                for MT in ["target0_hunt", "target1_hunt", "target2_hunt"]:
                    act_inc_ph[LX][MT] = tf.placeholder(tf.float32, [None, None],
                                                        name="act_inc_ph_" + str(LX) + "_" + str(MT))

                MT = "area_wh_hunt"
                act_inc_ph[LX][MT] = tf.placeholder(tf.float32, [None, None],
                                                    name="act_inc_ph_" + str(LX) + "_" + str(MT))
                for MT in ["area0_hunt", "area1_hunt"]:
                    act_inc_ph[LX][MT] = tf.placeholder(tf.float32, [None, None],
                                                        name="act_inc_ph_" + str(LX) + "_" + str(MT))
        return act_inc_ph

    def group_memory(self, npy_files):
        print("")
        print("npy_files:", npy_files)
        print("")

        mb_memory = [np.load(npy_file, allow_pickle=True).tolist() for npy_file in npy_files]

        nbatch = self.nbatch*len(npy_files)

        print("")
        print("nbatch:", nbatch)
        print("")

        mb_memory_group = {}
        for k in mb_memory[0]:
            tmp = []
            for item in mb_memory:
                if type(item[k]) == type(np.array([0])):
                    tmp += item[k].tolist()
                else:
                    tmp += item[k]

            mb_memory_group[k] = tmp

        return mb_memory_group, nbatch

    def step_train(self,npy_files):
        # mb_memory = np.load(npy_file,allow_pickle=True).tolist()
        #
        # print("")
        # print(np.array(mb_memory["mb_returns"]).shape)
        # print("")

        mb_memory, nbatch = self.group_memory(npy_files)
        assert nbatch == len(mb_memory["mb_obs"])

        # nbatch = self.nbatch
        perbatch = self.minibatch

        obs = mb_memory["mb_obs"]
        returns = mb_memory["mb_returns"]
        masks = mb_memory["mb_dones"]
        actions = mb_memory["mb_actions"]
        values = mb_memory["mb_values"]
        neglogpacs = mb_memory["mb_neglogpacs"]
        states = mb_memory["mb_states"]
        act_include = mb_memory["mb_act_include"]
        op_act_inc = mb_memory["mb_op_act_inc"]

        lrnow = 1e-5
        cliprangenow = 0.2

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        # recurrent version
        # envsperbatch = nenvs // nminibatches
        # envinds = np.arange(nenvs)
        inds = np.arange(nbatch)

        # flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        for _ in range(1):
            np.random.shuffle(inds)
            for start in range(0, nbatch, perbatch):
                end = start + perbatch
                mbinds = inds[start:end]
                # mbflatinds = flatinds[mbenvinds].ravel()

                mini_obs = []
                mini_returns = []
                mini_masks = []
                mini_actions = []
                mini_values = []
                mini_neglogpacs = []
                mini_states = []
                mini_act_include = []
                mini_op_act_inc = []
                for i in mbinds:
                    mini_obs.append(obs[i])
                    mini_returns.append(returns[i])
                    mini_masks.append(masks[i])
                    mini_actions.append(actions[i])
                    mini_values.append(values[i])
                    mini_neglogpacs.append(neglogpacs[i])
                    mini_states.append(states[i])
                    mini_act_include.append(act_include[i])
                    mini_op_act_inc.append(op_act_inc[i])

                # padding for dimension alignment when use batch training.
                # mini_actions = self.actions_split(mini_obs, mini_actions)
                # mini_obs, mini_actions = self.obs_padding(mini_obs, mini_actions)
                # mini_actions = self.actions_join(mini_actions)

                # padding for dimension alignment when use batch training.
                mini_obs, mini_actions, mini_act_include, mini_op_act_inc = self.obs_padding(mini_obs, mini_actions, mini_act_include, mini_op_act_inc)

                self.train(lrnow, cliprangenow, mini_obs, mini_returns, mini_masks, mini_actions, mini_values, mini_neglogpacs, mini_act_include, mini_op_act_inc, mini_states)
                # slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                # mbstates = states[mbinds]
                # mblossvals.append(self.train(lrnow, cliprangenow, *slices, mbstates))

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, act_inc, op_act_inc, states=None):

        '''
            collect act_inc.
        '''
        act_inc2 = {}
        for LX in ["11", "12", "14", "15"]:
            act_inc2[LX] = {}
            MT = "areapatrol"
            act_inc2[LX][MT] = [i[LX][MT][0] for i in act_inc]

            if LX == "15":
                MT = "target_wh_hunt"
                act_inc2[LX][MT] = [i[LX][MT][0] for i in act_inc]

                for MT in ["target0_hunt", "target1_hunt", "target2_hunt"]:
                    act_inc2[LX][MT] = [i[LX][MT][0] for i in act_inc]

                MT = "area_wh_hunt"
                act_inc2[LX][MT] = [i[LX][MT][0] for i in act_inc]

                for MT in ["area0_hunt", "area1_hunt"]:
                    act_inc2[LX][MT] = [i[LX][MT][0] for i in act_inc]
        act_inc = act_inc2

        '''
            collect actions.
        '''
        op_act_inc2 = {}
        actions2 = [{},{}]
        for LX in ["11", "12", "14", "15"]:
            # ?????? [0]
            actions2[0][LX] = [i[0][LX][0] for i in actions]
            op_act_inc2[LX] = [i[LX][0] for i in op_act_inc]

            actions2[1][LX] = {}
            MT = "areapatrol"
            actions2[1][LX][MT] = [i[1][LX][MT][0] for i in actions]

            if LX == "15":
                MT = "target_wh_hunt"
                '''
                    ????
                '''
                actions2[1][LX][MT] = [i[1][LX][MT][0] for i in actions]

                for MT in ["target0_hunt", "target1_hunt", "target2_hunt"]:
                    actions2[1][LX][MT] = [i[1][LX][MT][0] for i in actions]

                MT = "area_wh_hunt"
                actions2[1][LX][MT] = [i[1][LX][MT][0] for i in actions]
                for MT in ["area0_hunt", "area1_hunt"]:
                    actions2[1][LX][MT] = [i[1][LX][MT][0] for i in actions]
        actions = actions2
        op_act_inc = op_act_inc2

        '''
            batch group obs.
        '''
        qb_LX = [11, 12, 15, 21, 18, 28, 41, 42, 31, 32]

        obs_encode = {}
        obs_encode["airport"] = [item["airport"] for item in obs]

        obs_encode["teams"] = {}
        obs_encode["units"] = {}
        for LX in "11, 12, 13, 14, 15, 21, 32".split(", "):
            obs_encode["teams"][LX] = [item["teams"][LX] for item in obs]
            obs_encode["units"][LX] = [item["units"][LX] for item in obs]

        obs_encode["qb"] = {}
        for LX in [str(LX) for LX in qb_LX]:
            obs_encode["qb"][LX] = [item["qb"][LX] for item in obs]
        obs_encode["rockets"] = [item["rockets"] for item in obs]

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = np.array(returns) - np.array(values)

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.airport_ph: np.array(obs_encode["airport"]),
            # self.qb_ph: np.array(obs_encode["qb"]),
            self.rockets_ph: np.array(obs_encode["rockets"]),

            self.lstm_init_state[0]: np.concatenate([item[0] for item in states],axis=0),
            self.lstm_init_state[1]: np.concatenate([item[1] for item in states],axis=0),

            self.ADV : np.array(advs),
            self.R : np.array(returns),
            self.LR : np.array(lr),
            self.CLIPRANGE : np.array(cliprange),
            self.OLDVPRED : np.array(values),
            self.OLDNEGLOGPAC: np.array(neglogpacs),
        }
        '''
            feed teams and units.
        '''
        td_map.update( {self.teams_ph[LX]: obs_encode["teams"][LX] for LX in "11, 12, 13, 14, 15, 21, 32".split(", ")} )
        td_map.update( {self.units_ph[LX]: obs_encode["units"][LX] for LX in "11, 12, 13, 14, 15, 21, 32".split(", ")} )
        td_map.update( {self.qb_ph[LX]: obs_encode["qb"][LX] for LX in obs_encode["qb"]} )

        '''
            feed act_inc.
        '''
        for LX in ["11", "12", "14", "15"]:
            MT = "areapatrol"
            '''
                use squeeze convert from (batch_size, team_size, 1) to (batch_size, team_size)
            '''

            td_map.update( {self.act_inc[LX][MT]: np.squeeze(np.array(act_inc[LX][MT]),axis=2)} )

            if LX == "15":
                MT = "target_wh_hunt"

                td_map.update({self.act_inc[LX][MT]: np.squeeze(np.array(act_inc[LX][MT]), axis=2)})

                for MT in ["target0_hunt", "target1_hunt", "target2_hunt"]:
                    td_map.update({self.act_inc[LX][MT]: np.squeeze(np.array(act_inc[LX][MT]), axis=2)})


                MT = "area_wh_hunt"
                td_map.update({self.act_inc[LX][MT]: np.squeeze(np.array(act_inc[LX][MT]), axis=2)})

                for MT in ["area0_hunt", "area1_hunt"]:
                    td_map.update({self.act_inc[LX][MT]: np.squeeze(np.array(act_inc[LX][MT]), axis=2)})



        '''
            feed action.
        '''
        for LX in ["11", "12", "14", "15"]:
            td_map.update({self.A[0][LX]: actions[0][LX]})
            td_map.update({self.op_act_inc[LX]: op_act_inc[LX]})

            MT = "areapatrol"
            td_map.update( {self.A[1][LX][MT]: actions[1][LX][MT] } )

            if LX == "15":
                MT = "target_wh_hunt"
                td_map.update({self.A[1][LX][MT]: actions[1][LX][MT]})

                for MT in ["target0_hunt", "target1_hunt", "target2_hunt"]:
                    td_map.update({self.A[1][LX][MT]: actions[1][LX][MT]})

                MT = "area_wh_hunt"
                td_map.update({self.A[1][LX][MT]: actions[1][LX][MT]})

                for MT in ["area0_hunt", "area1_hunt"]:
                    td_map.update({self.A[1][LX][MT]: actions[1][LX][MT]})

        print("- train -"*5)

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]

    def feed_act(self, actions, td_map):
        k = "maintype"
        td_map[self.A[k]] = np.array([item[k] for item in actions])

        # available
        aval_key = "self_id, cov_id, target_id".split(
            ", ")
        for k in aval_key:
            td_map[self.A[k]] = np.array([item[k][0] for item in actions])

        # continuous
        for k in ["point_list", "fly_num", "point_x", "point_y", "point_z", "direction", "length", "width", "speed",
                  "patrol_time", "range", "area_direct", "area_len", "area_wid", "offset"]:
            for MT in self.A[k]:
                td_map[self.A[k][MT]] = np.array([item[k][MT][0] for item in actions])

        # discrete
        for k in ["fly_type", "on_off", "radar_state", "mode", "flag", "type", "area_type"]:
            for MT in self.A[k]:
                td_map[self.A[k][MT]] = np.array([item[k][MT] for item in actions])

        return td_map

    def obs_padding(self, obs_list, actions_list, act_inc_list, op_act_inc_list):

        qb_LX = [11, 12, 15, 21, 18, 28, 41, 42, 31, 32]

        max_len = {k:max([len(item[k]) for item in obs_list]) for k in ["rockets"]}

        max_len2 = {k:{LX:max([len(item[k][LX]) for item in obs_list]) for LX in "11, 12, 13, 14, 15, 21, 32".split(", ")} for k in ["teams", "units"]}

        max_len_qb = {k:{LX:max([len(item[k][LX]) for item in obs_list]) for LX in [str(LX) for LX in qb_LX]} for k in ["qb"]}

        for i in range(len(obs_list)):
            for k in ["rockets"]:
                while len(obs_list[i][k])<max_len[k]:
                    obs_list[i][k].append(obs_list[i][k][-1].copy())

            for k in ["qb"]:
                for LX in [str(LX) for LX in qb_LX]:
                    while len(obs_list[i][k][LX]) < max_len_qb[k][LX]:
                        obs_list[i][k][LX].append(obs_list[i][k][LX][-1].copy())

            for k in ["units"]:
                for LX in "11, 12, 13, 14, 15, 21, 32".split(", "):
                    while len(obs_list[i][k][LX]) < max_len2[k][LX]:
                        obs_list[i][k][LX].append(obs_list[i][k][LX][-1].copy())

            for k in ["teams"]:
                for LX in "13, 21, 32".split(", "):
                    while len(obs_list[i][k][LX]) < max_len2[k][LX]:
                        obs_list[i][k][LX].append(obs_list[i][k][LX][-1].copy())

                for LX in "11, 12, 14, 15".split(", "):
                    while len(obs_list[i][k][LX]) < max_len2[k][LX]:
                        obs_list[i][k][LX].append(obs_list[i][k][LX][-1].copy())

                        MT = 'areapatrol'
                        n_params = 3
                        '''
                            a not elegant operation for append of numpy in dict.
                        '''
                        assert actions_list[i][0][LX].shape[0] == 1
                        assert actions_list[i][1][LX][MT].shape[0] == 1
                        assert act_inc_list[i][LX][MT].shape[0] == 1
                        actions_list[i][0][LX] = np.concatenate([actions_list[i][0][LX], [[0]]], axis=1)
                        op_act_inc_list[i][LX] = np.concatenate([op_act_inc_list[i][LX], [[0]]], axis=1)
                        if 0:
                            actions_list[i][1][LX][MT] = np.concatenate([actions_list[i][1][LX][MT], [[[0]]]], axis=1)
                        else:
                            actions_list[i][1][LX][MT] = np.concatenate([actions_list[i][1][LX][MT], [[[0]*n_params]]], axis=1)
                        act_inc_list[i][LX][MT] = np.concatenate([act_inc_list[i][LX][MT], [[[0]]]], axis=1)

                        if LX == "15":
                            MT = "target_wh_hunt"
                            actions_list[i][1][LX][MT] = np.concatenate([actions_list[i][1][LX][MT], [[0]]], axis=1)
                            act_inc_list[i][LX][MT] = np.concatenate([act_inc_list[i][LX][MT], [[[0]]]], axis=1)

                            for MT in ["target0_hunt", "target1_hunt", "target2_hunt"]:
                                n_params = 3
                                actions_list[i][1][LX][MT] = np.concatenate([actions_list[i][1][LX][MT], [[[0] * n_params]]],
                                                                            axis=1)
                                act_inc_list[i][LX][MT] = np.concatenate([act_inc_list[i][LX][MT], [[[0]]]], axis=1)

                            MT = "area_wh_hunt"
                            actions_list[i][1][LX][MT] = np.concatenate([actions_list[i][1][LX][MT], [[0]]], axis=1)
                            act_inc_list[i][LX][MT] = np.concatenate([act_inc_list[i][LX][MT], [[[0]]]], axis=1)

                            for MT in ["area0_hunt", "area1_hunt"]:
                                n_params = 3+3
                                actions_list[i][1][LX][MT] = np.concatenate([actions_list[i][1][LX][MT], [[[0] * n_params]]],
                                                                            axis=1)
                                act_inc_list[i][LX][MT] = np.concatenate([act_inc_list[i][LX][MT], [[[0]]]], axis=1)
        return obs_list, actions_list, act_inc_list, op_act_inc_list

if __name__ == '__main__':
    print("")
