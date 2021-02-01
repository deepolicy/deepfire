import tensorflow as tf
import numpy as np

from env.env_cmd import EnvCmd

from .utils import softmax_argmax, embedding, get_cmd_list, Point, get_xy_obs_by_id, clip_2, scale_to, targethunt_rule, areahunt_rule
from .distributions import CategoricalPd, DiagGaussianPd, CategoricalPd_var_mask

def action_nlogp_nn(act, op_pd, param_pd, act_inc, op_act_inc):

    op_act, param_act = act

    nlogp_list = []
    for LX in ["11", "12", "14", "15"]:

        '''
            a not elegant operation here with reduce_sum
        '''
        '''
            need add act_inc here or not? experiments.
        '''

        '''
            op_pd[LX].neglogp(op_act[LX])
            (4, ?)
        '''
        nlogp_list.append( tf.reduce_sum(op_pd[LX].neglogp(op_act[LX])*op_act_inc[LX],axis=1) )

        MT = "areapatrol"

        '''
            param_pd[LX][MT].neglogp(param_act[LX][MT])
            (4, ?)
        '''
        nlogp_list.append( tf.reduce_sum(param_pd[LX][MT].neglogp(param_act[LX][MT]) * act_inc[LX][MT], axis=1) )

        if LX == "15":
            MT = "target_wh_hunt"
            nlogp_list.append(tf.reduce_sum(param_pd[LX][MT].neglogp(param_act[LX][MT]) * act_inc[LX][MT], axis=1))

            for MT in ["target0_hunt", "target1_hunt", "target2_hunt"]:
                nlogp_list.append(tf.reduce_sum(param_pd[LX][MT].neglogp(param_act[LX][MT]) * act_inc[LX][MT], axis=1))

            MT = "area_wh_hunt"
            nlogp_list.append(tf.reduce_sum(param_pd[LX][MT].neglogp(param_act[LX][MT]) * act_inc[LX][MT], axis=1))

            for MT in ["area0_hunt", "area1_hunt"]:
                nlogp_list.append(tf.reduce_sum(param_pd[LX][MT].neglogp(param_act[LX][MT]) * act_inc[LX][MT], axis=1))

    return tf.add_n(nlogp_list)

def action_entropy_nn(op_ent, param_ent, act_inc, op_act_inc):

    ent_list = []
    for LX in ["11", "12", "14", "15"]:
        ent_list.append(tf.expand_dims(tf.reduce_sum(op_ent[LX]*op_act_inc[LX], axis=1), 1))

        '''
            a not elegant operation here with reduce_sum and expand_dims
        '''

        MT = "areapatrol"
        '''
            param_ent[LX][MT] * act_inc[LX][MT]
            (4, ?)
        '''
        ent_list.append( tf.expand_dims(tf.reduce_sum(param_ent[LX][MT] * act_inc[LX][MT], axis=1),1) )

        if LX == "15":
            MT = "target_wh_hunt"
            ent_list.append(tf.expand_dims(tf.reduce_sum(param_ent[LX][MT] * act_inc[LX][MT], axis=1), 1))

            for MT in ["target0_hunt", "target1_hunt", "target2_hunt"]:
                ent_list.append(tf.expand_dims(tf.reduce_sum(param_ent[LX][MT] * act_inc[LX][MT], axis=1), 1))

            MT = "area_wh_hunt"
            ent_list.append(tf.expand_dims(tf.reduce_sum(param_ent[LX][MT] * act_inc[LX][MT], axis=1), 1))

            for MT in ["area0_hunt", "area1_hunt"]:
                ent_list.append(tf.expand_dims(tf.reduce_sum(param_ent[LX][MT] * act_inc[LX][MT], axis=1), 1))
    return tf.add_n(ent_list)

def get_pd_continuous(mean, i_size):
    '''
        action["point_x"], neglogp_continuous["point_x"], ent_continuous["point_x"], pd_continuous["point_x"] = get_pd_continuous(tf.slice(nodes2, [0, i_slice2], [-1, 1]), i_size=1)
    '''
    logstd = tf.Variable(np.zeros([1,1, i_size]), dtype=tf.float32)
    pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=-1)
    diag_gaussian_pd = DiagGaussianPd(pdparam)
    act = diag_gaussian_pd.sample()
    return act, diag_gaussian_pd.neglogp(act), diag_gaussian_pd.entropy(), diag_gaussian_pd

def get_pd_continuous_2(mean, logstd):
    '''
        action["point_x"], neglogp_continuous["point_x"], ent_continuous["point_x"], pd_continuous["point_x"] = get_pd_continuous(tf.slice(nodes2, [0, i_slice2], [-1, 1]), i_size=1)
    '''
    # logstd = tf.Variable(np.zeros([1,1, i_size]), dtype=tf.float32)
    pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=-1)
    diag_gaussian_pd = DiagGaussianPd(pdparam)
    act = diag_gaussian_pd.sample()
    return act, diag_gaussian_pd.neglogp(act), diag_gaussian_pd.entropy(), diag_gaussian_pd

def get_pd_discrete(latent):
    pd = CategoricalPd(latent)
    act = pd.sample()
    return act, pd.neglogp(act), pd.entropy(), pd

def get_type4cmd():
    return tf.constant([
        # 作战飞机
        [1., 1., 1., 1., 1., 0., 0., 0., 1.],
        [1., 1., 1., 1., 1., 0., 0., 0., 1.],
        [1., 1., 1., 1., 1., 0., 0., 0., 1.],
        [1., 1., 1., 1., 1., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0., 1., 0.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1.],
        #
        # # 地防 disable for red side.
        # [0., 0., 0., 0., 0., 0., 0., 0., 1.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 1.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 1.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 1.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 1.],

        # 护卫舰
        # [0., 0., 0., 0., 0., 1., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1., 0., 0., 1.],

        # 预警机
        [0., 1., 0., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0., 0., 0., 1.],

        # 干扰机
        [0., 0., 1., 0., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0., 0., 1.],

        # 无人侦察机
        [0., 0., 0., 1., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 0., 0., 1.],

        # 地面雷达
        [0., 0., 0., 0., 0., 0., 1., 0., 1.]])

def action_nn(nodes, teams_key, units_key, qb_key,teams_mask, units_mask, qb_mask, rockets_mask):
    # 战场地图边界(x正向, x负向, y正向, y负向), 单位: 米
    X_Max = 174500
    X_Min = -170000
    Y_Max = 174500
    Y_Min = -171000

    # target_id_keys = tf.layers.dense( qb_key ,5,activation=tf.nn.relu)

    _concat = {}
    nodes = tf.nn.relu(nodes)
    for LX in teams_key:
        _concat[LX] = tf.concat( [teams_key[LX], tf.tile( tf.expand_dims(nodes,1), [1,tf.shape(teams_key[LX])[1],1] )], axis=2 )

    op_act = {}
    op_nlogp = {}
    op_ent = {}
    op_pd = {}

    param_act = {}
    param_nlogp = {}
    param_ent = {}
    param_pd = {}

    # aval_act = {}
    # target_id_pd = {}

    _act = {}
    for LX in ["11", "12", "14"]:
        '''
            (2+1+2+1) = 6
            (pxy, direction, length and width, delay_time/patrol_time)
            to
            (2+1) = 3
            (pxy, delay_time/patrol_time)
        '''
        if 0:
            _act[LX] = tf.layers.dense( _concat[LX], 2+(2+1) )
        else:
            _act[LX] = tf.layers.conv1d( _concat[LX] , 2+(2+1), 1)
        op_act[LX], op_nlogp[LX], op_ent[LX], op_pd[LX] = get_pd_discrete(tf.slice(_act[LX], [0, 0, 0], [-1, -1, 2]))
        param_act[LX] = {}
        param_nlogp[LX] = {}
        param_ent[LX] = {}
        param_pd[LX] = {}
        MT = "areapatrol"
        param_act[LX][MT], param_nlogp[LX][MT], param_ent[LX][MT], param_pd[LX][MT] = get_pd_continuous(tf.slice(_act[LX], [0, 0, 2], [-1, -1, 3]), i_size=3)

    LX = "15"
    i_slice = 0
    i_size = 4  # noop areapatrol targethunt areahunt
    if 0:
        _act[LX] = tf.layers.dense(_concat[LX], i_size + (2 + 1) + (3+3*2) + (2+2*6))
    else:
        _act[LX] = tf.layers.conv1d(_concat[LX], i_size + (2 + 1)*2 + (3+3*3*2) + (2+2*6*2), 1)

    op_act[LX], op_nlogp[LX], op_ent[LX], op_pd[LX] = get_pd_discrete(tf.slice(_act[LX], [0, 0, 0], [-1, -1, i_size]))
    i_slice += i_size

    assert i_slice == 4

    param_act[LX] = {}
    param_nlogp[LX] = {}
    param_ent[LX] = {}
    param_pd[LX] = {}

    def tf_scale_2(latent, mm):
        pass
        # tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)

    MT = "areapatrol"
    mean = []
    # all 3
    for minmax in [[X_Min + 1100, -93408], [(Y_Min + 1100), (Y_Max - 1100)], [0, 100]]:
        i_size = 1
        mean.append( scale_to(tf.tanh(tf.slice(_act[LX], [0, 0, i_slice], [-1, -1, i_size])), minmax[0], minmax[1]) )
        i_slice += i_size
    mean = tf.concat(mean, axis=-1)

    logstd = []
    # all 3
    for m in [10e3, 10e3, 10.]:
        i_size = 1
        logstd.append( tf.slice(_act[LX], [0, 0, i_slice], [-1, -1, i_size]) + tf.log(m) )
        i_slice += i_size
    logstd = tf.concat(logstd, axis=-1)

    param_act[LX][MT], param_nlogp[LX][MT], param_ent[LX][MT], param_pd[LX][MT] = get_pd_continuous_2(mean, logstd)

    assert i_slice == 4 + 6

    '''
        will it suitable for discrete choice? check it with experiments.
    '''
    MT = "target_wh_hunt"
    i_size = 3
    param_act[LX][MT], param_nlogp[LX][MT], param_ent[LX][MT], param_pd[LX][MT] = get_pd_discrete(tf.slice(_act[LX], [0, 0, i_slice], [-1, -1, i_size]))
    i_slice += i_size

    assert i_slice == 4 + 6 + 3


    for MT in ["target0_hunt", "target1_hunt", "target2_hunt"]:
        # direction, range and delay_time


        mean = []
        # all 3
        for minmax in [[0, 360], [60, 100], [0, 100]]:
            i_size = 1
            mean.append(scale_to(tf.tanh(tf.slice(_act[LX], [0, 0, i_slice], [-1, -1, i_size])), minmax[0], minmax[1]))
            i_slice += i_size
        mean = tf.concat(mean, axis=-1)

        logstd = []
        # all 3
        for m in [10., 5., 10.]:
            i_size = 1
            logstd.append(tf.slice(_act[LX], [0, 0, i_slice], [-1, -1, i_size]) + tf.log(m))
            i_slice += i_size
        logstd = tf.concat(logstd, axis=-1)

        param_act[LX][MT], param_nlogp[LX][MT], param_ent[LX][MT], param_pd[LX][MT] = get_pd_continuous_2(mean, logstd)

    assert i_slice == 4 + 6 + 3 + 3*3*2

    MT = "area_wh_hunt"
    i_size = 2  # south or north
    param_act[LX][MT], param_nlogp[LX][MT], param_ent[LX][MT], param_pd[LX][MT] = get_pd_discrete(tf.slice(_act[LX], [0, 0, i_slice], [-1, -1, i_size]))
    i_slice += i_size
    assert i_slice == 4 + 6 + 3 + 3*3*2 + 2

    for MT in ["area0_hunt", "area1_hunt"]:
        # direction, range and delay_time
        # area_direction, area_length, area_width

        mean = []
        for minmax in [[0, 360], [60, 100], [0, 100], [0, 180], [5e3, 50e3], [5e3, 50e3]]:
            i_size = 1
            mean.append(scale_to(tf.tanh(tf.slice(_act[LX], [0, 0, i_slice], [-1, -1, i_size])), minmax[0], minmax[1]))
            i_slice += i_size
        mean = tf.concat(mean, axis=-1)

        logstd = []
        for m in [10., 5., 10., 5., 5e3, 5e3]:
            i_size = 1
            logstd.append(tf.slice(_act[LX], [0, 0, i_slice], [-1, -1, i_size]) + tf.log(m))
            i_slice += i_size
        logstd = tf.concat(logstd, axis=-1)

        param_act[LX][MT], param_nlogp[LX][MT], param_ent[LX][MT], param_pd[LX][MT] = get_pd_continuous_2(mean, logstd)
    assert i_slice == 4 + 6 + 3 + 3*3*2 + 2 + 2*6*2

    assert i_slice == 4 + (2 + 1)*2 + (3+3*3*2) + (2+2*6*2)

    return op_act, op_nlogp, op_ent, op_pd, param_act, param_nlogp, param_ent, param_pd

    units_keys = {}
    for LX in units_key:
        units_keys[LX] = tf.layers.dense( units_key[LX] ,5,activation=tf.nn.relu)

    teams_keys = {}
    for LX in teams_key:
        teams_keys[LX] = tf.layers.dense( teams_key[LX] ,5,activation=tf.nn.relu)

    target_id_keys = tf.layers.dense( qb_key ,5,activation=tf.nn.relu)

    type4cmd = get_type4cmd()

    # Parse all the args from nodes.
    action = {}
    i_slice = 0

    neglogp_discrete = {}
    ent_discrete = {}
    pd_discrete = {}

    neglogp_continuous = {}
    ent_continuous = {}
    pd_continuous = {}

    k = "maintype"
    i_size = 36-6
    action[k], neglogp_discrete[k], ent_discrete[k], pd_discrete[k] = get_pd_discrete(
        tf.slice(nodes, [0, i_slice], [-1, i_size]))
    i_slice += i_size

    '''
        self_id choose from obs.teams or units
    '''

    # print(action["maintype"])
    # Tensor("ArgMax:0", shape=(1,), dtype=int64)
    with tf.variable_scope("maintype-mask"):
        maintype_hot = tf.one_hot(action["maintype"],36-6)
        assert maintype_hot.shape.as_list()[1:]==[36-6]
        maintype_mask = tf.boolean_mask(tf.tile(tf.expand_dims(type4cmd, 0), [tf.shape(maintype_hot)[0], 1, 1]),maintype_hot)
    '''
        print("inputs of mask-keys")
        print("="*30)
        print(units_keys)
        # [tf.Tensor 'dense/Relu:0' shape=(?, ?, 5) dtype=float32]
        print(teams_keys)
        # [tf.Tensor 'dense_7/Relu:0' shape=(?, ?, 5) dtype=float32]
        print(maintype_mask)
        # Tensor("maintype-mask/boolean_mask/GatherV2:0", shape=(?, 9), dtype=float32)
        print(units_mask)
        # [tf.Tensor 'Slice_6:0' shape=(?, ?, 1) dtype=float32]
        print(teams_mask)
        # [tf.Tensor 'Slice_57:0' shape=(?, ?, 1) dtype=float32]
        print("")
    '''

    with tf.variable_scope("mask-keys"):

        unit_team_mask = []
        # for i, LX in enumerate(units_keys):
        for i, LX in enumerate("11, 12, 13, 14, 15, 21, 32".split(", ")):
            mask = tf.tensordot(tf.slice(maintype_mask, [0, 7], [-1, 2]), tf.constant([1., 0.]), axes=1) * \
                   tf.tensordot(tf.slice(maintype_mask, [0, 0], [-1, 7]), \
                                tf.constant([1. if k == i else 0. for k in range(len(units_keys))]), axes=1)
            # shape=(batch_size,)
            mask = tf.expand_dims(mask, -1)
            mask = tf.expand_dims(mask, 1)
            # shape=(batch_size,1,1)
            mask = tf.tile(mask, [1, tf.shape(units_keys[LX])[1], 1])
            # shape=(batch_size,LX_length,1)

            mask = mask * units_mask[LX]
            # shape=(batch_size,LX_length,1)

            unit_team_mask.append(mask)

        # for i, LX in enumerate(teams_keys):
        for i, LX in enumerate("11, 12, 13, 14, 15, 21, 32".split(", ")):
            mask = tf.tensordot(tf.slice(maintype_mask, \
                                         [0, 7], [-1, 2]), tf.constant([0., 1.]), axes=1) * \
                   tf.tensordot(tf.slice(maintype_mask, [0, 0], [-1, 7]), \
                                tf.constant([1. if k == i else 0. for k in range(len(teams_keys))]), axes=1)
            # shape=(batch_size,)
            mask = tf.expand_dims(mask, -1)
            mask = tf.expand_dims(mask, 1)
            mask = tf.tile(mask, [1, tf.shape(teams_keys[LX])[1], 1])
            mask = mask * teams_mask[LX]
            # shape=(batch_size,LX_length,1)
            unit_team_mask.append(mask)

        unit_team_mask = tf.concat(unit_team_mask, axis=1)
        # shape=(batch_size,LX_length_all,1)

        # collect all keys of units and teams.
        unit_team_keys = []
        # for i, LX in enumerate(units_keys):
        for i, LX in enumerate("11, 12, 13, 14, 15, 21, 32".split(", ")):
            unit_team_keys.append(units_keys[LX])
        # for i, LX in enumerate(teams_keys):
        for i, LX in enumerate("11, 12, 13, 14, 15, 21, 32".split(", ")):
            unit_team_keys.append(teams_keys[LX])
        unit_team_keys = tf.concat(unit_team_keys, axis=1)
        # shape=(batch_size,LX_length_all,5)

    unit_team_dot = tf.keras.backend.batch_dot(tf.slice(nodes, [0, i_slice], [-1, 5]), unit_team_keys, axes=[1, 2])
    # shape = (batch_size,LX_length_all)

    if 0:
        action["self_id"] = tf.argmax(tf.multiply(tf.nn.softmax(unit_team_dot),tf.squeeze(unit_team_mask,2)),axis=1)
    else:
        self_id_pd = CategoricalPd_var_mask(unit_team_dot, tf.squeeze(unit_team_mask,2))
        action["self_id"] = self_id_pd.sample()
        # self_id_pd.neglogp(act_sample)
        # self_id_ent = self_id_pd.entropy()

    ink = [unit_team_mask]
    # shape = (batch_size,)
    i_slice += 5

    '''
        cov_id choose from obs.teams
    '''

    type4cov = ["12", "13", "15"]

    with tf.variable_scope("cov-keys"):
        cov_mask = []
        for LX in type4cov:
            cov_mask.append(teams_mask[LX])

        cov_mask = tf.concat(cov_mask, axis=1)

        # collect all keys of teams for cov.
        cov_id_keys = []
        for LX in type4cov:
            cov_id_keys.append(teams_keys[LX])
        cov_id_keys = tf.concat(cov_id_keys, axis=1)

    cov_id_dot = tf.keras.backend.batch_dot(tf.slice(nodes, [0,i_slice], [-1,5]),cov_id_keys,axes=[1,2])
    if 0:
        action["cov_id"] = tf.argmax(tf.multiply(tf.nn.softmax(cov_id_dot),tf.squeeze(cov_mask,2)),axis=1)
    else:
        cov_id_pd = CategoricalPd_var_mask(cov_id_dot, tf.squeeze(cov_mask,2))
        action["cov_id"] = cov_id_pd.sample()
        # self_id_pd.neglogp(act_sample)
        # self_id_ent = self_id_pd.entropy()

    i_slice += 5

    '''
        target_id choose from qb, MAKE SURE TARGET IS ENEMY! exclude neutral items
    '''

    target_id_dot = tf.keras.backend.batch_dot(tf.slice(nodes, [0,i_slice], [-1,5]), target_id_keys, axes=[1, 2])

    if 0:
        action["target_id"] = tf.argmax(tf.multiply(tf.nn.softmax(target_id_dot),tf.squeeze(qb_mask,2)),axis=1)
    else:
        target_id_pd = CategoricalPd_var_mask(target_id_dot, tf.squeeze(qb_mask,2))
        action["target_id"] = target_id_pd.sample()
        # self_id_pd.neglogp(act_sample)
        # self_id_ent = self_id_pd.entropy()

    i_slice += 5

    '''
        action["airport_id"]
            both the red and blue have only one airport

        action["area_id"]
            default int 1

        action["patrol_mode"]
            default 0

        action["fly_type"]
            red side has 5 fly_types [11,12,13,14,15]
            blue side has 3 fly_types
    '''

    '''
        continuous 
        
        split params by maintype.
    '''
    cmd_list = get_cmd_list()
    for k in ["point_list", "fly_num", "point_x", "point_y", "point_z", "direction", "length", "width", "speed", "patrol_time", "range", "area_direct", "area_len", "area_wid", "offset"]:
        i_size = 1
        if k=="point_list":
            i_size = 3*5

        action[k] = {}
        neglogp_continuous[k] = {}
        ent_continuous[k] = {}
        pd_continuous[k] = {}

        for cmd in cmd_list:
            MT = cmd["maintype"]

            if k == "patrol_time":
                k_name2 = "disturb_time"
            elif k == "mode":
                k_name2 = "modle"
            else:
                k_name2 = k

            if k in cmd or k_name2 in cmd or ("area" in cmd and k in ["area_id", "area_type", "point_list"]):
                action[k][MT], neglogp_continuous[k][MT], ent_continuous[k][MT], pd_continuous[k][MT] = get_pd_continuous(tf.slice(nodes, [0, i_slice], [-1, i_size]), i_size=i_size)
                i_slice += i_size

    '''
        discrete
    '''
    for k, i_size in zip(["fly_type", "on_off", "radar_state", "mode", "flag", "type", "area_type"],[5, 2, 2, 3, 4, 2, 2]):

        action[k] = {}
        neglogp_discrete[k] = {}
        ent_discrete[k] = {}
        pd_discrete[k] = {}

        for cmd in cmd_list:
            MT = cmd["maintype"]

            if k == "patrol_time":
                k_name2 = "disturb_time"
            elif k == "mode":
                k_name2 = "modle"
            else:
                k_name2 = k

            if k in cmd or k_name2 in cmd or ("area" in cmd and k in ["area_id", "area_type", "point_list"]):
                action[k][MT], neglogp_discrete[k][MT], ent_discrete[k][MT], pd_discrete[k][MT] = get_pd_discrete(tf.slice(nodes, [0,i_slice], [-1,i_size]))
                i_slice += i_size

    '''
        give a long enough nodes first for using. and later see how many nodes need actually.
    '''
    print("")
    print("i_slice:",i_slice)
    print("")

    ent_aval = {k: i.entropy() for k, i in zip(["self_id", "cov_id", "target_id"], [self_id_pd, cov_id_pd, target_id_pd])}

    return action, neglogp_discrete, ent_discrete, pd_discrete, neglogp_continuous, ent_continuous, pd_continuous, [pd.neglogp_all() for pd in [self_id_pd, cov_id_pd, target_id_pd]], ent_aval, self_id_pd, cov_id_pd, target_id_pd

def icm(x,min_val=0):
    return int(np.ceil(max(x, min_val)))



def get_maintype_list():
    return ['areapatrol',
     'takeoffareapatrol',
     'linepatrol',
     'takeofflinepatrol',
     'areahunt',
     'takeoffareahunt',
     'targethunt',
     'takeofftargethunt',
     'protect',
     'takeoffprotect',
     'airattack',
     'returntobase',

     # 'Ground_Add_Target',
     # 'Ground_Remove_Target',
     # 'GroundRadar_Control',
     # 'Ground_Set_Direction',
     # 'Ground_Move_Deploy',

     # 'Ship_Move_Deploy',
     'Ship_areapatrol',
     'Ship_Add_Target',
     'Ship_Remove_Target',
     'Ship_Radar_Control',

     'awcs_areapatrol',
     'awcs_linepatrol',
     'awcs_mode',
     'awcs_radarcontrol',
     'awcs_cancledetect',

     'area_disturb_patrol',
     'line_disturb_patrol',
     'set_disturb',
     'close_disturb',
     'stop_disturb',

     'uav_areapatrol',
     'uav_linepatrol',
     'uav_cancledetect',

     'base_radarcontrol']

def cmd_convert(action, cov_id_list, self_id_list, target_id_list, team_id, obs, obs_red_const, sim_time, next_act_delay):
    # 战场地图边界(x正向, x负向, y正向, y负向), 单位: 米
    X_Max = 174500
    X_Min = -170000
    Y_Max = 174500
    Y_Min = -171000

    op_act, op_nlogp, op_ent, param_act, param_nlogp, param_ent = action

    '''
        init act_inc
    '''
    import copy
    act_inc = copy.deepcopy(param_act)
    for LX in act_inc:
        for MT in act_inc[LX]:
            act_inc[LX][MT] = np.array([[[0]]*len(act_inc[LX][MT][0])])



    '''
        init op_act_inc
    '''
    op_act_inc = copy.deepcopy(op_act)
    for LX in op_act_inc:
        for i, _ in enumerate(op_act_inc[LX][0]):
            op_act_inc[LX][0][i] = 0
    '''
        test add op_act_inc and without it.
        if without it, it might bring noise for the operation choose of networks.
    '''
    # aval_act, neglogp_all = aval_action


    '''
        (2+1+2+1) = 6
        (pxy, direction, length and width, delay_time/patrol_time)
    '''

    cmd_list = []
    nlogp = 0
    speed = {"11": 250, "12": 200, "14": 90, "15": 200}
    for LX in speed:
        for i, id in enumerate(team_id[LX]):
            if not id == 0:
                '''
                    control with delay time.
                '''
                if (not id in next_act_delay) or sim_time > (next_act_delay[id]["sim_time"]+next_act_delay[id]["delay_time"]):
                    nlogp += op_nlogp[LX][0][i]
                    op_act_inc[LX][0][i] = 1

                    if op_act[LX][0][i] == 0:
                        # noop
                        cmd_list.extend([])

                        '''
                            add act_inc hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
                        '''
                    elif op_act[LX][0][i] == 1:
                        self_id = id
                        MT = "areapatrol"

                        # for minmax in [[X_Min + 1100, -93408], [(Y_Min + 1100), (Y_Max - 1100)], [0, 100]]:

                        if LX == "15":
                            point_x = param_act[LX][MT][0][i][0]
                            point_y = param_act[LX][MT][0][i][1]
                            delay_time = param_act[LX][MT][0][i][2]

                            act_inc[LX][MT][0][i][0] = 1

                            point_x = np.clip(point_x, X_Min + 1100, -93408)
                            point_y = np.clip(point_y, (Y_Min + 1100), (Y_Max - 1100))
                            delay_time = np.clip(delay_time, 0, 100)
                            delay_time = int(delay_time)
                        else:
                            point_x = param_act[LX][MT][0][i][0]
                            point_y = param_act[LX][MT][0][i][1]
                            # direction = param_act[LX][MT][0][i][2]
                            # length = param_act[LX][MT][0][i][3]
                            # width = param_act[LX][MT][0][i][4]
                            delay_time = param_act[LX][MT][0][i][2]

                            act_inc[LX][MT][0][i][0] = 1

                            point_x = clip_2(point_x, X_Min + 1100, X_Max - 1100)
                            point_y = clip_2(point_y, Y_Min + 1100, Y_Max - 1100)

                            # if LX == "13":
                            #     # disturb
                            #     if not sim_time > 30 * 60:
                            #         point_x += 146228 - 10e3
                            #     else:
                            #         point_x += -130870 + 20e3
                            #     point_x = np.clip(point_x, X_Min + 1100, X_Max - 1100)

                            # direction = clip_2(direction, 0, 180)
                            # length = scale_to(length, 5e3, 50e3)
                            # length = np.clip(length, 1e3, 200e3)
                            # width = scale_to(width, 5e3, 8e3)
                            # width = np.clip(width, 1e3, 10e3)
                            delay_time = scale_to(delay_time, 0, 30)
                            delay_time = np.clip(delay_time, 0, 60)
                            # direction = int(direction)
                            # length = int(length)
                            # width = int(width)
                            delay_time = int(delay_time)

                        point_x = np.clip(point_x, X_Min + 1100, X_Max - 1100)
                        point_y = np.clip(point_y, Y_Min + 1100, Y_Max - 1100)
                        point_x = int(point_x)
                        point_y = int(point_y)

                        direction = 90
                        length = 2e3
                        width = 2e3

                        '''
                            update next_act_delay
                        '''
                        next_act_delay[id] = {
                            "sim_time": sim_time,
                            "delay_time": delay_time,
                        }

                        if 0:  # use linepatrol
                            point_list = [Point(point_x, point_y, 8000)]
                            if LX == "12":
                                cmd_list.extend([EnvCmd.make_awcs_linepatrol(self_id, speed[LX], area_id=1, area_type="line", point_list=point_list)])
                            elif LX == "13":
                                cmd_list.extend([EnvCmd.make_disturb_linepatrol(self_id, speed[LX], area_id=1, area_type="line", point_list=point_list)])
                            elif LX == "14":
                                cmd_list.extend([EnvCmd.make_uav_linepatrol(self_id, speed[LX], area_id=1, area_type="line", point_list=point_list)])
                            elif LX in ["11", "15"]:
                                cmd_list.extend([EnvCmd.make_linepatrol(self_id, speed[LX], area_id=1, area_type="line", point_list=point_list)])
                            else:
                                assert 0
                        else:  # replace () with () for same effect.
                            patrol_point = [point_x, point_y, 8000]
                            patrol_params = [direction, length, width, speed[LX], 7200, 0]
                            if LX == "12":
                                cmd_list.extend([EnvCmd.make_awcs_areapatrol(self_id, *patrol_point, *patrol_params)])
                            elif LX == "13":
                                # special function for its params.
                                cmd_list.extend(
                                    [EnvCmd.make_disturb_areapatrol(self_id, *patrol_point, *patrol_params[:-1])])
                            elif LX == "14":
                                cmd_list.extend([EnvCmd.make_uav_areapatrol(self_id, *patrol_point, *patrol_params)])
                            elif LX == "21":
                                cmd_list.extend([EnvCmd.make_ship_areapatrol(self_id, *patrol_point, *patrol_params)])
                            elif LX in ["11", "15"]:
                                cmd_list.extend([EnvCmd.make_areapatrol(self_id, *patrol_point, *patrol_params)])
                            else:
                                assert 0

                        nlogp += param_nlogp[LX][MT][0][i]

                    elif LX == "15" and op_act[LX][0][i] == 2:
                        self_id = id
                        MT = "target_wh_hunt"
                        # 'target_wh_hunt': array([[1.0263256]], dtype=float32)
                        nlogp += param_nlogp[LX][MT][0][i]
                        act_inc[LX][MT][0][i][0] = 1

                        # 'target_wh_hunt': array([[0]])
                        target_wh_hunt = param_act[LX][MT][0][i]
                        assert target_wh_hunt in [0,1,2]

                        # for minmax in [[0, 360], [60, 100], [0, 100]]:

                        MT = ["target0_hunt", "target1_hunt", "target2_hunt"][target_wh_hunt]
                        # 'target1_hunt': array([[3.7250278]], dtype=float32)
                        nlogp += param_nlogp[LX][MT][0][i]
                        act_inc[LX][MT][0][i][0] = 1

                        direction = param_act[LX][MT][0][i][0]
                        range = param_act[LX][MT][0][i][1]
                        delay_time = param_act[LX][MT][0][i][2]

                        direction = np.clip(direction, 0, 360)
                        direction = int(direction)

                        range = np.clip(range, 60, 100)
                        range = int(range)

                        delay_time = np.clip(delay_time, 0, 100)
                        delay_time = int(delay_time)

                        '''
                            update next_act_delay
                        '''
                        next_act_delay[id] = {
                            "sim_time": sim_time,
                            "delay_time": delay_time,
                        }

                        cmd_list.extend( targethunt_rule(self_id, target_wh_hunt, direction, range, obs_red_const) )

                    elif LX == "15" and op_act[LX][0][i] == 3:
                        self_id = id
                        MT = "area_wh_hunt"
                        nlogp += param_nlogp[LX][MT][0][i]
                        act_inc[LX][MT][0][i][0] = 1

                        area_wh_hunt = param_act[LX][MT][0][i]
                        assert area_wh_hunt in [0,1]
                        MT = ["area0_hunt", "area1_hunt"][area_wh_hunt]
                        nlogp += param_nlogp[LX][MT][0][i]
                        act_inc[LX][MT][0][i][0] = 1

                        # for minmax in [[0, 360], [60, 100], [0, 100], [0, 180], [5e3, 50e3], [5e3, 50e3]]:

                        direction = param_act[LX][MT][0][i][0]
                        range = param_act[LX][MT][0][i][1]
                        delay_time = param_act[LX][MT][0][i][2]
                        area_direct = param_act[LX][MT][0][i][3]
                        area_len = param_act[LX][MT][0][i][4]
                        area_wid = param_act[LX][MT][0][i][5]

                        direction = np.clip(direction, 0, 360)
                        direction = int(direction)

                        range = np.clip(range, 60, 100)
                        range = int(range)

                        delay_time = np.clip(delay_time, 0, 100)
                        delay_time = int(delay_time)

                        area_direct = np.clip(area_direct, 0, 180)
                        area_direct = int(area_direct)

                        area_len = np.clip(area_len, 5e3, 50e3)
                        area_len = int(area_len)

                        area_wid = np.clip(area_wid, 5e3, 50e3)
                        area_wid = int(area_wid)

                        '''
                            update next_act_delay
                        '''
                        next_act_delay[id] = {
                            "sim_time": sim_time,
                            "delay_time": delay_time,
                        }

                        cmd_list.extend( areahunt_rule(self_id, area_wh_hunt, direction, range, obs_red_const, area_direct, area_len, area_wid) )
    return cmd_list, nlogp, act_inc, op_act_inc, next_act_delay

    maintype_list = get_maintype_list()

    # Parse all params.
    cmd = {}
    cmd["maintype"] = maintype_list[action["maintype"][0]]

    '''
        self_id choose from obs.units and obs.teams.
    '''
    assert action["self_id"][0]<=len(self_id_list)
    cmd["self_id"] = self_id_list[action["self_id"][0]] if action["self_id"][0]<len(self_id_list) else -1

    assert action["cov_id"][0]<=len(cov_id_list)
    cmd["cov_id"] = cov_id_list[action["cov_id"][0]] if action["cov_id"][0]<len(cov_id_list) else -1

    '''
        target_id choose from qb, MAKE SURE TARGET IS ENEMY! exclude neutral items.
    '''
    assert action["target_id"][0]<=len(target_id_list)
    cmd["target_id"] = target_id_list[action["target_id"][0]] if action["target_id"][0]<len(target_id_list) else -1

    '''
        cmd["airport_id"]
            both the red and blue have only one airport
        cmd["area_id"]
            default int 1
    '''

    # cmd["airport_id"] = airport_id
    # cmd["area_id"] = 1

    '''
        continuous 
    '''
    '''
        list 
        set the length be 5 (5*3=15) and test.
    '''
    cmd["point_list"] = {}
    for MT in action["point_list"]:
        cmd["point_list"][MT] = [
            Point(int(np.ceil(action["point_list"][MT][0][0] * 175e3)),
                  int(np.ceil(action["point_list"][MT][0][1] * 175e3)),
                  int(clip_2(action["point_list"][MT][0][2], 1000, 11000))),
            Point(int(np.ceil(action["point_list"][MT][0][3] * 175e3)),
                  int(np.ceil(action["point_list"][MT][0][4] * 175e3)),
                  int(clip_2(action["point_list"][MT][0][5], 1000, 11000))),
            Point(int(np.ceil(action["point_list"][MT][0][6] * 175e3)),
                  int(np.ceil(action["point_list"][MT][0][7] * 175e3)),
                  int(clip_2(action["point_list"][MT][0][8], 1000, 11000))),
            Point(int(np.ceil(action["point_list"][MT][0][9] * 175e3)),
                  int(np.ceil(action["point_list"][MT][0][10] * 175e3)),
                  int(clip_2(action["point_list"][MT][0][11], 1000, 11000))),
            Point(int(np.ceil(action["point_list"][MT][0][12] * 175e3)),
                  int(np.ceil(action["point_list"][MT][0][13] * 175e3)),
                  int(clip_2(action["point_list"][MT][0][14], 1000, 11000)))
        ]

    cmd["fly_num"] = {}
    for MT in action["fly_num"]:
        cmd["fly_num"][MT] = int(clip_2(action["fly_num"][MT][0][0],1,10))

    cmd["point_x"] = {}
    for MT in action["point_x"]:
        cmd["point_x"][MT] = int(np.ceil(action["point_x"][MT][0][0] * 175e3))

    cmd["point_y"] = {}
    for MT in action["point_y"]:
        cmd["point_y"][MT] = int(np.ceil(action["point_y"][MT][0][0] * 175e3))

    cmd["point_z"] = {}
    for MT in action["point_z"]:
        cmd["point_z"][MT] = int(clip_2(action["point_z"][MT][0][0],1000,11000))

    # special, scale with 0-360 or 0-180
    cmd["direction"] = {}
    for MT in action["direction"]:
        cmd["direction"][MT] = action["direction"][MT][0][0]

    cmd["length"] = {}
    for MT in action["length"]:
        cmd["length"][MT] = icm(action["length"][MT][0][0]*8000+8000,200)

    cmd["width"] = {}
    for MT in action["width"]:
        cmd["width"][MT] = icm(action["width"][MT][0][0]*8000+8000,200)

    cmd["patrol_time"] = {}
    for MT in action["patrol_time"]:
        cmd["patrol_time"][MT] = icm(action["patrol_time"][MT][0][0]*3600+3600,60)

    cmd["range"] = {}
    for MT in action["range"]:
        cmd["range"][MT] = int(clip_2(action["range"][MT][0][0],1,100))

    cmd["area_direct"] = {}
    for MT in action["area_direct"]:
        cmd["area_direct"][MT] = int(clip_2(action["area_direct"][MT][0][0],0,180))

    cmd["area_len"] = {}
    for MT in action["area_len"]:
        cmd["area_len"][MT] = icm(action["area_len"][MT][0][0]*8000+8000,200)

    cmd["area_wid"] = {}
    for MT in action["area_wid"]:
        cmd["area_wid"][MT] = icm(action["area_wid"][MT][0][0]*8000+8000,200)

    cmd["offset"] = {}
    for MT in action["offset"]:
        cmd["offset"][MT] = int(clip_2(action["offset"][MT][0][0],1,100))

    '''
        discrete
    '''

    '''
        cmd["patrol_mode"]
            default 0

        fly_type
            red 5 types [11,12,13,14,15]
            blue 3 types
    '''
    # cmd["patrol_mode"] = 0

    cmd["fly_type"] = {}
    for MT in action["fly_type"]:
        cmd["fly_type"][MT] = ([11,12,13,14,15])[action["fly_type"][MT][0]]

    cmd["speed"] = {}
    for MT in action["speed"]:
        cmd["speed"][MT] = action["speed"][MT][0][0]

    cmd["on_off"] = {}
    for MT in action["on_off"]:
        cmd["on_off"][MT] = ([0,1])[action["on_off"][MT][0]]

    cmd["radar_state"] = {}
    for MT in action["radar_state"]:
        cmd["radar_state"][MT] = ([0,1])[action["radar_state"][MT][0]]

    cmd["mode"] = {}
    for MT in action["mode"]:
        cmd["mode"][MT] = ([0,1,2])[action["mode"][MT][0]]

    cmd["flag"] = {}
    for MT in action["flag"]:
        cmd["flag"][MT] = ([1,2,3,4])[action["flag"][MT][0]]

    cmd["type"] = {}
    for MT in action["type"]:
        cmd["type"][MT] = ([0,1])[action["type"][MT][0]]

    cmd["area_type"] = {}
    for MT in action["area_type"]:
        cmd["area_type"][MT] = (["line", "area"])[action["area_type"][MT][0]]

    return cmd

def cmd_fetch(cmd, airport_id, obs):
    return [], {}
    default_cmd = {}
    default_cmd["airport_id"] = airport_id
    default_cmd["area_id"] = 1
    default_cmd["patrol_mode"] = 0

    # init act_include from cmd
    act_include = {}
    for k1 in cmd:
        if type(cmd[k1])==type({"a":0}):
            act_include[k1] = {k2:0 for k2 in cmd[k1]}
        else:
            act_include[k1] = 0

    # Choose params according to maintype.
    cmd_list = get_cmd_list()
    for item in cmd_list:
        if item["maintype"]==cmd["maintype"]:
            cmd2 = item
            break

    act_include["maintype"] = 1
    # # when self_id, cov_id or target_id is null, and cmd2 need them.
    # for k in cmd2:
    #     if k == "self_id" and cmd[k]==-1:
    #         act_include["self_id"] = 1
    #         return [], act_include
    #     if k == "cov_id" and cmd[k]==-1:
    #         act_include["self_id"] = 1
    #         return [], act_include
    #
    #     if k == "target_id" and cmd[k]==-1:
    #         act_include["self_id"] = 1
    #         return [], act_include

    MT = cmd2["maintype"]
    for k in cmd2:
        if k=="direction":
            # special param: direction
            if cmd["maintype"] in ['areahunt', 'takeoffareahunt', 'targethunt', 'takeofftargethunt', 'Ground_Set_Direction', 'Ground_Move_Deploy', 'Ship_Move_Deploy']:
                # 0-360
                cmd2[k] = int(clip_2(cmd[k][MT],0,360))
            else:
                # 0-180
                cmd2[k] = int(clip_2(cmd[k][MT],0,180))

            act_include[k][MT] = 1

        elif k=="area":
            cmd2[k] = EnvCmd._make_area(default_cmd["area_id"], cmd["area_type"][MT], cmd["point_list"][MT])

            act_include["area_type"][MT] = 1
            act_include["point_list"][MT] = 1

        elif k=="disturb_time":
            cmd2[k] = cmd["patrol_time"][MT]

            act_include["patrol_time"][MT] = 1

        elif k=="modle":
            cmd2[k] = cmd["mode"][MT]

            act_include["mode"][MT] = 1
        elif k in default_cmd:
            cmd2[k] = default_cmd[k]
        else:
            if k in ["maintype", "self_id", "cov_id", "target_id"]:
                cmd2[k] = cmd[k]
                act_include[k] = 1
            elif k in ["airport_id", "patrol_mode"]:
                cmd2[k] = cmd[k]
            else:
                try:
                    cmd2[k] = cmd[k][MT]
                    act_include[k][MT] = 1

                except Exception as e:
                    print(e)
                    print(k)
                    assert 0

    if cmd2['maintype']=='Ship_areapatrol':
        cmd2['point_z'] = 0  # same effect with clip(, 0, 0)

    if "speed" in cmd2:
        if "fly_type" in cmd2:
            cmd2["speed"] = get_speed(cmd["speed"][MT],cmd["maintype"],cmd["fly_type"][MT])

        cmd2 = deal_speed(cmd2, cmd["speed"][MT],obs)

    return [cmd2], act_include


def decode_cmd_fun(action, airport_id, cov_id_list, self_id_list, target_id_list,obs, team_id, obs_red_const, sim_time, next_act_delay):
    return cmd_convert(action, cov_id_list, self_id_list, target_id_list, team_id, obs, obs_red_const, sim_time, next_act_delay)

    # maintype_list = get_maintype_list()
    #
    # # Parse all params.
    # cmd = {}
    # cmd["maintype"] = maintype_list[action["maintype"][0]]
    #
    # '''
    #     self_id choose from obs.units and obs.teams.
    # '''
    # assert action["self_id"][0]<=len(self_id_list)
    # cmd["self_id"] = self_id_list[action["self_id"][0]] if action["self_id"][0]<len(self_id_list) else -1
    #
    # assert action["cov_id"][0]<=len(cov_id_list)
    # cmd["cov_id"] = cov_id_list[action["cov_id"][0]] if action["cov_id"][0]<len(cov_id_list) else -1
    #
    # '''
    #     target_id choose from qb, MAKE SURE TARGET IS ENEMY! exclude neutral items.
    # '''
    # assert action["target_id"][0]<=len(target_id_list)
    # cmd["target_id"] = target_id_list[action["target_id"][0]] if action["target_id"][0]<len(target_id_list) else -1
    #
    # '''
    #     cmd["airport_id"]
    #         both the red and blue have only one airport
    #     cmd["area_id"]
    #         default int 1
    # '''
    #
    # cmd["airport_id"] = airport_id
    # cmd["area_id"] = 1
    #
    # '''
    #     continuous
    # '''
    # '''
    #     list
    #     set the length be 5 (5*3=15) and test.
    # '''
    # cmd["point_list"] = {}
    # for MT in action["point_list"]:
    #     cmd["point_list"][MT] = [
    #         Point(int(np.ceil(action["point_list"][MT][0][0] * 175e3)),
    #               int(np.ceil(action["point_list"][MT][0][1] * 175e3)),
    #               int(clip_2(action["point_list"][MT][0][2], 1000, 11000))),
    #         Point(int(np.ceil(action["point_list"][MT][0][3] * 175e3)),
    #               int(np.ceil(action["point_list"][MT][0][4] * 175e3)),
    #               int(clip_2(action["point_list"][MT][0][5], 1000, 11000))),
    #         Point(int(np.ceil(action["point_list"][MT][0][6] * 175e3)),
    #               int(np.ceil(action["point_list"][MT][0][7] * 175e3)),
    #               int(clip_2(action["point_list"][MT][0][8], 1000, 11000))),
    #         Point(int(np.ceil(action["point_list"][MT][0][9] * 175e3)),
    #               int(np.ceil(action["point_list"][MT][0][10] * 175e3)),
    #               int(clip_2(action["point_list"][MT][0][11], 1000, 11000))),
    #         Point(int(np.ceil(action["point_list"][MT][0][12] * 175e3)),
    #               int(np.ceil(action["point_list"][MT][0][13] * 175e3)),
    #               int(clip_2(action["point_list"][MT][0][14], 1000, 11000)))
    #     ]
    #
    # cmd["fly_num"] = {}
    # for MT in action["fly_num"]:
    #     cmd["fly_num"][MT] = int(clip_2(action["fly_num"][MT][0][0],1,10))
    #
    # cmd["point_x"] = {}
    # for MT in action["point_x"]:
    #     cmd["point_x"][MT] = int(np.ceil(action["point_x"][MT][0][0] * 175e3))
    #
    # cmd["point_y"] = {}
    # for MT in action["point_y"]:
    #     cmd["point_y"][MT] = int(np.ceil(action["point_y"][MT][0][0] * 175e3))
    #
    # cmd["point_z"] = {}
    # for MT in action["point_z"]:
    #     cmd["point_z"][MT] = int(clip_2(action["point_z"][MT][0][0],1000,11000))
    #
    # # special, scale with 0-360 or 0-180
    # cmd["direction"] = {}
    # for MT in action["direction"]:
    #     cmd["direction"][MT] = action["direction"][MT][0][0]
    #
    # cmd["length"] = {}
    # for MT in action["length"]:
    #     cmd["length"][MT] = icm(action["length"][MT][0][0]*8000+8000,200)
    #
    # cmd["width"] = {}
    # for MT in action["width"]:
    #     cmd["width"][MT] = icm(action["width"][MT][0][0]*8000+8000,200)
    #
    # cmd["patrol_time"] = {}
    # for MT in action["patrol_time"]:
    #     cmd["patrol_time"][MT] = icm(action["patrol_time"][MT][0][0]*3600+3600,60)
    #
    # cmd["range"] = {}
    # for MT in action["range"]:
    #     cmd["range"][MT] = int(clip_2(action["range"][MT][0][0],1,100))
    #
    # cmd["area_direct"] = {}
    # for MT in action["area_direct"]:
    #     cmd["area_direct"][MT] = int(clip_2(action["area_direct"][MT][0][0],0,180))
    #
    # cmd["area_len"] = {}
    # for MT in action["area_len"]:
    #     cmd["area_len"][MT] = icm(action["area_len"][MT][0][0]*8000+8000,200)
    #
    # cmd["area_wid"] = {}
    # for MT in action["area_wid"]:
    #     cmd["area_wid"][MT] = icm(action["area_wid"][MT][0][0]*8000+8000,200)
    #
    # cmd["offset"] = {}
    # for MT in action["offset"]:
    #     cmd["offset"][MT] = int(clip_2(action["offset"][MT][0][0],1,100))
    #
    # '''
    #     discrete
    # '''
    #
    # '''
    #     cmd["patrol_mode"]
    #         default 0
    #
    #     fly_type
    #         red 5 types [11,12,13,14,15]
    #         blue 3 types
    # '''
    # cmd["patrol_mode"] = 0
    #
    # cmd["fly_type"] = {}
    # for MT in action["fly_type"]:
    #     cmd["fly_type"][MT] = ([11,12,13,14,15])[action["fly_type"][MT][0]]
    #
    # cmd["speed"] = {}
    # for MT in action["speed"]:
    #     cmd["speed"][MT] = action["speed"][MT][0][0]
    #
    # cmd["on_off"] = {}
    # for MT in action["on_off"]:
    #     cmd["on_off"][MT] = ([0,1])[action["on_off"][MT][0]]
    #
    # cmd["radar_state"] = {}
    # for MT in action["radar_state"]:
    #     cmd["radar_state"][MT] = ([0,1])[action["radar_state"][MT][0]]
    #
    # cmd["mode"] = {}
    # for MT in action["mode"]:
    #     cmd["mode"][MT] = ([0,1,2])[action["mode"][MT][0]]
    #
    # cmd["flag"] = {}
    # for MT in action["flag"]:
    #     cmd["flag"][MT] = ([1,2,3,4])[action["flag"][MT][0]]
    #
    # cmd["type"] = {}
    # for MT in action["type"]:
    #     cmd["type"][MT] = ([0,1])[action["type"][MT][0]]
    #
    # cmd["area_type"] = {}
    # for MT in action["area_type"]:
    #     cmd["area_type"][MT] = (["line", "area"])[action["area_type"][MT][0]]
    #
    # # init act_include from cmd
    # act_include = {}
    # for k1 in cmd:
    #     if type(cmd[k1])==type({"a":0}):
    #         act_include[k1] = {k2:0 for k2 in cmd[k1]}
    #     else:
    #         act_include[k1] = 0
    #
    # # Choose params according to maintype.
    # cmd_list = get_cmd_list()
    # for item in cmd_list:
    #     if item["maintype"]==cmd["maintype"]:
    #         cmd2 = item
    #         break
    #
    # act_include["maintype"] = 1
    # # # when self_id, cov_id or target_id is null, and cmd2 need them.
    # # for k in cmd2:
    # #     if k == "self_id" and cmd[k]==-1:
    # #         act_include["self_id"] = 1
    # #         return [], act_include
    # #     if k == "cov_id" and cmd[k]==-1:
    # #         act_include["self_id"] = 1
    # #         return [], act_include
    # #
    # #     if k == "target_id" and cmd[k]==-1:
    # #         act_include["self_id"] = 1
    # #         return [], act_include
    #
    # MT = cmd2["maintype"]
    # for k in cmd2:
    #     if k=="direction":
    #         # special param: direction
    #         if cmd["maintype"] in ['areahunt', 'takeoffareahunt', 'targethunt', 'takeofftargethunt', 'Ground_Set_Direction', 'Ground_Move_Deploy', 'Ship_Move_Deploy']:
    #             # 0-360
    #             cmd2[k] = int(clip_2(cmd[k][MT],0,360))
    #         else:
    #             # 0-180
    #             cmd2[k] = int(clip_2(cmd[k][MT],0,180))
    #
    #         act_include[k][MT] = 1
    #
    #     elif k=="area":
    #         cmd2[k] = EnvCmd._make_area(cmd["area_id"], cmd["area_type"][MT], cmd["point_list"][MT])
    #
    #         assert cmd["area_id"] == 1
    #         # act_include["area_id"][MT] = 1
    #         act_include["area_type"][MT] = 1
    #         act_include["point_list"][MT] = 1
    #
    #     elif k=="disturb_time":
    #         cmd2[k] = cmd["patrol_time"][MT]
    #
    #         act_include["patrol_time"][MT] = 1
    #
    #     elif k=="modle":
    #         cmd2[k] = cmd["mode"][MT]
    #
    #         act_include["mode"][MT] = 1
    #
    #     else:
    #         if k in ["maintype", "self_id", "cov_id", "target_id"]:
    #             cmd2[k] = cmd[k]
    #             act_include[k] = 1
    #         elif k in ["airport_id", "patrol_mode"]:
    #             cmd2[k] = cmd[k]
    #         else:
    #             try:
    #                 cmd2[k] = cmd[k][MT]
    #                 act_include[k][MT] = 1
    #
    #             except Exception as e:
    #                 print(e)
    #                 print(k)
    #                 assert 0
    #
    # if cmd2['maintype']=='Ship_areapatrol':
    #     cmd2['point_z'] = 0  # same effect with clip(, 0, 0)
    #
    # if "speed" in cmd2:
    #     if "fly_type" in cmd2:
    #         cmd2["speed"] = get_speed(action["speed"][MT][0][0],cmd["maintype"],cmd["fly_type"][MT])
    #
    #     cmd2 = deal_speed(cmd2,action["speed"][MT][0][0],obs)
    #
    # return [cmd2], act_include

def get_speed(speed,maintype,fly_type):
    speed_range = {
        11: [100, 300],  # 歼击机速度约为900-1000km/h
        12: [100, 250],  # 预警机速度约为600-800km/h
        13: [100, 250],  # 预警机速度约为600-800km/h
        14: [50, 100],  # 无人机速度约为180-350km/h
        15: [100, 250],  # 轰炸机速度约为600-800km/h
        21: [0, 20],  # 舰船速度约为0-30节(白皮书书写有误), 等价于0-54km/h
        31: [0, 30]  # 地防速度约为0-90km/h(白皮书书写有误)
    }

    if maintype == 'takeoffprotect':
        fly_type = 11  # 起飞护航指令默认起飞歼击机
    elif maintype in ['takeoffareahunt', 'takeofftargethunt']:
        fly_type = 15  # 起飞突击类指令默认起飞轰炸机

    speed = speed*(speed_range[fly_type][1]-speed_range[fly_type][0])*0.5+(speed_range[fly_type][1]+speed_range[fly_type][0])*0.5
    speed = np.clip(speed,speed_range[fly_type][0],speed_range[fly_type][1])
    return speed

def deal_speed(cmd2,origin_speed,obs_own):
    speed_range = {
        11: [100, 300],  # 歼击机速度约为900-1000km/h
        12: [100, 250],  # 预警机速度约为600-800km/h
        13: [100, 250],  # Disturb 机速度约为600-800km/h
        14: [50, 100],  # 无人机速度约为180-350km/h
        15: [100, 250],  # 轰炸机速度约为600-800km/h
        21: [0, 20],  # 舰船速度约为0-30节(白皮书书写有误), 等价于0-54km/h
        31: [0, 30]  # 地防速度约为0-90km/h(白皮书书写有误)
    }

    if "self_id" in cmd2 and (not cmd2["self_id"]==-1) and "speed" in cmd2:

        self_id = cmd2["self_id"]
        # get the LX(fly_type) by self_id
        unit = [u for u in obs_own['units'] if u['ID'] == self_id]
        team = [u for u in obs_own['teams'] if u['TMID'] == self_id]

        obj = unit[0] if len(unit) > 0 else team[0]
        fly_type = obj['LX']

        speed = origin_speed * (speed_range[fly_type][1] - speed_range[fly_type][0]) * 0.5 + (
                    speed_range[fly_type][1] + speed_range[fly_type][0]) * 0.5
        speed = np.clip(speed, speed_range[fly_type][0], speed_range[fly_type][1])
        cmd2["speed"] = speed
    return cmd2

if __name__=="main":
    pass
