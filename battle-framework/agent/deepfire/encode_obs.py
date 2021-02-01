import tensorflow as tf
import numpy as np

from .utils import softmax_argmax, embedding


def value_nn(nodes):

    return tf.layers.dense( nodes ,1)

def lstm_nn(obs_concat,batch_size=1):
    with tf.variable_scope("lstm"):

        obs_concat_fc = tf.layers.dense( tf.expand_dims(obs_concat,1) ,640,activation=tf.nn.relu)

        lstm_unit = 2048
        lstm = tf.nn.rnn_cell.LSTMCell(num_units=lstm_unit)

        init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=obs_concat_fc, initial_state=init_state)

        cell_out = tf.reshape(outputs, [-1, lstm_unit], name='flatten_lstm_outputs')

        act_fc = tf.layers.dense(cell_out,64+260,activation=None)

    return act_fc, init_state, final_state

def obs_nn2(airport_ly1, teams_ly1, units_ly1, qb_ly1, rockets_ly1,units_id,teams_pt,teams_mask, units_mask, qb_mask, rockets_mask):
    '''
        using conv1d, maxpooling here.
    '''

    with tf.variable_scope("fc-globalmaxpool"):

        airport_fc1 = tf.layers.dense(airport_ly1,64,activation=tf.nn.relu)

        airport_fc2 = tf.layers.dense(airport_fc1,64,activation=tf.nn.relu)

        with tf.variable_scope("units"):
            units_fc1 = {}
            units_fc2 = {}
            units_key = {}
            for LX in "11, 12, 13, 14, 15, 21, 32".split(", "):

                units_fc1[LX] = tf.nn.relu(tf.layers.conv1d(units_ly1[LX],64,1))

                units_fc2[LX] = tf.nn.relu(tf.layers.conv1d(units_fc1[LX],64,1))

                units_key[LX] = tf.slice(units_fc2[LX],[0,0,0],[-1,-1,16])

        with tf.variable_scope("teams"):
            teams_fc1 = {}
            teams_fc2 = {}
            teams_key = {}
            teams_gmp = {}
            for LX in "11, 12, 13, 14, 15, 21, 32".split(", "):

                print(teams_ly1[LX])
                # Tensor("concat_7:0", shape=(?, ?, length), dtype=float32)

                teams_fc1[LX] = tf.nn.relu(tf.layers.conv1d(teams_ly1[LX],64,1))
                # should be (?,?,64)

                with tf.variable_scope("children_units"):
                    mask = tf.matmul(teams_pt[LX], tf.transpose(units_id[LX], [0, 2, 1]))
                    mask = tf.expand_dims(mask, -1)
                    children_units = tf.reduce_max(
                        tf.multiply(mask, tf.tile(tf.expand_dims(units_fc2[LX], 1), [1, tf.shape(mask)[1], 1, 1])),
                        axis=2)
                    print("children_units:",children_units)
                    # Tensor("Max_15:0", shape=(?, ?, 64), dtype=float32)
                ink_out = [mask, tf.tile(tf.expand_dims(units_fc2[LX], 1), [1, tf.shape(mask)[0], 1, 1]),units_fc2[LX]]
                # (4, 2, 2, 1)
                # (4, 4, 2, 64)
                # (4, 2, 64)

                print("children_units")
                print(children_units)
                # Tensor("fc-globalmaxpool/teams/global_max_pooling1d_2/Max:0", shape=(?, 64), dtype=float32)
                print(teams_fc1[LX])
                # Tensor("fc-globalmaxpool/teams/Relu_2:0", shape=(?, ?, 64), dtype=float32)

                teams_fc2[LX] = tf.nn.relu(tf.layers.conv1d( tf.concat( [teams_fc1[LX],children_units], axis=2) ,64,1))  # or concat the latent of units here

                teams_key[LX] = tf.slice(teams_fc2[LX],[0,0,0],[-1,-1,16]) # or without relu
                # print(teams_key[LX])
                # Tensor("fc-globalmaxpool/teams/Slice:0", shape=(?, 1, 16), dtype=float32)

                print("")

                teams_gmp[LX] = tf.keras.layers.GlobalMaxPooling1D()(tf.multiply(teams_fc2[LX],teams_mask[LX]))
        '''
                with tf.variable_scope("units"):
            units_fc1 = {}
            units_fc2 = {}
            units_key = {}
            for LX in "11, 12, 13, 14, 15, 21, 32".split(", "):

                units_fc1[LX] = tf.nn.relu(tf.layers.conv1d(units_ly1[LX],64,1))

                units_fc2[LX] = tf.nn.relu(tf.layers.conv1d(units_fc1[LX],64,1))

                units_key[LX] = tf.slice(units_fc2[LX],[0,0,0],[-1,-1,16])
        '''
        with tf.variable_scope("qb"):
            qb_fc1 = {}
            qb_fc2 = {}
            qb_key = {}
            qb_gmp = {}
            for LX in [11, 12, 15, 21, 18, 28, 41, 42, 31, 32]:
                LX = str(LX)
                qb_fc1[LX] = tf.nn.relu(tf.layers.conv1d(qb_ly1[LX],64,1))

                qb_fc2[LX] = tf.nn.relu(tf.layers.conv1d(qb_fc1[LX],64,1))

                qb_key[LX] = tf.slice(qb_fc2[LX],[0,0,0],[-1,-1,16])

                qb_gmp[LX] = tf.keras.layers.GlobalMaxPooling1D()(tf.multiply(qb_fc2[LX],qb_mask[LX]))

        with tf.variable_scope("rockets"):

            rockets_fc1 = tf.nn.relu(tf.layers.conv1d(rockets_ly1,64,1))

            rockets_fc2 = tf.nn.relu(tf.layers.conv1d(rockets_fc1,64,1))

            rockets_gmp = tf.keras.layers.GlobalMaxPooling1D()(tf.multiply(rockets_fc2,rockets_mask))

    with tf.variable_scope("maxpool-concat"):

        obs_concat = tf.concat([airport_fc2]+ [teams_gmp[k] for k in teams_gmp] + [qb_gmp[k] for k in qb_gmp] + [rockets_gmp],axis=1)

    return obs_concat, teams_key, units_key, qb_key

def obs_nn(units_num):
    '''
        build place_holder network
    '''


    '''
        placeholder layer
        the last one in teams_ph, units_ph, qb_ph, rockets_ph is mask for filter invalid padding data.
    '''
    airport_ph = tf.placeholder(tf.float32, shape=(None,13)) # both red and blue has only one airport.
    teams_ph = {LX: tf.placeholder(tf.float32, shape=(None,None,2+ units_num[LX] +1)) for LX in "11, 12, 13, 14, 15, 21, 32".split(", ")}
    units_ph = {LX: tf.placeholder(tf.float32, shape=(None,None,14+ units_num[LX] +1)) for LX in "11, 12, 13, 14, 15, 21, 32".split(", ")}
    qb_ph = {str(LX): tf.placeholder(tf.float32, shape=(None,None,8+1)) for LX in [11, 12, 15, 21, 18, 28, 41, 42, 31, 32]}
    rockets_ph = tf.placeholder(tf.float32, shape=(None,None,8+1))

    '''
        1st layer
        
        airport
    '''
    airport_ly1 = airport_ph # without any operation.

    '''
        units
        
        red has not COMMAND, so all the units have same data structure.

        TMID, SBID, do not include them in input.

        X
        Y
        Z
        HX
        SP
        LX embedding 13 -> 3 (unlist 71 for airport now, but it should be 42)
        TM
        Locked 0/1
        WH 0/1
        DA
        Hang
        Fuel
        ST embedding 26 -> 4
        WP_name [360,170,519] IS THERE ANY MORE? 3 -> 2 OR 3 -> 1? (include 0 for empty data, so 4->2)
        WP_num

    '''
    units_ly1 = {}
    units_mask = {}
    units_id = {}  # with format of one_hot, keep the id of unit, to match with team_pt.
    for LX in "11, 12, 13, 14, 15, 21, 32".split(", "):
        units_ly1[LX] = tf.concat([
            tf.slice(units_ph[LX],[0,0,0],[-1,-1,12]),  # rms
            embedding(units_ph[LX][:,:,12],4,2),  # WP0
            embedding(units_ph[LX][:,:,13],26,4),  # ST
            tf.slice(units_ph[LX],[0,0,14],[-1,-1,-1]),  # mask
            ], axis=2 )









            # [ tf.slice(units_ph[LX],[0,0,0],[-1,-1,5]),embedding(units_ph[LX][:,:,5],13,3),tf.slice(units_ph[LX],[0,0,6],[-1,-1,1]),tf.slice(units_ph[LX],[0,0,7],[-1,-1,1]),tf.slice(units_ph[LX],[0,0,8],[-1,-1,1]),tf.slice(units_ph[LX],[0,0,9],[-1,-1,3]),embedding(units_ph[LX][:,:,12],26,4),embedding(units_ph[LX][:,:,13],4,2),tf.slice(units_ph[LX],[0,0,14],[-1,-1,1+units_num[LX]]) ], axis=2 )
        units_id[LX] = tf.slice(units_ph[LX],[0, 0, 14], [-1, -1, units_num[LX]])
        # units_mask[LX] = tf.slice(units_ph[LX],[0,0,14+units_num[LX]],[-1,-1,1])
        units_mask[LX] = units_ph[LX][:,:,-1:]


    '''
        teams
        
        ST embedding 44 -> 5
        LX embedding 13 -> 3 (unlist 71 for airport now, but it should be 42)
        Num Int
    '''
    teams_ly1 = {}
    teams_mask = {}
    teams_pt = {}
    for LX in "11, 12, 13, 14, 15, 21, 32".split(", "):
        teams_ly1[LX] = tf.concat([
            teams_ph[LX][:,:,:1],  # NOTICE HERE :1 rather than 0
            embedding(teams_ph[LX][:,:,1],44,5),
            tf.slice(teams_ph[LX],[0,0,2],[-1,-1,-1]) ], axis=2 )




        teams_pt[LX] = tf.slice(teams_ph[LX], [0, 0, 2], [-1, -1, units_num[LX]])
        teams_mask[LX] = teams_ph[LX][:,:,-1:]

    '''
        qb
        X, Y, Z, HX, SP, WH, DA, TM
    '''
    qb_ly1 = {}
    qb_mask = {}
    for LX in [11, 12, 15, 21, 18, 28, 41, 42, 31, 32]:
        LX = str(LX)
        qb_ly1[LX] = qb_ph[LX]
        qb_mask[LX] = qb_ph[LX][:,:,-1:]

    '''
        rockets
        
        X
        Y
        Z
        FY
        HG
        HX
        WH
        TM
    '''
    rockets_ly1 = rockets_ph
    rockets_mask = rockets_ph[:,:,-1:]

    return airport_ph, teams_ph, units_ph, qb_ph, rockets_ph, airport_ly1, teams_ly1, units_ly1, qb_ly1, rockets_ly1,units_id,teams_pt,teams_mask, units_mask, qb_mask, rockets_mask

if __name__ == '__main__':
    pass