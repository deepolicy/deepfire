import tensorflow as tf
import numpy as np
from env.env_cmd import EnvCmd

def south_COMMAND_dead(obs_red):
    observation = obs_red
    COMMAND_units = [qb for qb in observation["qb"] if qb["LX"] == 41]
    south_alive = 0
    for unit in COMMAND_units:
        if unit["Y"] < 0 and unit["WH"] == 1:
            south_alive = 1
            break

    if not south_alive:
        print_2("SOUTH COMMAND DEAD!")

    return (not south_alive)

def LX15_all_die(observation):

    if len([item for item in observation["red"]["units"] if item["LX"]==15]) == 0 and observation["red"]["airports"][0]["BOM"] < 2:
        if len([item for item in observation["blue"]["units"] if item["LX"]==41]) > 0:
            return True

    return False

def get_xy_obs_by_id(self_id, obs):
    for team in obs["teams"]:
        if team["TMID"] == self_id:
            assert len(team["PT"])>0
            unit_id = team["PT"][0][0]

            for unit in obs["units"]:
                if unit["ID"] == unit_id:
                    return unit["X"], unit["Y"]

def print_2(data):
    print("")
    print("="*30)
    print(data)
    print("")

def filter_repeat(_list):
    _list_only = []
    for i in _list:
        if not i in _list_only:
            _list_only.append(i)
    return _list_only

def areahunt_rule(self_id, area_wh_hunt, direction, range, observation, area_direct, area_len, area_wid):
    COMMAND_units = [qb for qb in observation["qb"] if qb["LX"] == 41]

    if area_wh_hunt == 0:
        # y>0, north COMMAND
        target_id = [COMMAND["ID"] for COMMAND in COMMAND_units if COMMAND["Y"]>0]

    elif area_wh_hunt == 1:
        # y<0, south COMMAND
        target_id = [COMMAND["ID"] for COMMAND in COMMAND_units if COMMAND["Y"]<0]

    target_id = filter_repeat(target_id)
    assert len(target_id) <= 1
    if len(target_id) == 0:
        return []
    else:
        target_id = target_id[0]
        for COMMAND in COMMAND_units:
            if COMMAND["ID"] == target_id:
                px = COMMAND["X"]
                py = COMMAND["Y"]
                pz = 7000
                break

    return [EnvCmd.make_areahunt(self_id, direction, range, px, py, pz, *[area_direct, area_len, area_wid])]

def targethunt_rule(self_id, target_wh_hunt, direction, range, observation):
    COMMAND_units = [qb for qb in observation["qb"] if qb["LX"] == 41]
    ship_units = [qb for qb in observation["qb"] if qb["LX"] == 21]

    if target_wh_hunt == 0:
        assert len(ship_units) <= 1
        target_id = [ship["ID"] for ship in ship_units]

    elif target_wh_hunt == 1:
        # y>0, north COMMAND
        target_id = [COMMAND["ID"] for COMMAND in COMMAND_units if COMMAND["Y"]>0]


    elif target_wh_hunt == 2:
        # y<0, south COMMAND
        target_id = [COMMAND["ID"] for COMMAND in COMMAND_units if COMMAND["Y"]<0]

    target_id = filter_repeat(target_id)
    assert len(target_id) <= 1
    if len(target_id) == 0:
        return []
    else:
        target_id = target_id[0]

    return [EnvCmd.make_targethunt(self_id, target_id, direction, range=range)]

def clip_2(x,min_val,max_val):
    assert max_val > min_val
    x = x*(max_val-min_val)*0.5+(max_val+min_val)*0.5
    x = np.clip(x, min_val, max_val)  # repair and add code here at 08-06 09:21
    return x

def scale_to(x,min_val,max_val):
    assert max_val > min_val
    x = x*(max_val-min_val)*0.5+(max_val+min_val)*0.5
    return x

def softmax_argmax(nodes):
    index = tf.argmax(tf.nn.softmax(nodes),axis=-1)

    return index

def embedding(x,m,n):
    return tf.keras.layers.Embedding(m,n)(x)

class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def get_cmd_list():
	return [
            {
                'maintype': 'areapatrol',
                'self_id': None,             # 己方编队编号(整型)
                'point_x': None,                  # 区域中心x轴坐标(float型, 单位: 米)
                'point_y': None,
                'point_z': None,
                'direction': None,         # 区域长轴与正北方向角度(整型, 顺时针方向, 单位: 度, [0, 180])
                'length': None,               # 区域长度(整型, 单位: 米)
                'width': None,                 # 区域宽度(整型, 单位: 米)
                'speed': None,                 # 巡逻速度(浮点型, 单位: 米/秒, 歼击机参考值250m/s, 轰炸机参考值167m/s)
                'patrol_time': None,     # 巡逻时间(整型, 单位: 秒)
                'patrol_mode': None      # 巡逻模式(整型, 0表普通模式, 1表示对角巡逻, 预留接口, 暂无实际效果)
            },

            {
                'maintype': 'takeoffareapatrol',
                'airport_id': None,       # 机场编号(整型)
                'fly_num': None,             # 起飞数量(整型)
                'fly_type': None,           # 起飞战机类型(11-歼击机; 12-预警机; 13-电子干扰机; 14-无人侦察机；15-轰炸机)
                'point_x': None,                  # 区域中心点x轴坐标(浮点型, 单位: 米)
                'point_y': None,
                'point_z': None,
                'direction': None,         # 区域长轴与正北方向角度(整型, 顺时针方向, 单位: 度, [0, 180])
                'length': None,               # 区域长度(整型, 单位: 米)
                'width': None,                 # 区域宽度(整型, 单位: 米)
                'speed': None,                 # 巡逻速度(浮点型, 单位: 米/秒)
                'patrol_time': None      # 巡逻时间(整型, 单位: 秒)
            },

            {
                'maintype': 'linepatrol',
                'self_id': None,             # 己方编队编号(整型)
                'speed': None,                 # 巡逻速度(浮点型, 单位: 米/秒)
                'area': None,
            },

            {
                'maintype': 'takeofflinepatrol',
                'airport_id': None,       # 机场编号(整型)
                'fly_num': None,             # 起飞数量(整型)
                'fly_type': None,           # 起飞战机类型
                'speed': None,                 # 巡逻速度(浮点型, 单位: 米/秒)
                'area': None,
            },

            {
                'maintype': 'areahunt',
                'self_id': None,             # 己方轰炸机编队编号(整型)
                'direction': None,         # 突击方向, 相对正北方向角度(整型, 逆时针方向, 单位: 度, [0, 360])
                'range': None,                 # 武器发射距离与最大射程的百分比(整型, [1, 100], 距离越近命中率越高, 突防难度也更大)
                'point_x': None,                  # 区域中心点x轴坐标(浮点型, 单位: 米)
                'point_y': None,
                'point_z': None,
                'area_direct': None,  # 区域长轴与正北方向角度(整型, 顺时针方向, 单位: 度, [0, 180])
                'area_len': None,        # 区域长度(整型, 单位: 米)
                'area_wid': None          # 区域宽度(整型, 单位: 米)
            },

            {
                'maintype': 'takeoffareahunt',
                'airport_id': None,       # 机场编号(整型)
                'fly_num': None,             # 起飞数量(整型)
                'direction': None,         # 突击方向, 相对正北方向角度(整型, 逆时针方向, 单位: 度, [0, 360])
                'range': None,                 # 武器发射距离与最大射程的百分比(整型, [1, 100], 距离越近命中率越高, 突防难度也更大)
                'point_x': None,                  # 区域中心点x轴坐标(浮点型, 单位: 米)
                'point_y': None,
                'point_z': None,
                'area_direct': None,  # 区域长轴与正北方向角度(整型, 顺时针方向, 单位: 度, [0, 180])
                'area_len': None,        # 区域长度(整型, 单位: 米)
                'area_wid': None,         # 区域宽度(整型, 单位: 米)
                'speed': None                  # 突击速度(浮点型, 单位: 米/秒, 参考速度166m/s)
            },

            {
                'maintype': 'targethunt',
                'self_id': None,             # 己方轰炸机编队编号(整型)
                'target_id': None,         # 敌方平台编号(整型)
                'direction': None,         # 突击方向, 相对正北方向角度(整型, 逆时针方向, 单位: 度, [0, 360])
                'range': None                  # 武器发射距离与最大射程的百分比(整型, [1, 100], 距离越近命中率越高, 突防难度也更大)
            },

            {
                'maintype': 'takeofftargethunt',
                'airport_id': None,       # 机场编号(整型)
                'fly_num': None,             # 起飞数量(整型)
                'target_id': None,         # 敌方平台编号(整型)
                'direction': None,         # 突击方向, 相对正北方向角度(整型, 逆时针方向, 单位: 度, [0, 360])
                'range': None,                 # 武器发射距离与最大射程的百分比(整型, [1, 100], 距离越近命中率越高, 突防难度也更大)
                'speed': None                  # 突击速度(浮点型, 单位: 米/秒, 参考速度166m/s)
            },

            {
                'maintype': 'protect',
                'self_id': None,             # 己方歼击机编队编号(整型)
                'cov_id': None,               # 被护航编队编号(整型, 护航对象类型不能是无人侦察机)
                'flag': None,                   # 护航方式(整型, 1前/2后/3左/4右)
                'offset': None                # 与护航目标间的距离(整型, 单位:百米, [1, 100])
            },

            {
                'maintype': 'takeoffprotect',
                'airport_id': None,       # 机场编号(整型)
                'fly_num': None,             # 起飞数量(整型)
                'cov_id': None,               # 被护航编队编号(整型, 护航对象类型不能是无人侦察机)
                'flag': None,                   # 护航方式(整型, 1前/2后/3左/4右)
                'offset': None,               # 与护航目标间的距离(整型, 单位:百米, [1, 100])
                'speed': None                  # 速度(浮点型, 单位: 米/秒)
            },

            {
                'maintype': 'airattack',
                'self_id': None,             # 己方歼击机平台编号(整型)
                'target_id': None,         # 敌方平台编号(整型)
                'type': None                    # 拦截的引导方法(整型, 0/1)
            },

            {
                'maintype': 'returntobase',
                'self_id': None,             # 己方编队/单个平台编号(整型)
                'airport_id': None        # 己方机场编号(整型)
            },

            {
                'maintype': 'Ground_Add_Target',
                'self_id': None,             # 己方地防编队编号(整型)
                'target_id': None          # 敌方平台编号(整型)
            },

            {
                'maintype': 'Ground_Remove_Target',
                'self_id': None,             # 己方地防编队编号(整型)
                'target_id': None          # 敌方平台编号(整型)
            },

            {
                'maintype': 'GroundRadar_Control',
                'self_id': None,             # 己方地防编队编号(整型)
                'on_off': None                # 开关机(整型, 0: off; 1: on)
            },

            {
                'maintype': 'Ground_Set_Direction',
                'self_id': None,             # 己方地防编队编号(整型)
                'direction': None          # 防御方向, 与正北方向夹角(整型, 逆时针方向, 单位: 度, [0, 360])
            },

            {
                'maintype': 'Ground_Move_Deploy',
                'self_id': None,             # 己方地防编队编号(整型)
                'point_x': None,                  # 目标点x轴坐标(浮点型, 单位: 米)
                'point_y': None,
                'point_z': None,
                'direction': None,         # 防御方向, 与正北方向夹角(整型, 逆时针方向, 单位: 度, [0, 360])
                'radar_state': None      # 雷达开关机状态(整型, 0: off; 1: on)
            },

            {
                'maintype': 'Ship_Add_Target',
                'self_id': None,             # 己方舰船编队编号(整型)
                'target_id': None          # 敌方平台编号(整型)
            },

            {
                'maintype': 'Ship_Remove_Target',
                'self_id': None,             # 己方舰船编队编号(整型)
                'target_id': None          # 敌方平台编号(整型)
            },

            {
                'maintype': 'Ship_Radar_Control',
                'self_id': None,         # 己方舰船编队编号(整型)
                'on_off': None            # 开关机(整型, 0: off; 1: on)
            },

            {
                'maintype': 'Ship_Move_Deploy',
                'self_id': None,             # 己方舰船编队编号(整型)
                'point_x': None,                  # 目标点x轴坐标(浮点型, 单位: 米)
                'point_y': None,
                'point_z': None,
                'direction': None,         # 防御方向, 与正北方向夹角(可给任意值, 因为护卫舰是360度防空的)
                'radar_state': None      # 雷达开关机状态(默认为1)
            },

            {
                'maintype': 'Ship_areapatrol',
                'self_id': None,             # 己方舰船编队编号(整型)
                'point_x': None,                  # 区域中心点x轴坐标(浮点型, 单位: 米)
                'point_y': None,
                'point_z': None,
                'direction': None,         # 区域长轴与正北方向角度(整型, 顺时针方向, 单位: 度, [0, 180])
                'length': None,               # 区域长度(整型, 单位: 米)
                'width': None,                 # 区域宽度(整型, 单位: 米)
                'speed': None,                 # 巡逻速度(浮点型, 单位: 米/秒, 参考值8m/s)
                'patrol_time': None,     # 巡逻时间(整型, 单位: 秒)
                'patrol_mode': None      # 巡逻模式(整型, 0表普通模式, 1表示对角巡逻, 预留接口, 暂无实际效果)
            },

            {
                'maintype': 'awcs_areapatrol',
                'self_id': None,             # 己方预警机编队编号(整型)
                'point_x': None,                  # 区域中心点x轴坐标(浮点型, 单位: 米)
                'point_y': None,
                'point_z': None,
                'direction': None,         # 区域长轴与正北方向角度(整型, 顺时针方向, 单位: 度, [0, 180])
                'length': None,               # 区域长度(整型, 单位: 米)
                'width': None,                 # 区域宽度(整型, 单位: 米)
                'speed': None,                 # 巡逻速度(浮点型, 单位: 米/秒, 参考值166m/s)
                'patrol_time': None,     # 巡逻时间(整型, 单位: 秒)
                'patrol_mode': None      # 巡逻模式(整型, 0表普通模式, 1表示对角巡逻, 预留接口, 暂无实际效果)
            },

            {
                'maintype': 'awcs_linepatrol',
                'self_id': None,             # 己方预警机编队编号(整型)
                'speed': None,                 # 巡逻速度(浮点型, 单位: 米/秒)
                'area': None,
            },

            {
                'maintype': 'awcs_mode',
                'self_id': None,             # 己方预警机编队编号(整型)
                'modle': None                   # 探测模式(整型, 0-对空; 1-对海; 2-空海交替)
            },

            {
                'maintype': 'awcs_radarcontrol',
                'self_id': None,             # 己方预警机编队编号(整型)
                'on_off': None                # 开关机(整型, 0: off; 1: on)
            },

            {
                'maintype': 'awcs_cancledetect',
                'self_id': None,             # 己方预警机编队编号(整型)
            },

            {
                'maintype': 'area_disturb_patrol',
                'self_id': None,             # 己方干扰机编队编号(整型)
                'point_x': None,                  # 区域中心点x轴坐标(浮点型, 单位: 米)
                'point_y': None,
                'point_z': None,
                'direction': None,         # 区域长轴与正北方向角度(整型, 顺时针方向, 单位: 度, [0, 180])
                'length': None,               # 区域长度(整型, 单位: 米)
                'width': None,                 # 区域宽度(整型, 单位: 米)
                'speed': None,                 # 巡逻速度(浮点型, 单位: 米/秒, 参考值166m/s)
                'disturb_time': None    # 干扰持续时间(整型，单位: 秒)
            },

            {
                'maintype': 'line_disturb_patrol',
                'self_id': None,             # 己方干扰机编队编号(整型)
                'speed': None,
                'area': None,
            },

            {
                'maintype': 'set_disturb',
                'self_id': None,             # 己方干扰机编队编号(整型)
                'mode': None                    # 干扰模式(整型, 0阻塞干扰，1瞄准干扰-瞄准干扰暂无效果)
            },

            {
                'maintype': 'close_disturb',
                'self_id': None,             # 己方干扰机编队编号(整型)
            },

            {
                'maintype': 'stop_disturb',
                'self_id': None,             # 己方干扰机编队编号(整型)
            },

            {
                'maintype': 'uav_areapatrol',
                'self_id': None,             # 己方无人侦察机编队编号(整型)
                'point_x': None,                  # 区域中心点x轴坐标(浮点型, 单位: 米)
                'point_y': None,
                'point_z': None,
                'direction': None,         # 区域长轴与正北方向角度(整型, 顺时针方向, 单位: 度, [0, 180])
                'length': None,               # 区域长度(整型, 单位: 米)
                'width': None,                 # 区域宽度(整型, 单位: 米)
                'speed': None,                 # 巡逻速度(浮点型, 单位: 米/秒, 参考值166m/s)
                'patrol_time': None,     # 巡逻时间(整型，单位: 秒)
                'patrol_mode': None      # 巡逻模式(整型, 0表普通模式, 1表示对角巡逻, 预留接口, 暂无实际效果)
            },

            {
                'maintype': 'uav_linepatrol',
                'self_id': None,             # 己方无人机编队编号(整型)
                'speed': None,
                'area': None,
            },

            {
                'maintype': 'uav_cancledetect',
                'self_id': None,             # 己方无人机编队编号(整型)
            },

            {
                'maintype': 'base_radarcontrol',
                'self_id': None,             # 己方地面雷达编队编号(整型)
                'on_off': None                # 开关机(整型, 0: off; 1: on)
            },
        ]

def repair_unit_TMID(obs):
    for i in range(len(obs["units"])):
        if obs["units"][i]["TMID"]==0:
            # detect
            print("")
            print("TMID of unit is 0, it will be repaired here ...")
            print(obs["units"][i])
            print("")

            # look up
            unit_id = obs["units"][i]["ID"]
            TMID = None
            for team in obs["teams"]:
                if len(team["PT"])>0:
                    for i_PT in team["PT"]:
                        if i_PT[0]==unit_id:
                            TMID = team["TMID"]
                if not TMID==None:
                    break
            assert not TMID==None

            # repair
            obs["units"][i]["TMID"]=TMID
    return obs