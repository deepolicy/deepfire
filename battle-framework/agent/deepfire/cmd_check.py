import time
import random
import os
import subprocess

class Cmd_validate(object):
    def __init__(self):
        pass


    def _action_validate(self, actions, obs_own, side, sim_time):
        """指令有效性基本检查"""
        # 检查项: 执行主体, 目标, 护航对象, 机场相关指令是否有效,
        # 同时检查速度设置是否越界
        # 检查项: 地防/护卫舰的初始部署位置只能在己方半场, 且只允许在开局2分钟内进行调整;
        for action in actions:
            maintype = action['maintype']
            if maintype in ['Ship_Move_Deploy', 'Ground_Move_Deploy']:
                pos2d = (action['point_x'], action['point_y'])
                self._validate_deploy(side, pos2d, sim_time)
            if 'self_id' in action:
                speed = int(action['speed']) if 'speed' in action else None
                self_id = int(action['self_id'])
                self._validate_self_id(maintype, self_id, speed, obs_own)
            if 'target_id' in action:
                target_id = int(action['target_id'])
                self._validate_target_id(target_id, obs_own)
            if 'cov_id' in action:
                cov_id = int(action['cov_id'])
                self._validate_cov_id(cov_id, obs_own)
            if 'airport_id' in action:
                airport_id = int(action['airport_id'])
                speed = int(action['speed']) if 'speed' in action else None
                self._validate_airport(airport_id, speed, action, obs_own)

        return "allowed"

    @staticmethod
    def _validate_deploy(side, pos2d, sim_time):
        assert sim_time <= 120
        if side == 'red':
            assert pos2d[0] >= 0
        else:
            assert pos2d[0] <= 0

    def _validate_self_id(self, maintype, self_id, speed, obs_own):
        """判断执行主体是否有效"""
        # 空中拦截指令(make_airattack)执行主体只能是单平台(需要选手考虑目标分配问题)
        # 返航指令(make_returntobase)执行主体可以是单平台也可以是编队(考虑编队内单机油量不足提前返航)
        # 其他指令执行主体原则上必须是编队, 为防止出错目前服务端也支持给单平台下指令.
        unit = [u for u in obs_own['units'] if u['ID'] == self_id]
        team = [u for u in obs_own['teams'] if u['TMID'] == self_id]
        if maintype == 'airattack':
            if len(unit) == 0:
                raise Exception("无效平台编号%s" % self_id)
        else:
            if len(unit) == 0 and len(team) == 0:
                raise Exception("无效执行主体编号%s" % self_id)
            else:
                obj = unit[0] if len(unit) > 0 else team[0]
                if obj['LX'] not in type4cmd[maintype]:
                    raise Exception("类型为%s的平台或者编队%s无法执行%s指令" % (obj['LX'], self_id, maintype))
                # 检查速度设置是否越界
                if speed is not None:
                    self._validate_speed(obj['LX'], speed)

    @staticmethod
    def _validate_target_id(target_id, obs_own):
        """判断目标编号是否合法"""
        # 所有指令的目标编号, 必须是敌方单平台号
        unit = [u for u in obs_own['qb'] if u['ID'] == target_id]
        if len(unit) == 0:
            raise Exception("无效目标平台编号%s" % target_id)

    @staticmethod
    def _validate_cov_id(cov_id, obs_own):
        """护航对象是否有效"""
        # 护航对象编号, 必须为己方编队号
        team = [u for u in obs_own['teams'] if u['TMID'] == cov_id]
        if len(team) == 0:
            raise Exception("无效护航对象编号%s" % cov_id)
        else:
            type4cov = [12, 13, 15]
            if int(team[0]['LX']) not in type4cov:
                raise Exception("非法护航目标类型%s" % team[0]['LX'])

    def _validate_airport(self, airport_id, speed, action, obs_own):
        """判断机场相关指令的有效性"""
        airports = obs_own['airports']
        airport = [u for u in airports if u['ID'] == airport_id]
        # 根据机场情况判断指令合法性
        maintype = action['maintype']
        if len(airport) == 0:
            raise Exception("无效机场编号%s" % airport_id)
        elif maintype == 'returntobase':
            pass
        else:
            obj = airport[0]
            if not obj['WH']:
                raise Exception("机场%s修复中无法执行起飞指令" % obj['ID'])
            if maintype == 'takeoffprotect':
                fly_type = 11   # 起飞护航指令默认起飞歼击机
            elif maintype in ['takeoffareahunt', 'takeofftargethunt']:
                fly_type = 15   # 起飞突击类指令默认起飞轰炸机
            else:
                fly_type = action['fly_type']
            fly_num = action['fly_num']
            if int(fly_type) not in type4cmd[maintype]:
                raise Exception("机场无法起降类型为%s的单位" % fly_type)
            type_map = {11: 'AIR', 12: 'AWCS', 13: 'JAM', 14: 'UAV', 15: 'BOM'}
            attr = type_map[fly_type]
            if fly_num > obj[attr]:
                print('指7645736346令>>>', action)
                print("")
                print(obj)
                print("")
                raise Exception("起飞数量%d大于机场可起飞数量%d" % (fly_num, obj[attr]))
            # 检查速度设置是否越界
            if speed is not None:
                self._validate_speed(fly_type, speed)

    @staticmethod
    def _validate_speed(unit_type, speed):
        """判断速度设置是否越界(单位: m/s)，范围适当放宽"""
        speed_range = {
            11: [100, 300],     # 歼击机速度约为900-1000km/h
            12: [100, 250],     # 预警机速度约为600-800km/h
            13: [100, 250],     # 预警机速度约为600-800km/h
            14: [50, 100],      # 无人机速度约为180-350km/h
            15: [100, 250],     # 轰炸机速度约为600-800km/h
            21: [0, 20],        # 舰船速度约为0-30节(白皮书书写有误), 等价于0-54km/h
            31: [0, 30]         # 地防速度约为0-90km/h(白皮书书写有误)
        }
        sp_limit = speed_range[unit_type]

        # assert sp_limit[0] <= speed <= sp_limit[1]

        if not sp_limit[0] <= speed <= sp_limit[1]:
            raise Exception("sp_limit:", sp_limit[0], speed, sp_limit[1])

# 不同指令可执行主体的类型列表
type4cmd = {
    # 作战飞机
    "areapatrol": [11, 12, 13, 14, 15],
    "takeoffareapatrol": [11, 12, 13, 14, 15],
    "linepatrol": [11, 12, 13, 14, 15],
    "takeofflinepatrol": [11, 12, 13, 14, 15],
    "areahunt": [15],
    "takeoffareahunt": [15],
    "targethunt": [15],
    "takeofftargethunt": [15],
    "protect": [11],
    "takeoffprotect": [11],
    "airattack": [11],
    "returntobase": [11, 12, 13, 14, 15],
    # 地防
    "Ground_Add_Target": [31],
    "Ground_Remove_Target": [31],
    "GroundRadar_Control": [31],
    "Ground_Set_Direction": [31],
    "Ground_Move_Deploy": [31],
    # 护卫舰
    "Ship_Move_Deploy": [21],
    "Ship_areapatrol": [21],
    "Ship_Add_Target": [21],
    "Ship_Remove_Target": [21],
    "Ship_Radar_Control": [21],
    # 预警机
    "awcs_areapatrol": [12],
    "awcs_linepatrol": [12],
    "awcs_mode": [12],
    "awcs_radarcontrol": [12],
    "awcs_cancledetect": [12],
    # 干扰机
    "area_disturb_patrol": [13],
    "line_disturb_patrol": [13],
    "set_disturb": [13],
    "close_disturb": [13],
    "stop_disturb": [13],
    # 无人侦察机
    "uav_areapatrol": [14],
    "uav_linepatrol": [14],
    "uav_cancledetect": [14],
    # 地面雷达
    "base_radarcontrol": [32]
}
