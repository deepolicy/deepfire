import argparse
import threading
import time
import json

from env.env_client import EnvClient
from env.env_cmd import EnvCmd
from env.env_runner import EnvRunner

from agent.red_rule_agent import RedRuleAgent
from agent.blue_rule_agent import BlueRuleAgent
from record.attach2docker import net_todocker_init


def connect_loop(rpyc_port):
    """根据映射出来的宿主机端口号rpyc_port，与容器内的仿真平台建立rpyc连接"""
    while True:
        try:
            env_client = EnvClient('127.0.0.1', rpyc_port)
            observation = env_client.get_observation()
            if len(observation['red']['units']) != 0:
                return env_client

        except Exception as e:
            print(e)
            print("rpyc connect failed")

        time.sleep(3)


def print_info(units):
    for unit in units:
        if unit['LX'] != 41:
            print('id: {:3d}, tmid: {:4d}, speed: {:3.0f}, x: {:6.0f}, y: {:6.0f}, z: {:5.0f}, '
                  'type: {:3d}, state: {:3d}, alive: {:2d}, hang: {:3.0f}'.format
                  (unit['ID'], unit['TMID'], unit['SP'], unit['X'], unit['Y'], unit['Z'],
                   unit['LX'], unit['ST'], unit['WH'], unit['Hang']))


class WarRunner(EnvRunner):
    def __init__(self, env_id, server_port, agents, config, replay):
        EnvRunner.__init__(self, env_id, server_port, agents, config, replay)   # 仿真环境初始化

    def _run_env(self):
        time.sleep(1)

    def run(self, num_episodes, speed):
        """对战调度程序"""
        # 启动仿真环境, 与服务端建立rpyc连接
        self._start_env()
        self.env_client = connect_loop(self.env_manager.get_server_port())

        self.env_client.take_action([EnvCmd.make_simulation("SPEED", "", speed)])

        f = open("state.json", "w")
        battle_results = [0, 0, 0]  # [红方获胜局数, 平局数量, 蓝方获胜局数]
        for i in range(num_episodes):
            num_frames = 0
            sim_time = 0  # 记录容器或者程序崩溃时的仿真时间
            self._reset()
            self._run_env()

            # 开启记录本轮数据的两个线程
            if self.save_replay:
                data_port = self.env_manager.get_data_port()
                folder = self.replay_dir
                net_todocker_init('127.0.0.1', data_port, self.agents[0].name, self.agents[1].name, i, folder)

            while True:
                try:
                    num_frames += 1
                    observation = self._get_observation()   # 获取态势
                    sim_time = observation['sim_time']
                    print(i + 1, sim_time)
                    print_info(observation['red']['units'])
                    print()

                    done = self._get_done(observation)     # 推演结束(分出胜负或达到最大时长)
                    if len(observation['red']['rockets']) > 0:
                        # 写入所得的json会在同一行里, 打开文件后按Ctrl+Alt+L可自动转换成字典格式
                        f.write(json.dumps(observation, ensure_ascii=False))

                    self._run_agents(observation)           # 发送指令

                    if done[0]:  # 对战结束后环境重置
                        # 统计胜负结果
                        if done[1] == 0 or done[2] == 0:
                            battle_results[1] += 1
                        if done[1] == 1:
                            battle_results[0] += 1
                        if done[2] == 1:
                            battle_results[2] += 1
                        # 环境重置
                        self.env_manager.reset()
                        self.env_client = connect_loop(self.env_manager.get_server_port())
                        self.env_client.take_action([EnvCmd.make_simulation("SPEED", "", speed)])
                        break
                    self._run_env()
                except Exception as e:
                    print(e)
                    print("容器运行出现异常需要重启")
                    # 记录错误信息
                    self.logger.debug("第{}局程序运行{}秒后出错, 错误信息为: {}\n".format(i + 1, sim_time, e))
                    self._start_env()
                    self.env_client = connect_loop(self.env_manager.get_server_port())
                    self.env_client.take_action([EnvCmd.make_simulation("SPEED", "", speed)])
                    break
        # 关闭文件
        f.close()
        return battle_results


config = {
    'server_port': 6100,
    'config': {
        'scene_name': '/home/Joint_Operation_scenario.ntedt',     # 容器里面想定文件绝对路径
        'prefix': './',                          # 容器管理的脚本manage_client所在路径(这里给的相对路径)
        'image_name': 'combatmodserver:v1.3',    # 镜像名
        'volume_list': [],
        'max_game_len': 350             # 最大决策次数
    },
    'agents': {
        'red_name': {                   # 战队名
            'class': RedRuleAgent,      # 智能体类名
            'side': 'red'               # 智能体所属军别(不可更改!)
        },
        'blue_name': {                  # 战队名
            'class': BlueRuleAgent,     # 智能体类名
            'side': 'blue'              # 智能体所属军别(不可更改!)
        }
    },
    'replay': {                         # 记录回放相关设置
        'save_replay': False,           # 是否记录
        'replay_dir': './replays'       # 回放保存路径
    }
}


def main(env_id, num_episodes, speed):
    """根据环境编号env_id、对战轮数num_episodes和配置config, 构建并运行仿真环境"""
    dg_runner = WarRunner(env_id, **config)
    results = dg_runner.run(num_episodes, speed)
    print('battle results: {}'.format(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=int, required=False, default=1, help='id of environment')
    parser.add_argument('--speed', type=int, required=False, default=0, help='simulation speed')
    parser.add_argument('--num_episode', type=int, required=False, default=100, help='num episodes per env')

    args = vars(parser.parse_args())

    env_id = args['env_id']
    speed = args['speed']
    num_episodes = args['num_episode']

    main(env_id, num_episodes, speed)
