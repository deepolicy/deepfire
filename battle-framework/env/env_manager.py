import os
import subprocess
import time
import socket


class EnvManager(object):
    def __init__(self, env_id, base_port, scene_name, prefix,
                 docker_hostname='trsuser5', image_name='combatmodserver:1.3.0.1'):
        """基本设置"""
        # base_port为基准端口号, 由此生成映射到容器内部各端口的宿主机端口号
        # scene_name为想定文件的绝对路径;
        # prefix为容器管理脚本manage_client(对战框架的一部分)所在路径
        # image_name为docker镜像名称
        self.env_id = env_id
        self.base_port = base_port
        self.scene_name = scene_name
        self.prefix = prefix
        self.docker_hostname = docker_hostname
        self.image_name = image_name
        self.ip = '127.0.0.1'

        self.docker_name = 'env_{}'.format(self.env_id)
        self.ports = [self.base_port + self.env_id * 10 + i for i in range(4)]

    def start_docker(self, volume_list=[]):
        """容器的启动"""
        # volume_list为宿主机目录与容器内部目录的映射列表，默认为空列表[], 用于挂载更新后的脚本和新想定等
        volume_map_str = ''
        for volume_map in volume_list:
            volume_map_str += '-v {}:{} '.format(volume_map[0], volume_map[1])
        docker_run = 'docker run -itd --hostname {} --name {} {} -p {}:3641 -p {}:5454 -p {}:5455 -p {}:5901 {}'.format(
            self.docker_hostname, self.docker_name, volume_map_str,
            self.ports[0], self.ports[1], self.ports[2], self.ports[3], self.image_name)
        print(docker_run)
        os.system(docker_run)
        time.sleep(2)

        self.open()
        self.start()

        return True

    def stop_docker(self):
        """容器的停止和删除"""
        docker_stop = 'docker stop {}'.format(self.docker_name)
        print(docker_stop)
        os.system(docker_stop)
        time.sleep(1)

        docker_rm = 'docker rm {}'.format(self.docker_name)
        print(docker_rm)
        os.system(docker_rm)
        time.sleep(1)

    def open(self):
        # open_cmd = '{}/manage_client -host {} -port {} -exercise {} -script mainserv open'.format(
        #     self.prefix, self.ip, self.ports[0], self.scene_name)
        # # EnvManager._exec_command(open_cmd)
        # while os.system(open_cmd) != 0:
        #     time.sleep(1)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.ip, self.ports[0]))
        sock.send(bytes("OPEN*" + self.scene_name + "*" + "mainserv" + "\n", encoding="utf-8"))
        while True:
            try:
                msg = sock.recv(1024).decode("utf-8")
                if "SUCCESS" in msg:
                    break
            except Exception as e:
                print("仿真平台初始化中>>>", e)
                sock.close()
                time.sleep(3)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.ip, self.ports[0]))
                sock.send(bytes("OPEN*" + self.scene_name + "*" + "mainserv" + "\n", encoding="utf-8"))
        sock.close()

    def start(self):
        # start_cmd = '{}/manage_client -host {} -port {} start'.format(self.prefix, self.ip, self.ports[0])
        # EnvManager._exec_command(start_cmd)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.ip, self.ports[0]))
        sock.send(bytes("START\n", encoding="utf-8"))
        msg = sock.recv(1024).decode("utf-8")
        if msg != "SUCCESS\n":
            print(msg)
        sock.close()

    def pause(self):
        # cmd = '{}/manage_client -host {} -port {} pause'.format(self.prefix, self.ip, self.ports[0])
        # EnvManager._exec_command(cmd)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.ip, self.ports[0]))
        sock.send(bytes("PAUSE\n", encoding="utf-8"))
        msg = sock.recv(1024).decode("utf-8")
        if msg != "SUCCESS\n":
            print(msg)
        sock.close()

    def resume(self):
        # cmd = '{}/manage_client -host {} -port {} resume'.format(self.prefix, self.ip, self.ports[0])
        # EnvManager._exec_command(cmd)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.ip, self.ports[0]))
        sock.send(bytes("RESUME\n", encoding="utf-8"))
        msg = sock.recv(1024).decode("utf-8")
        if msg != "SUCCESS\n":
            print(msg)
        sock.close()

    def stop(self):
        # cmd = '{}/manage_client -host {} -port {} stop'.format(self.prefix, self.ip, self.ports[0])
        # EnvManager._exec_command(cmd)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.ip, self.ports[0]))
        sock.send(bytes("STOP\n", encoding="utf-8"))
        msg = sock.recv(1024).decode("utf-8")
        if msg != "SUCCESS\n":
            print(msg)
        sock.close()

    def close(self):
        # cmd = '{}/manage_client -host {} -port {} close'.format(self.prefix, self.ip, self.ports[0])
        # EnvManager._exec_command(cmd)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.ip, self.ports[0]))
        sock.send(bytes("CLOSE\n", encoding="utf-8"))
        msg = sock.recv(1024).decode("utf-8")
        if msg != "SUCCESS\n":
            print(msg)
        sock.close()

    def reset(self):
        """容器化仿真环境重置"""
        if self.image_name == 'sim_fast:1.0':   # 极速版重启方法
            os.popen('docker restart {}'.format(self.docker_name))
            print('docker restart {}'.format(self.docker_name))
            time.sleep(25)
            self.open()
            time.sleep(5)
            self.start()
        else:   # 普通版重启方法
            self.stop()
            time.sleep(10)
            self.close()
            self.open()
            self.start()

    @staticmethod
    def _exec_command(cmd):
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

    def get_server_port(self):
        """返回用于建立rpyc连接的宿主机端口号(对应容器内部的5455端口)"""
        return self.ports[2]

    def get_data_port(self):
        """返回记录数据(也是态势显示)的宿主机端口号(对应容器内部的5454端口)"""
        return self.ports[1]
