#######################################################################################################################
""" docker的客户端---接收仿真平台的数据、向仿真平台发送指令 """
#######################################################################################################################

import socket
import ctypes
import inspect
import time
from threading import Thread
from .datamanage import ReceiveDataManage
import os
import datetime

close_dic = {}


#########################################################################################################
# docker的客户端
class DockerClient:
    def __init__(self, docker_ip, docker_port):
        self.__docker_ip = docker_ip
        self.__docker_port = docker_port
        self.__docker_socket = None
        while True:
            try:
                self.__docker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                print("[DockerClient]尝试连接网络 __docker_ip:{} __docker_port:{}".
                      format(self.__docker_ip, self.__docker_port))
                time.sleep(1)
                self.__docker_socket.connect((self.__docker_ip, self.__docker_port))
                print("[DockerClient] 网络连接成功")
                break
            except Exception as e:
                self.__docker_socket.close()
                print("[DockerClient] connect except: %s" % e)

    # 保存从网络接收到的数据
    @staticmethod
    def read_command(chunk):
        ReceiveDataManage.save_data(chunk)

    # 通过网络发送数据，无用，目前socket只接受数据，不向服务端发送数据。
    # 接受的数据用于记录文件。并且缓存，通过自己的socket服务端发给各个客户端。
    def send_command(self, send_data_str):
        s = send_data_str + "\n"
        try:
            self.__docker_socket.send(s.encode("utf-8"))
        except Exception as e:
            print("[DockerClient] send except: %s" % e)

    # 更新接收数据
    def update(self):
        try:
            data = 'test'
            self.__docker_socket.send(data.encode('utf-8'))
            data = self.__docker_socket.recv(102400)
            if data:  # 如果有收到数据
                rec_data = data.decode('utf-8')
                self.read_command(rec_data)

        except Exception as e:
            print('网络断开', e)
            self.__docker_socket.close()
            time.sleep(3)


#########################################################################################################
# 线程-接收数据
def start_net_todocker_thread(docker_client):
    # 开始循环接收数据
    while True:
        docker_client.update()
        time.sleep(0.1)


# 保存文件
def save_txt(folder, ROUND, text, tm, flie):
    # linux路径
    if ROUND < 10:
        file_path = folder + '/' + flie + '0' + str(ROUND) + '.txt'
    else:
        file_path = folder + '/' + flie + str(ROUND) + '.txt'
    # 若仿真一直运行只记录前9000秒的数据
    if float(tm) <= 9000:
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                dt_ms3 = tm
                text = dt_ms3 + ':' + text
                f.write(text)
                f.write('\n')
                data_size = os.path.getsize(file_path)
            return data_size
        except Exception as e:
            print('replay saving error:', e)
    return 100*1024*1024


# 线程-处理、解析接收的数据
def start_analysis_data_thread(red, blue, episode, folder):
    dt_ms = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # print(dt_ms)
    file = red + '-' + blue + '-' + str(episode) + '-'
    folder = folder + '/' + file + dt_ms
    if not os.path.exists(folder):
        os.makedirs(folder)
    # ROUND用来区分文件的顺序
    ROUND = 1
    data_size = 0
    tm = 0
    lose_data = ''
    while True:
        data_str = ReceiveDataManage.get_data()
        # 将数据转化为相应的结构
        for i in range(len(data_str)):
            chunk = data_str[i]
            lines = chunk.split("\n")
            # print(lines)
            for line in lines:
                if data_size < 10 * 1024 * 1024:  # 限制每个文件为10M
                    # 数据完整时
                    if 'beg' in line and 'end' in line:
                        TIME = line.split('+')[-2]
                        if 'TM' in TIME:
                            tm = TIME[3:]
                        elif 'sim_time' in TIME:
                            tm = TIME[9:]
                        data_size = save_txt(folder, ROUND, line, tm, file)
                    # 只有头没有尾
                    elif 'beg' in line and 'end' not in line:
                        lose_data = line
                    # 只有尾没有头
                    elif 'end' in line and 'beg' not in line:
                        # 与上一条拼接
                        lose_data += line
                        TIME = lose_data.split('+')[-2]
                        if 'TM' in TIME:
                            tm = TIME[3:]
                        elif 'sim_time' in TIME:
                            tm = TIME[9:]
                        data_size = save_txt(folder, ROUND, lose_data, tm, file)
                        lose_data = ''
        # 文件大于10M时就再创建一个文件
        if data_size >= 10 * 1024 * 1024:
            data_size = 0
            ROUND += 1


#########################################################################################################
# 接口
#########################################################################################################
# 关闭线程
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_task(ident):
    _async_raise(ident, SystemExit)


# socket网络初始化---连接到docker的网络初始化
def net_todocker_init(docker_ip, docker_port, red, blue, episode, folder):
    # red和blue是字符串形式的战队名
    global close_dic
    if 'thread_analysis' in close_dic:  # 关闭不会自动结束的上一轮数据处理线程
        stop_task(close_dic['thread_analysis'])
    if 'thread_receive' in close_dic:
        stop_task(close_dic['thread_receive'])
    # 开启接收数据线程
    docker_client = DockerClient(docker_ip, docker_port)
    t2 = Thread(target=start_net_todocker_thread, args=(docker_client,))
    t2.start()
    # 开启处理数据线程
    t3 = Thread(target=start_analysis_data_thread, args=(red, blue, episode, folder))
    t3.start()
    # 记录线程
    close_dic['thread_receive'] = t2.ident
    close_dic['thread_analysis'] = t3.ident

