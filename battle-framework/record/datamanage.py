#######################################################################################################################
""" 保存通过网络接收到的数据 """
#######################################################################################################################
from threading import Lock


class ReceiveDataManage:
    __write_data_list = []  # 接收到的写文件数据列表
    __write_lock = Lock()
    """
    __send_data_list = []  # 接收到的及时发给各个客户端数据列表(暂不启用)
    __send_lock = Lock()
    """

    # 保存数据
    @classmethod
    def save_data(cls, new_data_str):
        cls.__write_lock.acquire()
        cls.__write_data_list.append(new_data_str)
        cls.__write_lock.release()

        """
        cls.__send_lock.acquire()
        cls.__send_data_list.append(new_data_str)
        cls.__send_lock.release()
        """

    # 取出数据
    @classmethod
    def get_data(cls):
        cls.__write_lock.acquire()
        return_data_list = cls.__write_data_list[0:]
        cls.__write_data_list.clear()
        cls.__write_lock.release()
        return return_data_list

    """
    # 取出发送数据
    @classmethod
    def get_send_data(cls):
        cls.__send_lock.acquire()
        return_data_list = cls.__send_data_list[0:]
        cls.__send_data_list.clear()
        cls.__send_lock.release()
        return return_data_list
    """
