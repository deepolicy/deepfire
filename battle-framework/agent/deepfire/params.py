
class train_params():
    submit = 0  # for submit or not.
    dev = 0  # develop model
    if dev:
        horizon = 128//8
        mini_batch = 32//8
        num_group_memory = 1

    else:
        horizon = 1024
        mini_batch = 128
        num_group_memory = 1

class data_params():
    if train_params.submit == 1:
        data_dir = "./"
    else:
        data_dir = "../"+"battle-framework/"

    write_cmd_json = {
        "blue": 0,
    }

class run_params():
    ip = "226"
    env = "0"
