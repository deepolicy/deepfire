import os
import numpy as np
import matplotlib.pyplot as plt

'''
    plot final reward per epsilon with data from reward directory.
'''

def plot_reward_ep(reward_ep_list, plot_epsilon=True):
    sum_reward_list = []
    sum_reward = 0
    for reward in reward_ep_list:
        sum_reward += reward
        sum_reward_list.append(sum_reward)
    if plot_epsilon:
        plt.plot(sum_reward_list)
        plt.show()
    return sum_reward_list[-1]

def sort(file_name):
    def get_file_index(file):
        return int(file.split("-")[1].split(".")[0])

    L = [[get_file_index(file), file] for file in file_name]
    L.sort(key=lambda x: x[0])

    file_name = [i[1] for i in L]

    return file_name

def main():


    def get_dirs(path='./'):
        return [i for i in os.listdir(path) if os.path.isdir(path+i)]

    path = './'
    dir_list = get_dirs(path=path)

    plot_all = []

    for i_dir in dir_list:
        print('')
        print(i_dir)
        print('')

        for path, save_fig in zip([i_dir+"/battle-framework/agent/deepfire/data/reward/"], [i_dir+".reward.png"]):
            file_name = os.listdir(path)

            last_reward_list = []

            file_name = sort(file_name)

            for file in file_name:
                last_reward_list.append(plot_reward_ep(np.load(path + file), plot_epsilon=False))
                # print(file)
                # print("." * 50)
                # print("")

            plt.plot(last_reward_list)
            import time
            plt.title(i_dir+"-Plot at " + time.strftime("%m-%d %H:%M", time.localtime()))
            plt.savefig(save_fig)
            plt.cla()
            plot_all.append([i_dir, last_reward_list])


    for i_r in plot_all:
        i, r = i_r
        plt.plot(r)
    plt.legend([i[0] for i in plot_all])
    import time
    plt.title("All-Plot at " + time.strftime("%m-%d %H:%M", time.localtime()))
    plt.savefig('all.png')

if __name__ == "__main__":
    main()