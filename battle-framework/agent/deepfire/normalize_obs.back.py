import numpy as np

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape=(), to_update=1, epsilon=1e-4):
        self.to_update = to_update
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        
        self.epsilon=1e-8
        self.clip_range=5.0

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
        
    def norm(self,x):
        if self.to_update:
            self.update(x)

        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_range, self.clip_range)
    
    def set_data(self,data):
        self.mean,self.var,self.count = data
        
    def get_data(self):
        return [self.mean,self.var,self.count]
        
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class NormalizeObs():
    def __init__(self,obs):

        # will not call me.
        assert 0

        self.rms = {k:{} for k in ["airports","teams","units","qb","rockets"]}
        
        keys = ["X","Y","Z"]
        self.rms["airports"]["rms"] = RunningMeanStd(len(keys))
        
    def step(self,obs):
        '''
        
        return obs after normalization
        '''
        
        keys = ["X","Y","Z"]
        vals = self.rms["airports"]["rms"].norm( np.array([ obs["airports"][0][k] for k in keys ]) )
        for i,k in enumerate(keys):
            obs["airports"][0][k] = vals[i]

        return obs
            
    def save(self, filename="rms_data.npy"):
        rms_data = {k:{} for k in ["airports","teams","units","qb","rockets"]}
        for k in self.rms:
            for k2 in self.rms[k]:
                rms_data[k][k2] = self.rms[k][k2].get_data()
                
        np.save(filename,rms_data)
        
    def load(self, filename="rms_data.npy"):
        '''
            if rms_data has such key, set the value of self.rms by this key. 
        '''
        rms_data = np.load(filename, allow_pickle=True).tolist()
        for k in rms_data:
            for k2 in rms_data[k]:
                self.rms[k][k2].set_data( rms_data[k][k2] )


def normalize_obs_static(obs):
    '''
        airports
    '''
    assert len(obs["airports"]) == 1
    obs["airports"][0]["DA"] /= 100.0

    '''
        teams
    '''

    '''
        units
    '''

    for i in range(len(obs["units"])):
        obs["units"][i]["HX"] /= 360.0
        obs["units"][i]["DA"] /= 100.0
        obs["units"][i]["Hang"] /= 100000.
        obs["units"][i]["Fuel"] /= 10000.0

    k = "qb"
    for i in range(len(obs["qb"])):
        obs[k][i]["HX"] /= 360.0
        obs[k][i]["DA"] /= 100.0

    return obs


class NormalizeObsRunning():
    def __init__(self, update, rms_data_file):

        '''
            set RunningMeanStd norm at the same time,
        '''
        self.update = update  # update mean, var, count or not.
        self.rms_data_file = rms_data_file

        self.rms = {k: {} for k in ["airports", "teams", "units", "qb", "rockets"]}

        self.airports_keys = keys = "X, Y, Z, CA, NM, KY, AIR, AWCS, JAM, UAV, BOM".split(", ")
        self.rms["airports"]["rms"] = RunningMeanStd(len(keys), to_update=self.update)

        self.units_keys = keys = "X, Y, Z, SP, TM".split(", ")
        self.rms["units"]["rms"] = RunningMeanStd(len(keys), to_update=self.update)

        self.qb_keys = keys = "X, Y, Z, SP, TM".split(", ")
        self.rms["qb"]["rms"] = RunningMeanStd(len(keys), to_update=self.update)

        self.rockets_keys = keys = "X, Y, Z, FY, HG, HX, TM".split(", ")
        self.rms["rockets"]["rms"] = RunningMeanStd(len(keys), to_update=self.update)

        self.i_step = 0

        if not self.update:
            self.load()

    def step(self, obs):
        '''

        return obs after normalization
        '''

        keys = self.airports_keys
        vals = self.rms["airports"]["rms"].norm(np.array([obs["airports"][0][k] for k in keys]))
        for i, k in enumerate(keys):
            obs["airports"][0][k] = vals[i]

        keys = self.units_keys
        for i_unit in range(len(obs["units"])):
            vals = self.rms["units"]["rms"].norm(np.array([obs["units"][i_unit][k] for k in keys]))
            for i, k in enumerate(keys):
                obs["units"][i_unit][k] = vals[i]

        keys = self.qb_keys
        for i_qb in range(len(obs["qb"])):
            vals = self.rms["qb"]["rms"].norm(np.array([obs["qb"][i_qb][k] for k in keys]))
            for i, k in enumerate(keys):
                obs["qb"][i_qb][k] = vals[i]

        keys = self.rockets_keys
        for i_rocket in range(len(obs["rockets"])):
            vals = self.rms["rockets"]["rms"].norm(np.array([obs["rockets"][i_rocket][k] for k in keys]))
            for i, k in enumerate(keys):
                obs["rockets"][i_rocket][k] = vals[i]

        self.i_step += 1

        if self.update:
            if self.i_step == 1000:
                self.save()
                self.i_step = 0

        return obs

    def save(self, filename="rms_data.npy"):
        filename = self.rms_data_file
        rms_data = {k: {} for k in ["airports", "teams", "units", "qb", "rockets"]}
        for k in self.rms:
            for k2 in self.rms[k]:
                rms_data[k][k2] = self.rms[k][k2].get_data()

        np.save(filename, rms_data)

    def load(self, filename="rms_data.npy"):
        '''
            if rms_data has such key, set the value of self.rms by this key.
        '''
        filename = self.rms_data_file
        import os
        if not os.path.exists(filename):
            return False

        rms_data = np.load(filename, allow_pickle=True).tolist()
        for k in rms_data:
            for k2 in rms_data[k]:
                self.rms[k][k2].set_data(rms_data[k][k2])


def main():
    pass

if __name__ == '__main__':
    main()
