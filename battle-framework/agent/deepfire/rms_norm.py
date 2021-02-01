from .rms_norm_base import RunningMeanStd
import numpy as np

class _rms_norm():
    '''
        remember to set rms_len, it is important.
    '''
    @staticmethod
    def init(obs_in):

        rms = {}

        assert np.array(obs_in['airport']).shape == (13,)

        rms['airport'] = RunningMeanStd(shape=np.array([obs_in['airport']]).shape[1:])

        for key, rms_len in zip(['units', 'teams', 'qb'], [12, 1, 8]):
            for LX in obs_in[key]:
                assert len(np.array(obs_in[key][LX]).shape) == 2
                rms[key+'-'+str(LX)] = RunningMeanStd(shape=np.array(obs_in[key][LX])[:,:rms_len].shape[1:])

        key = 'rockets'
        rms_len = 8
        assert len(np.array(obs_in[key]).shape) == 2
        rms[key] = RunningMeanStd(shape=np.array(obs_in[key])[:,:rms_len].shape[1:])


        return rms

    @staticmethod
    def load(rms, rms_file):

        rms_data = np.load(rms_file, allow_pickle=True).tolist()

        for k in rms:
            rms[k].mean, rms[k].var, rms[k].count = rms_data[k]['mean'], rms_data[k]['var'], rms_data[k]['count']

        return rms

    @staticmethod
    def save(rms, rms_file):
        rms_data = {}
        for k in rms:
            rms_data[k] = {}
            rms_data[k]['mean'], rms_data[k]['var'], rms_data[k]['count'] = rms[k].mean, rms[k].var, rms[k].count

        np.save(rms_file, rms_data)

    @staticmethod
    def step(obs_in, rms):
        clip_range=[-5.0, 5.0]

        def step_2(x, i_rms):
            x = np.array(x)
            i_rms.update(x)
            assert np.isfinite((x - i_rms.mean) / np.sqrt(i_rms.var)).all()
            norm_x = np.clip((x - i_rms.mean) / np.sqrt(i_rms.var), min(clip_range), max(clip_range))
            return norm_x

        obs_in['airport'] = step_2([obs_in['airport']], rms['airport'])[0]

        # why need not trans obs_in['airport'] to list?

        for key, rms_len in zip(['units', 'teams', 'qb'], [12, 1, 8]):
            for LX in obs_in[key]:
                obs_in[key][LX] = np.array(obs_in[key][LX])
                obs_in[key][LX][:,:rms_len] = step_2(obs_in[key][LX][:,:rms_len], rms[key+'-'+str(LX)])
                obs_in[key][LX] = obs_in[key][LX].tolist()

        key = 'rockets'
        rms_len = 8
        obs_in[key] = np.array(obs_in[key])
        obs_in[key][:,:rms_len] = step_2(obs_in[key][:,:rms_len], rms[key])
        obs_in[key] = obs_in[key].tolist()

        return obs_in
