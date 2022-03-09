import h5py
import pickle
import numpy as np


SUBGOAL_REWARD_COEFFICIENTS = {
    'assembly-v2' : [1, 5, 10, 30]
}


SUBGOAL_BREAKDOWN = {
    'assembly-v2' : ['grasp_success', 'lift_success', 'align_success']
}


class MWDatasetWriter:
    def __init__(self, fname, env, task_name, res, camera, success_steps_for_termination):
        # The number of steps with with info/success = True required to trigger episode termination
        self.task_name = task_name
        self.success_steps_for_termination = success_steps_for_termination
        raw_metadata = {
            'task_name' : task_name,
            'horizon' : env.max_path_length,
            'fps': env.metadata["video.frames_per_second"],
            'frame_skip' : env.frame_skip,
            'img_width' : res[0],
            'img_height' : res[1],
            'img_format' : 'cwh',
            'camera' : camera,
            'subgoal_breakdown' : SUBGOAL_BREAKDOWN[task_name], 
            'success_steps_for_termination' : success_steps_for_termination
        }

        self.metadata = np.void(pickle.dumps(raw_metadata))
        self._datafile = h5py.File(fname, 'w')
        self._datafile.create_dataset("env_metadata", data=self.metadata)

        self.data = self._reset_data()
        self._num_episodes = 0
        

    def _reset_data(self):
        data = {
            'states': [],
            'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': []
            }
        
        for subgoal in SUBGOAL_BREAKDOWN[self.task_name]:
            data['infos/' + subgoal] = []
        
        return data


    def append_data(self, s, o, a, r, done, info):
        self.data['states'].append(s)
        #self.data['observations'].append(np.asarray(o))
        self.data['observations'].append(o)
        self.data['actions'].append(a)
        self.data['rewards'].append(r)
        self.data['terminals'].append(done)
        self.data['infos/goal'].append(info['success'])

        for subgoal in SUBGOAL_BREAKDOWN[self.task_name]:
            self.data['infos/' + subgoal].append(float(info[subgoal]))


    def write_trajectory(self, max_size=None, compression='gzip'):
        np_data = {}
        for k in self.data:
            if k == 'terminals':
                dtype = np.bool_
            else:
                dtype = np.float32
            data = np.array(self.data[k], dtype=dtype)
            if max_size is not None:
                data = data[:max_size]
            np_data[k] = data

        trajectory = self._datafile.create_group('traj_' + str(self._num_episodes))
        self._num_episodes += 1

        for k in np_data:
            trajectory.create_dataset(k, data=np_data[k], compression=compression)

    def close(self):
        self._datafile.close()
