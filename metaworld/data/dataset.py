import h5py
import pickle
import numpy as np
import time
from flatten_dict import flatten

SUBGOAL_REWARD_COEFFICIENTS = {
    'assembly-v2' : [1, 5, 10, 30]
}


SUBGOAL_BREAKDOWN = {
    'assembly-v2' : ['grasp_success', 'lift_success', 'align_success']
}


DTYPES = {
    'states': np.float64,
    'observations': np.uint8,
    'actions': np.float64,
    'terminals': np.bool_,
    'rewards': np.float64,
    'infos': np.bool_
}


def target_type(data_name):
    mod_data_name = data_name

    if data_name not in DTYPES and data_name.startswith('infos'):
        mod_data_name = 'infos'

    return DTYPES[mod_data_name]


def verify_type(data_name, dtype):
    mod_data_name = data_name

    if data_name not in DTYPES and data_name.startswith('infos'):
        mod_data_name = 'infos'

    assert DTYPES[mod_data_name] == dtype, f'{data_name}\'s np.array data type is {dtype}, but should be {DTYPES[mod_data_name]}'


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
            #'img_format' : 'cwh',
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
        self.data['observations'].append(o)
        self.data['actions'].append(a)
        self.data['rewards'].append(r)
        self.data['terminals'].append(done)
        self.data['infos/goal'].append(info['success'])

        for subgoal in SUBGOAL_BREAKDOWN[self.task_name]:
            self.data['infos/' + subgoal].append(info[subgoal])


    def write_trajectory(self, max_size=None, compression='gzip'):
        np_data = {}
        for k in self.data:
            data = np.array(self.data[k], dtype=target_type(k))

            if max_size is not None:
                data = data[:max_size]
            np_data[k] = data

        trajectory = self._datafile.create_group('traj_' + str(self._num_episodes))

        for k in np_data:
            trajectory.create_dataset(k, data=np_data[k], compression=compression)

        self._num_episodes += 1
        self.data = self._reset_data()

    def close(self):
        self._datafile.close()


def qlearning_dataset(dataset_path, reward_type):
    data = h5py.File(dataset_path, "r")
    env_metadata = pickle.loads(data['env_metadata'][()].tostring())

    # Retrieve the subgoal info for the task whose data was loaded
    subgoals = ['infos/' + key for key in (SUBGOAL_BREAKDOWN[env_metadata["task_name"]] + ['goal'])]
    if reward_type=='shaped':
        subgoal_coeffs = np.asarray(SUBGOAL_REWARD_COEFFICIENTS[env_metadata["task_name"]])
        assert len(subgoals) == len(subgoal_coeffs), "The number of subgoals, including the goal, and subgoal coefficients must be the same"
    elif reward_type=='sparse':
        subgoal_coeffs_shaped = np.asarray(SUBGOAL_REWARD_COEFFICIENTS[env_metadata["task_name"]])
        subgoal_coeffs = np.zeros_like(subgoal_coeffs_shaped, dtype=np.float32)
        subgoal_coeffs[-1] = subgoal_coeffs_shaped.max()
    else:
        raise NotImplementedError

    all_states = []
    all_next_states = []
    all_obs = []
    all_next_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []

    # We are going to concatenate all trajectories for this task, D4RL-style
    for traj in data.keys():

        if traj == 'env_metadata':
            continue

        print(f'Processing trajectory {traj}')

        state_ = []
        next_state_ = []
        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []
        dataset = flatten(data[traj], reducer='path')
        N = dataset['rewards'].shape[0]

        if N > 0:
            for k in dataset:
                verify_type(k, dataset[k][0].dtype)

        for i in range(N-1):
            time_start_typecast = time.time()
            state = dataset['states'][i]
            new_state = dataset['states'][i+1]
            obs = dataset['observations'][i]
            new_obs = dataset['observations'][i+1]
            action = dataset['actions'][i]
            subgoals_achieved = np.asarray([dataset[subgoal][i] for subgoal in subgoals], dtype=np.float32)
            #TODO: decide whether subgoal rewards should always be *summed*. E.g., what if one subgoal implies another?
            reward = np.dot(subgoal_coeffs, subgoals_achieved)
            done_bool = dataset['terminals'][i]

            state_.append(state)
            next_state_.append(new_state)
            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)

        """
        When the data was recorded, the agent had to spend at least env_metadata['success_steps_for_termination']
        consecutive steps at the goal for the trajectory to be considered successful, so successful trajectories
        have env_metadata['success_steps_for_termination'] near-duplicate timesteps at the end. Below, we'll get
        rid of all but 1 of them.
        """
        goal_reached = True

        for t in range(N - env_metadata['success_steps_for_termination'], N):
            if not dataset['infos/goal'][i]:
                goal_reached = False

        if goal_reached:
            state_ = state_[: -env_metadata['success_steps_for_termination'] + 1]
            next_state_ = next_state_[: -env_metadata['success_steps_for_termination'] + 1]
            obs_ = obs_[: -env_metadata['success_steps_for_termination'] + 1]
            next_obs_ = next_obs_[: -env_metadata['success_steps_for_termination'] + 1]
            action_ = action_[: -env_metadata['success_steps_for_termination'] + 1]
            reward_ = reward_[: -env_metadata['success_steps_for_termination'] + 1]
            done_ = done_[: -env_metadata['success_steps_for_termination'] + 1]
            done_[-1] = True

        all_states.extend(state_)
        all_next_states.extend(next_state_)
        all_obs.extend(obs_)
        all_next_obs.extend(next_obs_)
        all_actions.extend(action_)
        all_rewards.extend(reward_)
        all_dones.extend(done_)


    return {
        'states': np.array(all_states),
        'next_states': np.array(all_next_states),
        'observations': np.array(all_obs),
        'next_observations': np.array(all_next_obs),
        'actions': np.array(all_actions),
        'rewards': np.array(all_rewards),
        'terminals': np.array(all_dones),
    }


class MWQLearningDataset:
    def __init__(self, dataset_path):
        self.data = h5py.File(dataset_path, "r")
