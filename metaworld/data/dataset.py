import h5py
import pickle
import numpy as np
import time
from flatten_dict import flatten

MAX_MW_REWARD = 10

SUBGOAL_REWARD_COEFFICIENTS = {
    'assembly-v2' : [1, 5, 10, 30],
    'button-press-v2' : [1, 5, 10, 30]
}


SUBGOAL_BREAKDOWN = {
    'assembly-v2' : ['grasp_success', 'lift_success', 'align_success'],
    'button-press-v2' : ['nearby_success', 'near_button_success', 'button_pressed_success']
}


DTYPES = {
    'full_states': np.float64,
    'proprio_states': np.float64,
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


def check_action(a, act_lim):
        assert (-act_lim <= a).all() and (a <= act_lim).all(), f'Action {a} has entries outside the [{-act_lim}, {act_lim}] range.'


class MWDatasetWriter:
    def __init__(self, fname, env, task_name, res, camera, act_tolerance, success_steps_for_termination):
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
            'act_tolerance' : act_tolerance,
            'subgoal_breakdown' : SUBGOAL_BREAKDOWN[task_name],
            'success_steps_for_termination' : success_steps_for_termination
        }

        self._act_lim = 1 - act_tolerance
        self.metadata = np.void(pickle.dumps(raw_metadata))
        self._datafile = h5py.File(fname, 'w')
        self._datafile.create_dataset("env_metadata", data=self.metadata)

        self.data = self._reset_data()
        self._num_episodes = 0


    def _reset_data(self):
        data = {
            'full_states': [],
            'proprio_states': [],
            'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': []
            }

        for subgoal in SUBGOAL_BREAKDOWN[self.task_name]:
            data['infos/' + subgoal] = []

        return data


    def append_data(self, f, p, o, a, r, done, info):
        self.data['full_states'].append(f)
        self.data['proprio_states'].append(p)
        self.data['observations'].append(o)
        check_action(a, self._act_lim)
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
    act_lim = 1 - env_metadata['act_tolerance']

    # Retrieve the subgoal info for the task whose data was loaded
    subgoals = ['infos/' + key for key in (SUBGOAL_BREAKDOWN[env_metadata["task_name"]] + ['goal'])]
    if reward_type=='subgoal':
        subgoal_coeffs = np.asarray(SUBGOAL_REWARD_COEFFICIENTS[env_metadata["task_name"]])
        assert len(subgoals) == len(subgoal_coeffs), "The number of subgoals, including the goal, and subgoal coefficients must be the same"
    elif reward_type=='sparse':
        subgoal_coeffs_shaped = np.asarray(SUBGOAL_REWARD_COEFFICIENTS[env_metadata["task_name"]])
        subgoal_coeffs = np.zeros_like(subgoal_coeffs_shaped, dtype=np.float32)
        subgoal_coeffs[-1] = subgoal_coeffs_shaped.max()
    elif reward_type=='shaped':
        pass
    else:
        raise NotImplementedError

    all_full_states = []
    all_next_full_states = []
    all_proprio_states = []
    all_next_proprio_states = []
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

        full_state_ = []
        next_full_state_ = []
        proprio_state_ = []
        next_proprio_state_ = []
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
            full_state = dataset['full_states'][i]
            new_full_state = dataset['full_states'][i+1]
            proprio_state = dataset['proprio_states'][i]
            new_proprio_state = dataset['proprio_states'][i+1]
            obs = dataset['observations'][i]
            new_obs = dataset['observations'][i+1]
            action = dataset['actions'][i]
            check_action(action, act_lim)
            #TODO: decide whether subgoal rewards should always be *summed*. E.g., what if one subgoal implies another?
            if reward_type=='shaped':
                reward =  dataset['rewards'][i] - MAX_MW_REWARD
                if dataset['infos/goal'][i]:
                    reward = 0
            elif reward_type in ['sparse', 'subgoal']:
                subgoals_achieved = np.asarray([dataset[subgoal][i] for subgoal in subgoals], dtype=np.float32)
                # The "- np.max(subgoal_coeffs)" is to ensure that goal-state reward (which equals np.max(subgoal_coeffs)) is 0.
                reward = np.dot(subgoal_coeffs, subgoals_achieved) - np.max(subgoal_coeffs)
            else:
                raise NotImplementedError()
            done_bool = dataset['terminals'][i]

            full_state_.append(full_state)
            next_full_state_.append(new_full_state)
            proprio_state_.append(proprio_state)
            next_proprio_state_.append(new_proprio_state)
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
        """
        goal_reached = True

        for t in range(N - env_metadata['success_steps_for_termination'], N):
            if not dataset['infos/goal'][i]:
                goal_reached = False

        if goal_reached:
            full_state_ = full_state_[: -env_metadata['success_steps_for_termination'] + 1]
            next_full_state_ = next_full_state_[: -env_metadata['success_steps_for_termination'] + 1]
            obs_ = obs_[: -env_metadata['success_steps_for_termination'] + 1]
            next_obs_ = next_obs_[: -env_metadata['success_steps_for_termination'] + 1]
            action_ = action_[: -env_metadata['success_steps_for_termination'] + 1]
            reward_ = reward_[: -env_metadata['success_steps_for_termination'] + 1]
            done_ = done_[: -env_metadata['success_steps_for_termination'] + 1]
            done_[-1] = True
        """

        all_full_states.extend(full_state_)
        all_next_full_states.extend(next_full_state_)
        all_proprio_states.extend(proprio_state_)
        all_next_proprio_states.extend(next_proprio_state_)
        all_obs.extend(obs_)
        all_next_obs.extend(next_obs_)
        all_actions.extend(action_)
        all_rewards.extend(reward_)
        all_dones.extend(done_)


    return {
        'states': np.array(all_full_states),
        'next_states': np.array(all_next_full_states),
        'proprio_states': np.array(all_proprio_states),
        'next_proprio_states': np.array(all_next_proprio_states),
        'observations': np.array(all_obs),
        'next_observations': np.array(all_next_obs),
        'actions': np.array(all_actions),
        'rewards': np.array(all_rewards),
        'terminals': np.array(all_dones),
    }


class MWQLearningDataset:
    def __init__(self, dataset_path):
        self.data = h5py.File(dataset_path, "r")
