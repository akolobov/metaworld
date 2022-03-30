import metaworld
import random
import cv2
import os
import sys
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from metaworld.policies import *
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import test_cases_latest_nonoise
from metaworld.data.dataset import *
from datetime import datetime
import gym
import argparse

# Suppress float conversion warnings 
gym.logger.set_level(40)


# TODO: This doesn't work. How do we seed single goal envs?
"""
SEED = 0  # some seed number here
benchmark = metaworld.BENCHMARK(seed=SEED)
"""


###########################
# Instructions for using different renderers (CPU vs GPU) with mujoco-py: http://vedder.io/misc/mujoco_py.html
###########################


##########################
"""
For constructing semi-shaped (our) reward, note that every env. has an evaluate_state(.) method, which returns an info dict with
various reward components., such as in_place_reward, near_object, etc. We just need to interpret them and assign our reward instead
of the one provided by MW.
"""

def writer_for(tag, fps, res):
    if not os.path.exists('movies'):
        os.mkdir('movies')
    return cv2.VideoWriter(
        f'movies/{tag}.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        fps,
        res
    )

# TODO: which action noise level to apply? For ideas, see https://github.com/rlworkgroup/metaworld/blob/cfd837e31d65c9d2b62b7240c68a26b04a9166d9/tests/metaworld/envs/mujoco/sawyer_xyz/test_scripted_policies.py
def config_env(e):
    e._partially_observable = False
    e._freeze_rand_vec = False
    e._set_task_called = True


def gen_data(tasks, num_traj, noise, res, camera):
    res = (res, res)
    #camera = 'corner' # one of ['topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV']
    MAX_steps_at_goal = 10
    act_tolerance = 1e-5
    lim = 1 - act_tolerance

    #print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

    for case in test_cases_latest_nonoise:
        
        if case[0] not in tasks: # target_tasks:
            continue
        
        task_name = case[0]
        policy = case[1]

        print(f'----------Running task {task_name}------------')

        """
        # This uses a single goal distribution. For generating the training data, use ML1's training distrib instead.
        task_full_name = task_name + '-goal-observable'
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_full_name]()
        #print(f'\nPROCESSING TASK***{task_full_name}***')
        """

        # TODO: record fixed-length trajectories? Or only until success?
        env = metaworld.mw_gym_make(task_name, sparse_reward=False, stop_at_goal=True, steps_at_goal=MAX_steps_at_goal)
        action_space_ptp = env.action_space.high - env.action_space.low

        num_successes = 0

        print(f'Generating a video at {env.metadata["video.frames_per_second"]} fps')

        dt = datetime.now() 
        #str_date_time = dt.strftime("%d-%m-%Y-%H:%M:%S")
        #print(str_date_time)
        data_file_path = os.path.join(os.environ['JAXRL2_DATA'], task_name + '_' + str(num_traj) + '-noise_' + str(noise) + '-traj_' + dt.strftime("%d-%m-%Y-%H.%M.%S") + '.h5py')
        data_writer = MWDatasetWriter(data_file_path, env, task_name, res, camera, act_tolerance, MAX_steps_at_goal)

        for attempt in range(num_traj):
            writer = writer_for(task_name + '-' + str(attempt + 1), env.metadata['video.frames_per_second'], res)

            state = env.reset()
            obs = env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]
            writer.write(obs)

            for t in range(env.max_path_length):
                action = policy.get_action(state)
                action = np.random.normal(action, noise * action_space_ptp)
                # Clip the action
                action = np.clip(action, -lim, lim)
                new_state, reward, done, info = env.step(action)
                data_writer.append_data(state, obs, action, reward, done, info)

                strpr = f"Step {t} |||"
                for k in info:
                    strpr += f"{k}: {info[k]}, "
                #print(strpr)
                state = new_state
                obs = env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]
                #env.sim.render(*res, mode='window', camera_name=camera)
                writer.write(obs)
                
                if done:
                    if info['task_accomplished']:
                        print(f'Attempt {attempt + 1} succeeded at step {t}')
                        num_successes += 1
                    else:
                        print(f'Attempt {attempt + 1} ended unsuccessfully at time step {t}')
                    break

            data_writer.write_trajectory()

        data_writer.close()
        print(f'Success rate for {task_name}: {num_successes / num_traj}\n')
        
        # Check the created dataset
        qlearning_dataset(data_file_path, reward_type='subgoal')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tasks", type=str, nargs='+', help = "Tasks for which to generate trajectories from scripted policies")
    parser.add_argument("-n", "--num_traj", type=int, help = "Number of trajectories to generate for each task")
    parser.add_argument("-p", "--noise", type=float, default=0, help = "Action noise as a fraction of the action space, e.g., 0.1")
    parser.add_argument("-r", "--res", type=int, default=84, help = "Image resolution")
    parser.add_argument("-c", "--camera", type=str, default='corner', help = "Camera. Possible values: 'corner', 'topview', 'corner2', 'corner3', 'behindGripper', 'gripperPOV' ")
    args = parser.parse_args()
    print(f'Generating {args.num_traj} trajectories with action noise {args.noise} for tasks {args.tasks} with video resolution {args.res}x{args.res} and {args.camera} camera view.')
    gen_data(args.tasks, args.num_traj, args.noise, args.res, args.camera)
