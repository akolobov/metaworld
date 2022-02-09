import metaworld
import random
import cv2
import os
import gym
# Suppress float conversion warnings 
gym.logger.set_level(40)
from metaworld.policies import *
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import test_cases_latest_nonoise



###########################
# Instructions for using different renderers (CPU vs GPU) with mujoco-py: http://vedder.io/misc/mujoco_py.html
###########################


def writer_for(tag, fps, res):
    if not os.path.exists('movies'):
        os.mkdir('movies')
    return cv2.VideoWriter(
        f'movies/{tag}.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        fps,
        res
    )

# TODO: WHICH ACTION NOISE LEVEL TO CHOOSE? HOW TO SET IT? FOR HINTS, SEE https://github.com/rlworkgroup/metaworld/blob/cfd837e31d65c9d2b62b7240c68a26b04a9166d9/tests/metaworld/envs/mujoco/sawyer_xyz/test_scripted_policies.py
def config_env(e):
    e._partially_observable = False
    e._freeze_rand_vec = False
    e._set_task_called = True


res = (640, 480)
camera = 'corner' # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
#print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

for case in test_cases_latest_nonoise:
    #task_name = 'pick-place-v2'
    if case[0][-3:] != '-v2':
        continue

    task_name = case[0]
    print(f'----------Running task {task_name}------------')
    ml1 = metaworld.ML1(task_name) # Construct the benchmark, sampling tasks

    env = ml1.train_classes[task_name]()  # Create an environment with task `pick_place`

    # TODO: ML1 or single-goal envs? Figure out!
    task = random.choice(ml1.train_tasks)

    print(f'Generating a video at {env.metadata["video.frames_per_second"]} fps')

    env.set_task(task)  # Set task
    config_env(env)
    policy = case[1]

    """
    policy = None

    for case in test_cases_latest_nonoise:
        if case[0] == task_name:
            print(f'Found a scripted policy for {task_name}!')
            policy = case[1]
            break
    """

    #print(f'Problem horizon: {env.max_path_length}')

    num_successes = 0
    num_attemps = 10

    for attempt in range(num_attemps):
        #writer = writer_for(task_name + '/' + task_name + '-' + str(attempt), env.metadata['video.frames_per_second'], res)
        writer = writer_for(task_name + '-' + str(attempt + 1), env.metadata['video.frames_per_second'], res)
        state = env.reset()  # Reset environment
        success_recorded =  False
        
        t = 0
        for t in range(env.max_path_length):
            a = policy.get_action(state) #env.action_space.sample()  # Sample an action
            state, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
            obs = env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]
            writer.write(obs)
            # TODO: record fixed-length trajectories? Or only until success? 
            
            if info['success'] and not success_recorded:
                print(f'Attempt {attempt} succeeded at step {t}')
                num_successes += 1
                success_recorded = True
                #break

        print(f'Episode ended at time step {t}')

    print(f'Success rate for {task_name}: {num_successes / num_attemps}\n')
