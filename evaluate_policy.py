#!/usr/bin/env python3
"""Example evaluation script to evaluate a policy.

This is an example evaluation script for evaluating a "RandomPolicy".  Use this
as a base for your own script to evaluate your policy.  All you need to do is
to replace the `RandomPolicy` and potentially the Gym environment with your own
ones (see the TODOs in the code below).

This script will be executed in an automated procedure.  For this to work, make
sure you do not change the overall structure of the script!

This script expects the following arguments in the given order:
 - The trajectory as a JSON string
 - Path to the file to which the action log is written

It is then expected to initialize the environment with the given initial pose
and execute exactly one episode with the policy that is to be evaluated.

When finished, the action log, which is created by the TriFingerPlatform class,
is written to the specified file.  This log file is crucial as it is used to
evaluate the actual performance of the policy.
"""
import argparse
import json

from rrc_example_package import cube_trajectory_env
from rrc_example_package.example import PointAtTrajectoryPolicy

# New imports
import numpy as np
import torch
from rrc_example_package.her.rl_modules.models import actor
from rrc_example_package.her.arguments import get_args
import os


class RandomPolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()
    
# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, clip_obs=200, clip_range=5):
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -clip_range, clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -clip_range, clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


def main():
    print('Setting up evaluation...')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trajectory",
        type=json.loads,
        metavar="JSON",
        help="Goal trajectory as a JSON string.",
    )
    parser.add_argument(
        "action_log_file",
        type=str,
        help="File to which the action log is written.",
    )
    args_eval = parser.parse_args()

    # TODO: Replace with your environment if you used a custom one.
    # env = cube_trajectory_env.SimCubeTrajectoryEnv(
    #     goal_trajectory=args.trajectory,
    #     action_type=cube_trajectory_env.ActionType.POSITION,
    #     # IMPORTANT: Do not enable visualisation here, as this will result in
    #     # invalid log files (unfortunately the visualisation slightly influence
    #     # the behaviour of the physics in pyBullet...).
    #     visualization=False,
    # )
    env = cube_trajectory_env.CustomSimCubeEnv(difficulty=None, sparse_rewards=False, step_size=40, distance_threshold=0.02,
                                               max_steps=float('inf'), visualisation=False, goal_trajectory=args_eval.trajectory)
    # print('Created env...')
    # # TODO: Replace this with your model
    # # policy = RandomPolicy(env.action_space)
    # policy = PointAtTrajectoryPolicy(env.action_space, args.trajectory)
    # args = get_args()
    # print('Directory: {}'.format(os.getcwd()))
    # load the model param
    model_path = 'src/pkg/rrc_example_package/her/saved_models/' + 'rrc_run3/ac_model289.pt'
    o_mean, o_std, g_mean, g_std, model, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
     # get the env param
    observation = env.reset(difficulty=None)
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()

    # Execute one episode.  Make sure that the number of simulation steps
    # matches with the episode length of the task.  When using the default Gym
    # environment, this is the case when looping until is_done == True.  Make
    # sure to adjust this in case your custom environment behaves differently!
    is_done = False
    observation = env.reset()
    accumulated_reward = 0
    t = 0
    while not is_done:
        # Begin custom
        obs = observation['observation']
        g = observation['desired_goal']
        inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std)
        with torch.no_grad():
            pi = actor_network(inputs)
        action = pi.detach().numpy().squeeze()
        # End custom
        
        # action = policy.predict(observation)
        # action = policy.predict(observation, t)
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]
        accumulated_reward += reward
        
        # print('t={}, goal={}, r={}'.format(t, g, reward))

    print("Accumulated reward: {}".format(accumulated_reward))

    # store the log for evaluation
    env.env.platform.store_action_log(args_eval.action_log_file)


if __name__ == "__main__":
    main()
