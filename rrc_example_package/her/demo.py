import torch
from rrc_example_package.her.rl_modules.models import actor
from rrc_example_package.her.arguments import get_args
import gym
import numpy as np

from rrc_example_package import cube_trajectory_env
import time

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def main():
    args = get_args()
    # load the model param
    model_path = args.save_dir + 'rrc_run3/ac_model289.pt'
    o_mean, o_std, g_mean, g_std, model, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = cube_trajectory_env.CustomSimCubeEnv(difficulty=None, sparse_rewards=False, step_size=40, distance_threshold=0.02,
                                               max_steps=50, visualisation=True)
    # get the env param
    observation = env.reset(difficulty=None)
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    print('Observation Space:')
    print(env.observation_space)
    print('\nAction space:')
    print(env.action_space)
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    t0 = time.time()
    input()
    for i in range(1):
        # observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        is_done = False
        # print('obs.shape: {}'.format(obs.shape))
        for t in range(env._max_episode_steps*50):
        # while not is_done:
            obs = observation['observation']
            g = observation['desired_goal']
            # env.render()
            # print('obs: {}'.format(obs))
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # print('Action: {}'.format(action))
            # put actions into the environment
            observation, reward, is_done, info = env.step(action)
            # input()
            print('t={}, g={}, r={}'.format(info["time_index"], g, reward))
            # print(info)
            # obs = observation['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
    t1 = time.time()
    print('Time taken: {:.2f} seconds'.format(t1-t0))

if __name__ == '__main__':
    main()
    
