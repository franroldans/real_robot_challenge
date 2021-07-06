import numpy as np
import torch
from rrc_example_package.her.rl_modules.models import flatten_mlp, tanh_gaussian_actor
from rrc_example_package.her.rl_modules.replay_buffer import replay_buffer
from rrc_example_package.her.utils import get_action_info
from rrc_example_package.her.her_modules.her import her_sampler
from datetime import datetime
import copy
import os
import gym
import torch.nn as nn

"""
2019-Nov-12 - start to add the automatically tempature tuning
2019-JUN-05
author: Tianhong Dai
"""

# the soft-actor-critic agent
class sac_agent_rrc:
    def __init__(self, env, args, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # create eval environment
        #self.eval_env = gym.make(self.args.env_name)
        #self.eval_env.seed(args.seed * 2)
        # build up the network that will be used.
        self.qf1 = flatten_mlp(env_params['obs'], self.args.hidden_size, env_params['action'])
        self.qf2 = flatten_mlp(env_params['obs'], self.args.hidden_size, env_params['action'])
        # set the target q functions
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        # build up the policy network
        self.actor_net = tanh_gaussian_actor(env_params['obs'], env_params['action'], self.args.hidden_size, \
                                            self.args.log_std_min, self.args.log_std_max)
        # define the optimizer for them
        self.qf1_optim = torch.optim.Adam(self.qf1.parameters(), lr=self.args.q_lr)
        self.qf2_optim = torch.optim.Adam(self.qf2.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=self.args.p_lr)
        # entorpy target
        self.target_entropy = -np.prod(self.env.action_space.shape).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda' if self.args.cuda else 'cpu')
        # define the optimizer
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.p_lr)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # get the action max
        self.action_max = self.env.action_space.high[0]
        # if use cuda, put tensor onto the gpu
        if self.args.cuda:
            self.actor_net.cuda()
            self.qf1.cuda()
            self.qf2.cuda()
            self.target_qf1.cuda()
            self.target_qf2.cuda()
        # automatically create the folders to save models
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = os.path.join(self.args.save_dir, 'mymodels')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    # train the agent
    def learn(self):
        global_timesteps = 0
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._collect_exp()
        # reset the environment
        obs = self.env.reset(difficulty=self.sample_difficulty())
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.train_loop_per_epoch):
                # for each epoch, it will reset the environment
                for t in range(self.args.epoch_length):
                    # start to collect samples
                    with torch.no_grad():
                        obs_tensor = self._get_tensor_inputs(obs["observation"])
                        """obs = obs_tensor['observation']
                        ag = obs_tensor['achieved_goal']
                        g = obs_tensor['desired_goal']"""
                        pi = self.actor_net(obs_tensor)
                        action = get_action_info(pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                        action = action.cpu().numpy()[0]
                    # input the actions into the environment
                    obs_, reward, done, _ = self.env.step(self.action_max * action)
                    # store the samples
                    self.buffer.add(obs, action, reward, obs_, float(done))
                    # reassign the observations
                    obs = obs_
                    if done:
                        # reset the environment
                        obs = self.env.reset(difficulty=self.sample_difficulty())
                # after collect the samples, start to update the network
                for _ in range(self.args.update_cycles):
                    qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss = self._update_newtork()
                    # update the target network
                    if global_timesteps % self.args.target_update_interval == 0:
                        self._update_target_network(self.target_qf1, self.qf1)
                        self._update_target_network(self.target_qf2, self.qf2)
                    global_timesteps += 1
            # print the log information
            if epoch % self.args.display_interval == 0:
                # start to do the evaluation
                mean_rewards = self._evaluate_agent()
                print('[{}] Epoch: {} / {}, Frames: {}, Rewards: {:.3f}, QF1: {:.3f}, QF2: {:.3f}, AL: {:.3f}, Alpha: {:.5f}, AlphaL: {:.5f}'.format(datetime.now(), \
                            epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length, mean_rewards, qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss))
                # save models
                torch.save(self.actor_net.state_dict(), self.model_path + '/model.pt')
                
    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # delta calculation
        mb_obs = np.clip(mb_obs, -self.args.clip_obs, self.args.clip_obs)
        mb_delta = mb_obs[:,1:,:] - mb_obs[:,:-1,:]
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        self.delta_norm.update(mb_delta)
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        self.delta_norm.recompute_stats()

    # do the initial exploration by using the uniform policy
    def _collect_exp(self, rollouts=100, difficulty=1):
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for _ in range(rollouts):
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environment
            observation = self.env.reset(difficulty=difficulty)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            # start to collect samples
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    action = self._select_actions(pi)
                # feed the actions into the environment
                observation_new, _, _, info = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)
        # convert them into arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)
        # store the episodes
        self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
        
    # get tensors
    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor
    
    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)
    
    # update the network
    def _update_newtork(self):

        transitions = self.buffer.sample(self.args.batch_size)
        
        # Add intrinsic reward
        r_intrinsic = self.get_intrinsic_reward(transitions['obs'], transitions['actions'], transitions['obs_next'])
        transitions['r'] += r_intrinsic
        ri = np.mean(r_intrinsic)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
         # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        
        # start to update the actor network
        pis = self.actor_net(obs_norm)
        actions_info = get_action_info(pis, cuda=self.args.cuda)
        actions_, pre_tanh_value = actions_info.select_actions(reparameterize=True)
        log_prob = actions_info.get_log_prob(actions_, pre_tanh_value)
        # use the automatically tuning
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        # get the param
        alpha = self.log_alpha.exp()
        # get the q_value for new actions
        q_actions_ = torch.min(self.qf1(obs_norm, actions_), self.qf2(obs_norm, actions_))
        actor_loss = (alpha * log_prob - q_actions_).mean()
        # q value function loss
        q1_value = self.qf1(obs_norm, actions)
        q2_value = self.qf2(obs_norm, actions)
        with torch.no_grad():
            pis_next = self.actor_net(obses_)
            actions_info_next = get_action_info(pis_next, cuda=self.args.cuda)
            actions_next_, pre_tanh_value_next = actions_info_next.select_actions(reparameterize=True)
            log_prob_next = actions_info_next.get_log_prob(actions_next_, pre_tanh_value_next)
            target_q_value_next = torch.min(self.target_qf1(obs_next_norm, actions_next_), self.target_qf2(obs_next_norm, actions_next_)) - alpha * log_prob_next
            target_q_value = self.args.reward_scale * rewards + inverse_dones * self.args.gamma * target_q_value_next 
        qf1_loss = (q1_value - target_q_value).pow(2).mean()
        qf2_loss = (q2_value - target_q_value).pow(2).mean()
        # qf1
        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        self.qf1_optim.step()
        # qf2
        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        self.qf2_optim.step()
        # policy loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha.item(), alpha_loss.item()
    
  

    def get_intrinsic_reward(self, obs, a, obs_next, clip_max=0.8, scale=1):
        delta = obs_next - obs
        obs, delta = self._preproc_og(obs, delta)
        obs_norm = torch.tensor(self.o_norm.normalize(obs))
        delta_norm = self.delta_norm.normalize(delta)
        
        delta_pred = self.dynamics_model(obs_norm, torch.tensor(a, dtype=torch.float32))
        error = scale * np.mean(np.square(delta_pred.detach().numpy() - delta_norm), axis=-1)
        ri = np.expand_dims(np.clip(error, 0, clip_max), axis=-1)
        return ri
    
    # update the target network
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


    def sample_difficulty(self, d1_prob=0.2, d2_prob=0.1, d3_prob=0.7):
        difficulty = np.random.choice([1,2,3], p=[d1_prob,d2_prob,d3_prob])
        return difficulty

    # evaluate the agent
    def _evaluate_agent(self):
        total_reward = 0
        for _ in range(self.args.eval_episodes):
            observation = self.env.reset(difficulty=self.sample_difficulty())
            obs = observation['observation']
            g = observation['desired_goal']
            episode_reward = 0 
            while True:
                with torch.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)
                    pi = self.actor_net(obs_tensor)
                    action = get_action_info(pi, cuda=self.args.cuda).select_actions(exploration=False, reparameterize=False)
                    action = action.detach().cpu().numpy()[0]
                # input the action into the environment
                obs_, reward, done, _ = self.env.step(self.action_max * action)
                episode_reward += reward
                if done:
                    break
                obs = obs_
            total_reward += episode_reward
        return total_reward / self.args.eval_episodes
