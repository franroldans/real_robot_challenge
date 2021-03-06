import numpy as np
import torch
from rrc_example_package.her.rl_modules.models import flatten_mlp, tanh_gaussian_actor
from rrc_example_package.her.rl_modules.replay_buffer import replay_buffer
from rrc_example_package.her.her_modules.her import her_sampler
from rrc_example_package.her.utils import get_action_info
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
        torch.autograd.set_detect_anomaly(True)
        global_timesteps = 0
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._collect_exp() 

        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.train_loop_per_epoch):
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # for each epoch, it will reset the environment
                    # reset the environment
                    observation = self.env.reset(difficulty=self.sample_difficulty())
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    for t in range(self.env_params['max_timesteps']):
                        # start to collect samples
                        with torch.no_grad():
                            obs_tensor = self._get_tensor_inputs(obs)
                            pi = self.actor_net(obs_tensor)
                            action = get_action_info(pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                            action = action.cpu().numpy()[0]
                        # input the actions into the environment
                        observation_new, reward, done, _ = self.env.step(self.action_max * action)
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
                #mean_rewards = self._evaluate_agent()
                mean_rewards = self._eval_agent()
                print('[{}] Epoch: {} / {}, Frames: {}, Rewards: {:.3f}, QF1: {:.3f}, QF2: {:.3f}, AL: {:.3f}, Alpha: {:.5f}, AlphaL: {:.5f}'.format(datetime.now(), \
                            epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length, mean_rewards, qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss))
                # save models
                torch.save(self.actor_net.state_dict(), self.model_path + '/model.pt')
    
    # do the initial exploration by using the uniform policy
    def _initial_exploration(self, exploration_policy='gaussian'):
        # get the action information of the environment
        obs = self.env.reset(difficulty=self.sample_difficulty())
        for _ in range(self.args.init_exploration_steps):
            if exploration_policy == 'uniform':
                raise NotImplementedError
            elif exploration_policy == 'gaussian':
                # the sac does not need normalize?
                with torch.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs["observation"])
                    # generate the policy
                    pi = self.actor_net(obs_tensor)
                    action = get_action_info(pi).select_actions(reparameterize=False)
                    action = action.cpu().numpy()[0]
                # input the action input the environment
                obs_, reward, done, _ = self.env.step(self.action_max * action)
                # store the episodes
                self.buffer.add(obs, action, reward, obs_, float(done))
                obs = obs_
                if done:
                    # if done, reset the environment
                    obs = self.env.reset(difficulty=self.sample_difficulty())
        print("Initial exploration has been finished!")
    # get tensors
    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor
    
    # update the network
    def _update_newtork(self):
        # smaple batch of samples from the replay buffer
        transitions = self.buffer.sample(self.args.batch_size)

        # Add intrinsic reward
        """r_intrinsic = self.get_intrinsic_reward(obses, actions, obses_)
        transitions['r'] += r_intrinsic
        ri = np.mean(r_intrinsic)"""
        # preprocessing the data into the tensors, will support GPU later
        obses = torch.tensor(transitions['obs'], dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        actions = torch.tensor(transitions['actions'], dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        rewards = torch.tensor(transitions['r'], dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        obses_ = torch.tensor(transitions['obs_next'], dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - transitions['r'], dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        # start to update the actor network
        pis = self.actor_net(obses)
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
        q_actions_ = torch.min(self.qf1(obses, actions_), self.qf2(obses, actions_))
        actor_loss = (alpha * log_prob - q_actions_).mean()
        
        # policy loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # q value function loss
        q1_value = self.qf1(obses, actions)
        q2_value = self.qf2(obses, actions)
        with torch.no_grad():
            pis_next = self.actor_net(obses_)
            actions_info_next = get_action_info(pis_next, cuda=self.args.cuda)
            actions_next_, pre_tanh_value_next = actions_info_next.select_actions(reparameterize=True)
            log_prob_next = actions_info_next.get_log_prob(actions_next_, pre_tanh_value_next)
            target_q_value_next = torch.min(self.target_qf1(obses_, actions_next_), self.target_qf2(obses_, actions_next_)) - alpha * log_prob_next
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
       
        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha.item(), alpha_loss.item()
    
    # update the target network
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


    def sample_difficulty(self, d1_prob=0.5, d2_prob=0.5, d3_prob=0.0):
        difficulty = np.random.choice([1,2,3], p=[d1_prob,d2_prob,d3_prob])
        return difficulty

    # evaluate the agent
    def _evaluate_agent(self):
        total_reward = 0
        for _ in range(self.args.eval_episodes):
            observation = self.env.reset(difficulty=self.sample_difficulty())
            obs = observation['observation']
            #print(obs)
            #print(type(obs))
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
                obs = obs_["observation"]
            total_reward += episode_reward
        return total_reward / self.args.eval_episodes


    def _collect_exp(self, rollouts=100, difficulty=1):
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for _ in range(rollouts):
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environment
            observation = self.env.reset(difficulty=self.sample_difficulty())
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            # start to collect samples
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._get_tensor_inputs(obs)
                    pi = self.actor_net(input_tensor)
                    action = get_action_info(pi).select_actions(reparameterize=False)
                    action = action.cpu().numpy()[0]
                    
                # feed the actions into the environment
                observation_new, _, _, info = self.env.step(self.action_max * action)
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
        #self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
        

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset(difficulty=self.sample_difficulty())
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._get_tensor_inputs(obs)
                    pi = self.actor_net(input_tensor)
                    # convert the actions
                    action = get_action_info(pi).select_actions(reparameterize=False)
                    action = action.cpu().numpy()[0]
                observation_new, _, _, info = self.env.step(self.action_max *actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()


    """def get_intrinsic_reward(self, obs, a, obs_next, clip_max=0.8, scale=1):
        delta = obs_next - obs
        obs, delta = self._preproc_og(obs, delta)
        obs_norm = torch.tensor(self.o_norm.normalize(obs))
        delta_norm = self.delta_norm.normalize(delta)
        
        delta_pred = self.dynamics_model(obs_norm, torch.tensor(a, dtype=torch.float32))
        error = scale * np.mean(np.square(delta_pred.detach().numpy() - delta_norm), axis=-1)
        ri = np.expand_dims(np.clip(error, 0, clip_max), axis=-1)
        return r"""
