import copy
import pickle
import random
import gymnasium as gym
import torch
from collections import deque, namedtuple
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *


# Class for training an RL agent with Actor-Critic
class ACTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = ACAgent(env=self.env, params=self.params)
        self.actor_net = ActorNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.critic_net = CriticNet(input_size=self.env.observation_space.shape[0], output_size=1, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.actor_optimizer = Adam(params=self.actor_net.parameters(), lr=self.params['actor_lr'])
        self.critic_optimizer = Adam(params=self.critic_net.parameters(), lr=self.params['critic_lr'])
        self.trajectory = None

    def run_training_loop(self):
        list_ro_reward = list()
        for ro_idx in range(self.params['n_rollout']):
            self.trajectory = self.agent.collect_trajectory(
                policy=self.actor_net)
            self.update_critic_net()
            self.estimate_advantage()
            self.update_actor_net()
            # TODO: Calculate avg reward for this rollout
            # HINT: Add all the rewards from each trajectory. There should be "ntr" trajectories within a single rollout.
            list_rwrds = self.trajectory.get('reward')
            total_rwrd = 0
            for trajectory_reward_list in list_rwrds:
                total_rwrd += np.sum(trajectory_reward_list)
            total_rwrd = torch.tensor(total_rwrd)
            avg_ro_reward = (total_rwrd/len(list_rwrds)).item()
            print(
                f'End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')
            # Append average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def update_critic_net(self):
        torch.autograd.set_detect_anomaly(True)
        for critic_iter_idx in range(self.params['n_critic_iter']):
            self.update_target_value()
            
            for critic_epoch_idx in range(self.params['n_critic_epoch']):
                self.rerun_net()
                critic_loss = self.estimate_critic_loss_function()
                critic_loss.backward(retain_graph = True)
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()

    def update_target_value(self, gamma=0.99):
        # TODO: Update target values
        # HINT: Use definition of target-estimate from equation 7 of teh assignment PDF

        target_values = []
        rewards = self.trajectory['reward']
        observations = self.trajectory['obs']

        for idx in range(len(observations)):
            reward = rewards[idx]
            obs_tensor = observations[idx]
            next_obs = torch.cat([obs_tensor[1:].clone(), torch.zeros(1, obs_tensor.shape[1])], dim=0)
            rewards_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).detach()
            next_state_values_tensor = self.critic_net(next_obs)
            target_values_tensor = rewards_tensor + gamma * next_state_values_tensor.detach()
            target_values.append(target_values_tensor)

        self.trajectory['target_value'] = target_values

    def rerun_net(self):
        state_vals = []
        obs = self.trajectory['obs']

        for idx in range(len(obs)):
            state_values = self.critic_net(obs[idx])
            state_vals.append(state_values)
        self.trajectory['state_value'] = state_vals
         

    def estimate_advantage(self, gamma=0.99):
        # TODO: Estimate advantage
        # HINT: Use definition of advantage-estimate from equation 6 of teh assignment PDF
        #self.trajectory['advantage'] = [a - b for a, b in zip(self.trajectory['target_value'], self.trajectory['state_value'])]
        #Fetching the upated values from the Critic Model with updated weights
        advantage_values = list()
        self.rerun_net()
        state_vals = self.trajectory['state_value']
        target_vals = self.trajectory['target_value']
        self.update_target_value()
        for idx in range(len(state_vals)):
            advantage_vals_idx = list()
            target_vals_idx = target_vals[idx]
            state_vals_idx = state_vals[idx]
            for i in range(len(state_vals_idx)):
                adv = target_vals_idx[i] - state_vals_idx[i]
                advantage_vals_idx.append(adv.detach())
            advantage_values.append(advantage_vals_idx)
        self.trajectory['advantage'] = advantage_values


    def update_actor_net(self):
        actor_loss = self.estimate_actor_loss_function()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

    def estimate_critic_loss_function(self):
        # TODO: Compute critic loss function
        # HINT: Use definition of critic-loss from equation 7 of teh assignment PDF. It is the MSE between target-values and state-values.
        # Compute critic loss function
        # state_val = self.trajectory['state_value']
        # target_val = self.trajectory['target_value']
        # critic_loss = np.mean((target_val - state_val)**2)
        # return critic_loss


        # TODO: Compute critic loss function
        # HINT: Use definition of critic-loss from equation 7 of teh assignment PDF. It is the MSE between target-values and state-values.

        loss = torch.nn.MSELoss()
        target_val = self.trajectory['target_value']
        state_val = self.trajectory['state_value']
        critic_loss = torch.tensor(0.0, dtype=torch.float32, device=get_device())
        for tidx in range(len(state_val)):
            target_values_idx = target_val[tidx]
            state_values_idx = state_val[tidx]
            critic_loss = critic_loss + loss(state_values_idx, target_values_idx)
        return critic_loss


    def rerun_net(self):
        # Retrieve trajectory data
        observations = self.trajectory['obs']

        # Compute state values for each time step in the trajectory
        state_values = []
        for observation in observations:
            state_value = self.critic_net(observation)
            state_values.append(state_value)

        # Store state values in trajectory dictionary
        self.trajectory['state_value'] = state_values


    def estimate_actor_loss_function(self):
        actor_loss = list()
        log_prob_list = self.trajectory.get('log_prob')
        for t_idx in range(self.params['n_trajectory_per_rollout']):
            traj_log_prob = log_prob_list[t_idx]
            advantage = apply_discount([tensor.item() for  tensor in self.trajectory['advantage'][t_idx]])
            # TODO: Compute actor loss function
            loss = 0
            for idx in range(len(traj_log_prob)):
                loss = loss + traj_log_prob[idx]*advantage[idx]
            actor_loss.append(loss*-1)
        actor_loss = torch.stack(actor_loss).mean()
        return actor_loss
    

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action_idx, log_prob = self.actor_net(torch.tensor(obs, dtype=torch.float32, device=get_device()))
            obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


# CLass for actor-net
class ActorNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(ActorNet, self).__init__()
        # TODO: Define the actor net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        # TODO: Forward pass of actor net
        # HINT: (use Categorical from torch.distributions to draw samples and log-prob from model output)

        dis = Categorical(self.ff_net(obs))
        action_index = dis.sample()
        log_prob = dis.log_prob(action_index)

        return action_index, log_prob



# CLass for actor-net
class CriticNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(CriticNet, self).__init__()
        # Define the critic net
        # Use nn.Sequential to set up a 2 layer feedforward neural network.
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, obs):
        # TODO: Forward pass of critic net
        # HINT: (get state value from the network using the current observation)
        state_value = self.ff_net(obs)
        return state_value


# Class for agent
class ACAgent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [action for action in range(self.env.action_space.n)]

    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
            while True:
                obs = torch.tensor(obs, dtype=torch.float32, device=get_device())
                # Save observation
                trajectory_buffer['obs'].append(obs)
                action_idx, log_prob = policy(obs)
                obs, reward, terminated, truncated, info = self.env.step(self.action_space[action_idx.item()])
                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        return rollout_buffer

    # Converts a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['obs'].append(torch.stack(trajectory_buffer['obs']))
            serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer


class DQNTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.q_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = self.params['init_epsilon']
        self.optimizer = Adam(params=self.q_net.parameters(), lr=self.params['lr'])
        self.replay_memory = ReplayMemory(capacity=self.params['rm_cap'])

    def run_training_loop(self):
        list_ep_reward = list()
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        for idx_episode in range(self.params['n_episode']):
            ep_len = 0
            while True:
                ep_len += 1
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    self.epsilon = max(self.epsilon*self.params['epsilon_decay'], self.params['min_epsilon'])
                    next_obs = None
                    self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                    list_ep_reward.append(ep_len)
                    print(f'End of episode {idx_episode} with epsilon = {self.epsilon: 0.2f} and reward = {ep_len}, memory = {len(self.replay_memory.buffer)}')
                    obs, _ = self.env.reset()
                    break
                self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                obs = copy.deepcopy(next_obs)
                self.update_q_net()
                self.update_target_net()
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ep_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def get_action(self, obs):
        # TODO: Implement the epsilon-greedy behavior
        # HINT: The agent will will choose action based on maximum Q-value with
        # '1-ε' probability, and a random action with 'ε' probability. 
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()  
        else:
            with torch.no_grad():
                q_values = self.q_net(torch.Tensor(obs).to(get_device()))  # get Q-values from the Q-network
                action = q_values.argmax().item()  # choose the action with the maximum Q-value

        return action

    def update_q_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        
        # Sample a batch of transitions from replay memory

        states, acts, rwrds, nxt_states, nt_terms = self.replay_memory.sample(self.params['batch_size'])
        
        # Convert batch to tensors and move to device
        states = torch.tensor(np.array(states), dtype=torch.float32)
        rwrds = torch.tensor(rwrds, dtype=torch.float32)
        acts = torch.tensor(acts, dtype=torch.long).unsqueeze(1)
        nt_terms = torch.tensor(nt_terms, dtype=torch.bool)
        nxt_states = [state for state in nxt_states if state is not None]
        
        # Compute predicted Q-values for the current states and actions
        pred_state_val = self.q_net(states).gather(1, acts)

        # Compute target Q-values for the next states
        with torch.no_grad():
            nxt_state_vals = torch.zeros(self.params['batch_size'])
            nxt_state_vals[nt_terms] = self.target_net(torch.tensor(np.array(nxt_states), dtype=torch.float32).to(get_device())).max(1)[0].detach()
            target_value = rwrds + self.params['gamma'] * nxt_state_vals
        
        # Compute the Huber loss between predicted and target Q-values
        criterion = nn.SmoothL1Loss()
        q_loss = criterion(pred_state_val, target_value.unsqueeze(1))
        # Update the Q-network weights
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()


    def update_target_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        q_net_state_dict = self.q_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = self.params['tau']*q_net_state_dict[key] + (1 - self.params['tau'])*target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        self.epsilon = 0.0
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action = self.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(tuple(args))

    def sample(self, n_samples):
        batch = random.sample(self.buffer, n_samples)
        return zip(*batch)


class QNet(nn.Module):
    # TODO: Define Q-net
    # This is identical to policy network from HW1
    def __init__(self, input_size, output_size, hidden_dim):
        super(QNet, self).__init__()
        self.ff_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, obs):
        return self.ff_net(obs)

