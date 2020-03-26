import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.   # L2 weight decay
USE_BATCH_NORM = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Note that much of the source code is derived from the ddpg_agent.py
# file provided by the Udacity DRL github.

class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, 
            agent_id,
            numAgents, 
            state_size, 
            action_size, 
            random_seed, 
            use_batch_norm = USE_BATCH_NORM):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.agent_id = agent_id
        self.numAgents = numAgents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_loss = 0
        self.critic_loss = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, use_batch_norm=use_batch_norm).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, use_batch_norm=use_batch_norm).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, numAgents, random_seed, use_batch_norm=use_batch_norm).to(device)
        self.critic_target = Critic(state_size, action_size, numAgents, random_seed, use_batch_norm=use_batch_norm).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        #self.noise = OUNoise(action_size, random_seed)
        self.noise = OUNoise(action_size, random_seed)
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def samples(self):
        return self.noise.sample_count()

    def get_noise_scale(self):
        return self.noise.get_noise_scale()

    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        self.critic_optimizer.zero_grad()

        agent_id = torch.tensor([agent_id]).to(device)

        reward = rewards.index_select(1, agent_id)
        done = dones.index_select(1, agent_id)
     
        # ---------------------------- update critic -------------------------- #
        # flatten the actions again
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = reward + (gamma * Q_targets_next * (1.0 - done))
        Q_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # print("critic_loss {}".format(critic_loss))
        self.critic_loss = critic_loss.item()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()


        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [actions if i == self.agent_id else actions.detach() for i, actions in enumerate(all_actions)]
        #actions_pred = torch.cat(actions_pred, dim=1).to(device)
        self.actor_optimizer.zero_grad()
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # print("actor_loss {}".format(actor_loss))
        # Minimize the loss
        self.actor_loss = actor_loss.item()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class NormalNoise:
    def __init__(self, size, seed, scale=0.5, min_scale=0.2, decay_rate=1.0, exploration_samples=5000):
        self.size = size
        self.scale = scale
        self.min_scale = min_scale
        self.samples = 0
        self.decay_rate = decay_rate
        self.seed = random.seed(seed)
        self.exploration_samples = exploration_samples

    def reset(self):
        pass

    def sample_count(self):
        return self.samples

    def get_noise_scale(self):
        return self.scale

    def sample(self):
        self.samples += 1
        if self.samples >= self.exploration_samples:
            self.scale *= self.decay_rate
            self.scale = max(self.scale, self.min_scale)

        return self.scale * np.array([np.random.randn() for i in range(self.size)])

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

