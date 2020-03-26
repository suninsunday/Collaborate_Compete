import numpy as np
import random
import copy
from collections import namedtuple, deque
from copy import deepcopy

from ddpg_agent import DDPGAgent

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(2e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
USE_BATCH_NORM = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Note that much of the source code is derived from the ddpg_agent.py
# file provided by the Udacity DRL github.

class MADDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, 
            numAgents, 
            state_size, 
            action_size, 
            random_seed, 
            batch_size = BATCH_SIZE, 
            buffer_size = BUFFER_SIZE, 
            use_batch_norm = USE_BATCH_NORM):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.agents = [DDPGAgent(i, numAgents, state_size, action_size, random_seed, use_batch_norm) for i in range(numAgents)]
        self.numAgents = numAgents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

        self.batch_size = batch_size

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Input shapes
        # states is 2x24
        # next_states is 2x24
        # actions is 1x4
        # rewards is 2x1
        # dones is 2x1
        flat_states = states.reshape(1,-1) # now 1x48
        flat_next_states = next_states.reshape(1,-1) # now 1x48

        # add a single entry into the replay buffer
        self.memory.add(flat_states, actions, rewards, flat_next_states, dones)

        # Learn, if enough samples are available in memory
        
        if len(self.memory) > self.batch_size:
            # for each agent, select a unique mini-batch of tuples and then proceed to learn
            experiences = [self.memory.sample() for a in self.agents]
            self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        # 2 agents, 2 observations. Agents take actions on only their own observations
        actions = [a.act(s, True) for a,s in zip(self.agents, state)]
        # 2 agents with 2 controls per agent, reshape to 1x4
        return np.array(actions).reshape(1,-1)

    def reset_noise(self):
        [a.reset() for a in self.agents]

    def learn(self, experiences, gamma):
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
        combined_next_actions = []
        for i, agent in enumerate(self.agents):
            _, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            # next_state is 1x48 - reshape to 2x24 then extract the row with row number agent_id and compress that back to 1x24
            # this is done so that each actor is taking action upon its own observation
            next_state = next_states.reshape(-1, self.numAgents, self.state_size).index_select(1, agent_id).squeeze(1)

            # run the next state for this agent through the target actor to get the action
            combined_next_actions.append(agent.actor_target(next_state))

        combined_current_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, _ , _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            # state is 1x48 - reshape to 2x24 then extract the row with row number agent_id and compress that back to 1x24
            # this is done so that each actor is taking action upon its own observation
            state = states.reshape(-1, self.numAgents, self.state_size).index_select(1, agent_id).squeeze(1)
            # run the next state for this agent through the target actor to get the action
            combined_current_actions.append(agent.actor_local(state))
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, combined_next_actions, combined_current_actions)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
