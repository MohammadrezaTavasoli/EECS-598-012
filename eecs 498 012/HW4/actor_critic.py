#!/usr/bin/env python
# coding: utf-8

# In[19]:


# %load actor_critic.py
import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
get_ipython().run_line_magic('matplotlib', 'inline')

#parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
#parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                    help='discount factor (default: 0.99)')
#parser.add_argument('--seed', type=int, default=543, metavar='N',
#                    help='random seed (default: 543)')
#parser.add_argument('--render', action='store_true',
 #                   help='render the environment')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
 #                   help='interval between training status logs (default: 10)')
#args = parser.parse_args()

gamma=0.99
seed=543
render=True
log_interval=10

env = gym.make('CartPole-v0')
env.seed(seed)
torch.manual_seed(seed)




class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        ##### TODO ######
        ### Complete definition 
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        ##### TODO ######
        ### Complete definition 
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    return action

def sample_episode():

    state, ep_reward = env.reset(), 0
    episode = []

    for t in range(1, 10000):  # Run for a max of 10k steps

        action = select_action(state)

        # Perform action
        next_state, reward, done, _ = env.step(action.item())

        episode.append((state, action, reward))
        state = next_state

        ep_reward += reward

        if render:
            env.render()

        if done:
            break

    return episode, ep_reward

def compute_losses(episode):

    ####### TODO #######
    #### Compute the actor and critic losses
   # actor_loss, critic_loss = None, None
    R = 0
  #  saved_actions = episode.saved_actions
    actor_loss = []
    critic_loss = []
    returns = []
    for s, a, r in episode[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    for (state, action, reward), R in zip(episode, returns):
        prob,value = model.forward(torch.FloatTensor(state))
        m = Categorical(prob)
        advantage = R - value.item()
        actor_loss.append(-m.log_prob(action) * advantage)
        critic_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))

    return actor_loss, critic_loss


def main():
    running_reward = 10
    averagereward=[]
    for i_episode in count(1):

        episode, episode_reward = sample_episode()

        optimizer.zero_grad()
        
        

        actor_loss, critic_loss = compute_losses(episode)
        
        #loss = actor_loss + critic_loss

        loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
        loss.backward()

        optimizer.step()
        #print(running_reward)

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        averagereward.append(running_reward)
        #print(averagereward)
        plt.plot(averagereward, label="Average reward")
        #plt.show()
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, episode_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, len(episode)))
            break


if __name__ == '__main__':
    main()


# In[12]:


averagereward


# In[ ]:




