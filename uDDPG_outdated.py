import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import copy
import pickle
import time

import random
from collections import deque
import math



option = 1
start_time = 5000

if option == 1:
    env = gym.make('BipedalWalker-v3')
    env_test = gym.make('BipedalWalker-v3', render_mode="human")
    variable_steps = False
    limit_steps = 10000

elif option == 2:
    env = gym.make('BipedalWalkerHardcore-v3')
    env_test = gym.make('BipedalWalkerHardcore-v3', render_mode="human")
    variable_steps = True
    limit_steps = 70


# 4 random seeds
#r1 = random.randint(0,2**32-1)
#r2 = random.randint(0,2**32-1)
r1, r2 = 3684114277, 3176436971
r1, r2 = 1375924064, 4183630884
r1, r2 = 1495103356, 3007657725
r1, r2 = 830143436, 167430301
print(r1, ", ", r2)
torch.manual_seed(r1)
np.random.seed(r2)


#used to create random seeds in Actor -> less dependendance on the specific neural network random seed.
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


#Rectified Hubber Error Loss Function, stabilizes the learning speed
def ReHE(error):
    ae = torch.abs(error).mean()
    return ae*torch.tanh(ae)

#Rectified Hubber Assymetric Error Loss Function, stabilizes the learning speed
def ReHAE(error):
    e = error.mean()
    return torch.abs(e)*torch.tanh(e)



# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=32, max_action=1.0):
        super(Actor, self).__init__()
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
         )
        
        self.max_action = torch.mean(max_action).item()
        self.eps = 3.0
        self.x_coor = 0.0

    def accuracy(self):
        if self.eps>1e-3:
            self.eps = 3.0*self.max_action * math.exp(-self.x_coor) + 0.03
            self.x_coor += 3e-5
            return True
        return False

    def forward(self, state, mean=False):
        x = self.max_action*self.net(state)
        if mean: return x
        if self.accuracy(): x += torch.normal(torch.zeros_like(x), self.eps)
        return x.clamp(-1.0, 1.0)

        
        
# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        return self.net(x)



# we use cache to append transitions, and then update buffer to collect Q Returns, purging cache at the episode end.
class ReplayBuffer:
    def __init__(self, device, n_steps, capacity=300*2560):
        self.buffer, self.capacity, self.length =  deque(maxlen=capacity), capacity, 0 #buffer is prioritised limited memory
        self.indices, self.indexes, self.probs = [], np.array([]), [] #for priorities
        self.cache = [] #cache is episodic memory
        self.device = device
        self.n_steps = n_steps
        self.random = np.random.default_rng()
        self.batch_size = min(max(128, self.length//300), 2560) #in order for sample to describe population


    def add(self, transition):
        self.cache.append(transition)

    def purge(self):
        self.cache = []

    # Monte-Carlo calculation of Return, from step k > forward
    # As we neglect done (experiment continues to gather all discounted n_step rewards),
    # we need to gather "roll-out" for each k step.

    def update(self):
        delta = len(self.cache) - self.n_steps # e.g. for cache = 300, n_steps = 100: 300 - 100 = 200 active steps
        if delta>0:
            for t, (state, action, reward, next_state) in enumerate(self.cache): # t =[0 .. 299]
                if t<delta: # if t<200 [0..199]
                    Return = 0.0
                    discount = 1.0
                    for k in range(t, t+self.n_steps): # k = [0..99], [1..100], [2..101] ... [199..298]
                        Return += discount* self.cache[k][2] #  Bellman Equation: r_0 + 0.99*r_1 + (0.99^2)*r_2 + ... (reward_k = self.cache[k][2])
                        discount *= 0.99
                    
                    Return = math.tanh(Return) # this makes tail less important, Return is in [-1.0, 1.0]
                    self.store([state, action, reward, Return, next_state])
                else:
                    break
        self.cache =  self.cache[-self.n_steps:]

    
    def store(self, transition):
        self.buffer.append(transition)
       

        if self.length < self.capacity:
            self.length = len(self.buffer)
            self.batch_size = min(max(128, self.length//300), 2560)

            #updating priorities: less priority for older data
            self.indices.append(self.length-1)
            self.indexes = np.array(self.indices)
            if self.length>1:
                #priorities are normalized between 0 and 1 and squashed by 0.01
                #weights = 0.01*(self.indexes/self.length) #linear /
                weights =  0.01*self.fade(self.indexes/self.length) #non-linear # ▁/▔
                self.probs = weights/np.sum(weights)



    def sample(self):
        batch_indices = self.random.choice(self.indexes, p=self.probs, size=self.batch_size)
        batch = [self.buffer[indx-1] for indx in batch_indices]
        states, actions, rewards, Return, next_states = map(np.vstack, zip(*batch))

        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(Return).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
        )
    

    def __len__(self):
        return len(self.buffer)



# Define the actor-critic agent
class uDDPG(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0):

        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action=max_action).to(device)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=7e-4)
        self.critic_target_optimizer = optim.Adam(self.critic_target.parameters(), lr=7e-4)

        self.max_action = max_action
        self.device = device


    def select_action(self, state, mean=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(-1,state.shape[-1]).to(self.device)
            action = self.actor(state, mean=mean)
        return action.cpu().data.numpy().flatten()


    def train(self, i):
        if i%2==0:
            algo.mc(replay_buffer.sample())
        else:
            algo.td(replay_buffer.sample())

    # Monte-Carlo update
    def mc(self, batch):
        if batch != None:
            state, action, _, Return, _ = batch
            self.critic_direct(state, action, Return)
            self.actor_update(state)
            self.sync_target()
           
    # Temporal Difference update
    def td(self, batch):
        if batch != None:
            state, action, reward, _, next_state = batch
            self.critic_update(state, action, reward, next_state)
            self.actor_update(state)


    def critic_direct(self, state, action, q_Return):

        q = self.critic(state, action)
        critic_loss = ReHE(q_Return - q) #ReHE instead of MSE

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def critic_update(self, state, action, reward, next_state): 

        with torch.no_grad():
            next_action = self.actor(next_state, mean=True)
            q_next_target = self.critic_target(next_state, next_action)
            q_value = reward +  0.99 * q_next_target

        q = self.critic(state, action)
        critic_loss = ReHE(q_value - q) #ReHE instead of MSE

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def actor_update(self, state):

        action = self.actor(state, mean=True)
        q_new_policy = self.critic(state, action)
        actor_loss = -q_new_policy
        actor_loss = ReHAE(actor_loss)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def sync_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count())
        
print(device)





state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]

hidden_dim = 256
n_steps = 200

print('action space high', env.action_space.high)

max_action = torch.FloatTensor(env.action_space.high).to(device) if env.action_space.is_bounded() else 1.0
replay_buffer = ReplayBuffer(device, n_steps)
algo = uDDPG(state_dim, action_dim, hidden_dim, device, max_action)

num_episodes, total_rewards, total_steps, test_rewards, policy_training = 1000000, [], [], [], False


try:
    print("loading buffer...")
    with open('replay_buffer', 'rb') as file:
        dict = pickle.load(file)
        replay_buffer = dict['buffer']
        algo.actor.eps = dict['eps']
        algo.actor.x_coor = dict['x_coor']
        limit_steps = dict['limit_steps']
        total_steps = dict['total_steps']
        if len(replay_buffer)>=start_time and not policy_training: policy_training = True
    print('buffer loaded, buffer length', len(replay_buffer))

except:
    print("problem during loading buffer")

#load existing models

try:
    print("loading models...")
    algo.actor.load_state_dict(torch.load('actor_model.pt'))
    algo.critic.load_state_dict(torch.load('critic_model.pt'))
    algo.critic_target.load_state_dict(torch.load('critic_target_model.pt'))

    print('models loaded')

    #--------------------testing loaded model-------------------------

    for test_episode in range(3):
        state = env_test.reset()[0]
        rewards = []

        for steps in range(1,limit_steps+n_steps):
            action = algo.select_action(state, mean=True)
            next_state, reward, done, info , _ = env_test.step(action)
            state = next_state
            rewards.append(reward)
            if done: break
        print(f"Validation, Rtrn = {np.sum(rewards):.2f}, buffer len {len(replay_buffer)}")

except:
    print("problem during loading models")





for i in range(num_episodes):
    done_steps, rewards, terminal_reward, stop = 0, [], 0.0, False
    state = env.reset()[0]
    replay_buffer.purge()
    

    #---------------------------1. processor releave --------------------------
    #---------------------2. decreases dependence on random seed: ---------------
    #-----------3. slighlty random initial configuration as in OpenAI Pendulum----
    #-----------prevents often appearance of the same transitions in buffer-------

    #1
    if policy_training: time.sleep(1.0)
    #2
    if not policy_training and len(replay_buffer.buffer)<start_time:
        algo.actor.apply(init_weights)
    #3
    action = 0.3*max_action.to('cpu').numpy()*np.random.uniform(-1.0, 1.0, size=action_dim)

    for t in range(0,2):
        next_state, reward, done, info, _ = env.step(action)
        state = next_state
        rewards.append(reward)
    

    #------------------------------training------------------------------

    for steps in range(1, 1000000):

        if len(replay_buffer.buffer)>=start_time and not policy_training:
            print("started training")
            policy_training = True


        action = algo.select_action(state, mean=False)
        next_state, reward, done, info, _ = env.step(action)
        if steps>=limit_steps: stop = True

        if done or stop:
            if done: terminal_reward = reward
            if abs(terminal_reward) >=50: reward = terminal_reward/n_steps
            done_steps += 1


        rewards.append(reward)
        replay_buffer.add([state, action, reward/n_steps, next_state]) # we crudely bring reward closer to 0.0
        replay_buffer.update()
        state = next_state
        
        
        if policy_training: algo.train(steps)
        if done_steps > n_steps: break
            

    replay_buffer.update()
    total_rewards.append(np.sum(rewards))
    average_reward = np.mean(total_rewards[-100:])

    episode_steps = steps-n_steps
    total_steps.append(episode_steps)
    average_steps = np.mean(total_steps[-100:])
    if policy_training and variable_steps: limit_steps = int(average_steps) + 5 + int(0.05*average_steps)


    print(f"Ep {i}: Rtrn = {total_rewards[i]:.2f}, eps = {algo.actor.eps:.2f} | ep steps = {episode_steps}")



    if policy_training:

        #--------------------saving-------------------------
        if (i>=100 and i%100==0): 
            torch.save(algo.actor.state_dict(), 'actor_model.pt')
            torch.save(algo.critic.state_dict(), 'critic_model.pt')
            torch.save(algo.critic_target.state_dict(), 'critic_target_model.pt')
            #print("saving... len = ", len(replay_buffer), end="")
            with open('replay_buffer', 'wb') as file:
                pickle.dump({'buffer': replay_buffer, 'eps': algo.actor.eps, 'x_coor': algo.actor.x_coor, 'limit_steps':limit_steps, 'total_steps':total_steps}, file)
            #print(" > done")


        #-----------------validation-------------------------

        if total_rewards[i]>=330 or (i>=500 and i%500==0):
            test_episodes = 1000 if total_rewards[i]>=330 else 5
            env_val = env if test_episodes == 1000 else env_test
            print("Validation... ", test_episodes, " epsodes")
            test_rewards = []

            for test_episode in range(test_episodes):
                done_steps, rewards, terminal_reward, stop = 0, [], 0.0, False

                state = env_val.reset()[0]
                
                for steps in range(1,1000000):
                    action = algo.select_action(state, mean=True)
                    next_state, reward, done, info , _ = env_val.step(action)
                    if steps>=limit_steps: stop = True

                    if done or stop:
                        if done: terminal_reward = reward
                        if abs(terminal_reward) >=50: reward = terminal_reward/n_steps
                        done_steps += 1

                    rewards.append(reward)
                    state = next_state
                    if done_steps>n_steps: break
                    

                test_rewards.append(np.sum(rewards))

                validate_reward = np.mean(test_rewards[-100:])
                print(f"trial {test_episode}:, Rtrn = {test_rewards[test_episode]:.2f}, Average 100 = {validate_reward:.2f}")

                if test_episodes==1000 and validate_reward>=300: print("Average of 100 trials = 300 !!!CONGRATULATIONS!!!")
                    

        #====================================================
