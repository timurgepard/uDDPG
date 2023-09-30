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

from actor_critic import Actor, Critic
from buffer import ReplayBuffer

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
           
    # Tempora Difference update
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


#env = gym.make('BipedalWalker-v3')
#env_test = gym.make('BipedalWalker-v2', render_mode="human")

env = gym.make('BipedalWalkerHardcore-v3')
env_test = gym.make('BipedalWalkerHardcore-v3', render_mode="human")




state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]

hidden_dim = 256

n_steps = 200

print('action space high', env.action_space.high)

max_action = torch.FloatTensor(env.action_space.high).to(device) if env.action_space.is_bounded() else 1.0
replay_buffer = ReplayBuffer(device, n_steps)
algo = uDDPG(state_dim, action_dim, hidden_dim, device, max_action)

num_episodes, counter, total_rewards, total_steps, test_rewards, policy_training = 1000000, 0, [], [], [], False


#load existing models

try:
    print("loading models...")
    algo.actor.load_state_dict(torch.load('actor_model.pt'))
    algo.critic.load_state_dict(torch.load('critic_model.pt'))
    algo.critic_target.load_state_dict(torch.load('critic_target_model.pt'))

    print('models loaded')

    #--------------------testing loaded model-------------------------

    for test_episode in range(0):
        state = env_test.reset()[0]
        rewards = []

        for steps in range(1,1000):
            action = algo.select_action(state, mean=True)
            next_state, reward, done, info , _ = env_test.step(action)
            state = next_state
            rewards.append(reward)
            if done: break
        print(f"Validation, Rtrn = {np.sum(rewards):.2f}, buffer len {len(replay_buffer)}")

except:
    print("problem during loading models")

try:
    print("loading buffer...")
    with open('replay_buffer', 'rb') as file:
        dict = pickle.load(file)
        replay_buffer = dict['buffer']
        algo.actor.x_coor = dict['x_coor']
        if len(replay_buffer)>=2056 and not policy_training: policy_training = True
    print('buffer loaded, buffer length', len(replay_buffer))

except:
    print("problem during loading buffer")




for i in range(num_episodes):
    #processor releave
    if policy_training: time.sleep(1.0)
    state, Return = env.reset()[0], 0.0
    replay_buffer.purge()
    done_steps, rewards, terminal_reward = 0, [], 100.0


    #-------------------decreases dependence on random seed: ------------------
    if not policy_training and len(replay_buffer.buffer)<5000:
        algo.actor.apply(init_weights)


    #-------------slighlty random initial configuration as in OpenAI Pendulum-------------
    action = 0.3*max_action.to('cpu').numpy()*np.random.uniform(-1.0, 1.0, size=action_dim)

    for t in range(0,2):
        next_state, reward, done, info, _ = env.step(action)
        state = next_state
        rewards.append(reward)

    #------------------------training-------------------------



    for steps in range(10000):

        if len(replay_buffer.buffer)>=5000 and not policy_training:
            print("started training")
            policy_training = True

        action = algo.select_action(state, mean=False)
        next_state, reward, done, info, _ = env.step(action)
        

        if done:
            if done_steps==0: terminal_reward = reward
            if abs(terminal_reward) >=50: reward = terminal_reward/n_steps
            done_steps += 1


        rewards.append(reward)
        replay_buffer.add([state, action, reward/n_steps, next_state]) # we crudely bring reward closer to 0.0
        replay_buffer.update()
        state = next_state
        
        
        if policy_training: algo.train(i)
        if done_steps > n_steps: break
            

        

    replay_buffer.update()
    total_rewards.append(np.sum(rewards))
    average_reward = np.mean(total_rewards[-100:])


    print(f"Ep {i}: Rtrn = {total_rewards[i]:.2f}, eps = {algo.actor.std:.2f} | ep steps = {steps-n_steps}")
    #====================================================

    #--------------------saving-------------------------

    if policy_training:
            
        torch.save(algo.actor.state_dict(), 'actor_model.pt')
        torch.save(algo.critic.state_dict(), 'critic_model.pt')
        torch.save(algo.critic_target.state_dict(), 'critic_target_model.pt')
        #print("saving... len = ", len(replay_buffer), end="")
        with open('replay_buffer', 'wb') as file:
            pickle.dump({'buffer': replay_buffer, 'x_coor': algo.actor.x_coor}, file)
        #print(" > done")

        #====================================================

        #-----------------validation-------------------------

        if total_rewards[i]>=330 or (i>=100 and i%100==0):
            test_episodes = 1000 if total_rewards[i]>=330 else 10
            env_val = env if test_episodes == 1000 else env_test
            print("Validation... ", test_episodes, " episodes")
            test_rewards = []

            for test_episode in range(test_episodes):
                state = env_val.reset()[0]
                done_steps, terminal_reward = 0, 0.0
                for steps in range(1,1000):
                    action = algo.select_action(state, mean=True)
                    next_state, reward, done, info , _ = env_val.step(action)
                    state = next_state
                    
                    if done:
                        if done_steps==0: terminal_reward = reward
                        if abs(terminal_reward)>=50: reward = reward/n_steps
                        done_steps += 1

                    rewards.append(reward)
                    if done_steps>n_steps: break

                test_rewards.append(np.sum(rewards))

                validate_reward = np.mean(test_rewards[-100:])
                print(f"trial {test_episode}:, Rtrn = {test_rewards[test_episode]}, Average100 = {validate_reward:.2f}")

                if test_episodes==1000 and validate_reward>=300: print("Average of 100 trials = 300 !!!CONGRATULATIONS!!!")
                    

        #====================================================
