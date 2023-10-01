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


option = 1

if option == 1:
    env = gym.make('BipedalWalker-v3')
    env_test = gym.make('BipedalWalker-v3', render_mode="human")
    limit_steps = 10000

elif option == 2:
    env = gym.make('BipedalWalkerHardcore-v3')
    env_test = gym.make('BipedalWalkerHardcore-v3', render_mode="human")
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

        S2 = torch.var(q_Return)
        q, s2 = self.critic(state, action)
        critic_loss = ReHE(q_Return - q) + ReHE(S2-s2) #ReHE instead of MSE

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def critic_update(self, state, action, reward, next_state): 

        with torch.no_grad():
            next_action = self.actor(next_state, mean=True)
            q_next_target, s2_next = self.critic_target(next_state, next_action)
            q_value = reward +  0.99 * q_next_target
            s2_value = torch.var(reward) + 0.99*s2_next

        q, s2 = self.critic(state, action)
        critic_loss = ReHE(q_value - q) + ReHE(s2_value-s2) #ReHE instead of MSE

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def actor_update(self, state):

        action = self.actor(state, mean=True)
        q_new_policy, s2_new_policy = self.critic(state, action)
        actor_loss = torch.exp(-q_new_policy - s2_new_policy)
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


#load existing models

try:
    print("loading models...")
    algo.actor.load_state_dict(torch.load('actor_model.pt'))
    algo.critic.load_state_dict(torch.load('critic_model.pt'))
    algo.critic_target.load_state_dict(torch.load('critic_target_model.pt'))

    print('models loaded')

    #--------------------testing loaded model-------------------------

    for test_episode in range(10):
        state = env_test.reset()[0]
        rewards = []

        for steps in range(1,2000):
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
        algo.actor.eps = dict['eps']
        algo.actor.x_coor = dict['x_coor']
        limit_steps = dict['limit_steps']
        total_steps = dict['total_steps']
        if len(replay_buffer)>=2056 and not policy_training: policy_training = True
    print('buffer loaded, buffer length', len(replay_buffer))

except:
    print("problem during loading buffer")




for i in range(num_episodes):
    done_steps, rewards, terminal_reward, stop = 0, [], 0.0, False
    state = env.reset()[0]
    replay_buffer.purge()
    

    #---------------------------1. processor releave --------------------------
    #---------------------2. decreases dependence on random seed: ---------------
    #-----------3. slighlty random initial configuration as in OpenAI Pendulum----
    #-----------prevents often appearance of the same transitions in buffer-------

    #1
    if policy_training: time.sleep(2.0)
    #2
    if not policy_training and len(replay_buffer.buffer)<5000:
        algo.actor.apply(init_weights)
    #3
    action = 0.3*max_action.to('cpu').numpy()*np.random.uniform(-1.0, 1.0, size=action_dim)

    for t in range(0,2):
        next_state, reward, done, info, _ = env.step(action)
        state = next_state
        rewards.append(reward)
    

    #------------------------------training------------------------------

    for steps in range(1, 1000000):

        if len(replay_buffer.buffer)>=5000 and not policy_training:
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
    if policy_training and i>=100: limit_steps = int(average_steps) + 5 + int(0.05*average_steps)


    print(f"Ep {i}: Rtrn = {total_rewards[i]:.2f}, eps = {algo.actor.eps:.2f} | ep steps = {episode_steps}")



    if policy_training:

        #--------------------saving-------------------------
            
        torch.save(algo.actor.state_dict(), 'actor_model.pt')
        torch.save(algo.critic.state_dict(), 'critic_model.pt')
        torch.save(algo.critic_target.state_dict(), 'critic_target_model.pt')
        #print("saving... len = ", len(replay_buffer), end="")
        with open('replay_buffer', 'wb') as file:
            pickle.dump({'buffer': replay_buffer, 'eps': algo.actor.eps, 'x_coor': algo.actor.x_coor, 'limit_steps':limit_steps, 'total_steps':total_steps}, file)
        #print(" > done")


        #-----------------validation-------------------------

        if total_rewards[i]>=330 or (i>500 and i%500==0):
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
