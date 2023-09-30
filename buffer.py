from collections import deque
import numpy as np
import math
import torch

# we use cache to append transitions, and then update buffer to collect Q Returns, purging cache at the episode end.
class ReplayBuffer:
    def __init__(self, device, n_steps, capacity=300*1024):
        self.buffer, self.capacity, self.length =  deque(maxlen=capacity), capacity, 0
        self.indices, self.indexes, self.probs = [], np.array([]), []
        self.cache = []
        self.device = device
        self.n_steps = n_steps
        self.random = np.random.default_rng()
        self.batch_size = min(max(128, self.length//300), 1024) #in order for sample to describe population

    # Returns for old policies are less correct, we need to gradually forget past history.
    def fade(self, norm_index):
        return 0.01*np.tanh(3*norm_index**2) # ▁/▔

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
            self.batch_size = min(max(128, self.length//300), 1024)

            #updating priorities: less priority for older data
            self.indices.append(self.length-1)
            self.indexes = np.array(self.indices)
            if self.length>1:
                weights = self.fade(self.indexes/self.length)
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
