# utils imports
import pdb
import time
import numpy as np
import pandas as pd
import random
from collections import deque
import os
import matplotlib.pyplot as plt

# Gym imports
from gym.spaces import Discrete, Box

# Keras imports
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

os.environ["KERAS_BACKEND"] = "plaidml.bridge.keras"
NUM_CANDIDATES = 20
INFEASIBILITY = 1e6
EPSILON = 1e-6
OPTIMAL = 1
INFEASIBILITY = 1e6

BRANCHING_TYPES = ["Most Fractional", "Random", "Strong", "Minimum Infeasibility", "Maximum Infeasibility", "Pseudo-cost", "Pseudo-reduced-cost"]

class DQN:
    def __init__(self, memory_size=5*10**5, batch_size=32, gamma=0.99,
        exploration_max=1.0, exploration_min=0.01, exploration_decay=0.99999,
        learning_rate=0.001, tau=0.125):
        
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.learning_rate = learning_rate
        self.tau = tau

        self.action_space = Discrete(len(BRANCHING_TYPES))
        self.observation_space = Box(low=np.array([0,0]), high=np.array([2*10**4,1]), dtype=np.float64)

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.loss_history = []
        self.fit_count = 0

        self.nodes_queue = []

    def create_model(self):
        _model = Sequential()
        state_shape = self.observation_space.shape
        _model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        # _model.add(Dense(48, activation="relu"))
        _model.add(Dense(24, activation="relu"))
        _model.add(Dense(self.action_space.n))
        _model.compile(loss="mse", 
                       optimizer=Adam(learning_rate=self.learning_rate))

        return _model

    def get_action(self, state):
        self.exploration_max *= self.exploration_decay
        self.exploration_max = max(self.exploration_min, self.exploration_max)
        if np.random.random() < self.exploration_max:
            return self.action_space.sample()
        # q_values = self.model.predict(state, verbose=0)
        q_values = self.model(state).numpy()
        best_action = np.argmax(q_values[0])
        return best_action

    def remember(self, _state, action, reward, _next_state, done):
        state = _state
        next_state = _next_state
        self.memory.append([state, action, reward, next_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 
        
        samples = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        targets = []

        for state, action, reward, next_state, done in samples:
            
            target = self.target_model(state).numpy()
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            
            targets.append(target)
            
        states = np.array([i[0] for i in states])
        targets = np.array([i[0] for i in targets])
        self.loss_history.append(self.model.fit(states, targets,verbose=0).history['loss'][0])

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
    
    def calc_reward(self, state, next_state):
        T = 1 / np.exp(next_state[0][0])
        b = min(1,np.exp((state[0][1] - next_state[0][1])/1 / np.exp(next_state[0][0])))

        if next_state[0][1] < state[0][1]:
            return 100*b
        else:
            return T*b

    def save_model(self, fn):
        self.model.save(fn)