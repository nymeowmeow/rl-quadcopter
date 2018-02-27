import numpy as np
import random
import os
import pandas as pd
from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent
from collections import namedtuple, deque
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential

Experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer(object):
    def __init__(self, max_size=1000):
        self.size = max_size                 #maximum size of buffer
        self.buffer = deque(maxlen=max_size) #internal memory (list)-
 
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.buffer.append(e)
            
    def sample(self, batch_size=64):
        return random.sample(self.buffer, k=batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

class DQN(BaseAgent):
    def __init__(self, task):
        self.task = task
        #constrain state and action spaces
        self.state_size = 1
        self.state_low = self.task.observation_space.low[2]
        self.state_high = self.task.observation_space.high[2]
        self.state_range = self.state_high - self.state_low
        #only limit to z direction
        self.action_range = (self.task.action_space.high - self.task.action_space.low)[2]
        self.action_low = self.task.action_space.low[2]
        self.action_high = self.task.action_space.high[2]

        stepping = (self.action_high - 10.0)/16.0
        self.discrete_actions = np.arange(10.0, self.action_high+0.1, stepping)
        self.action_size = len(self.discrete_actions)
        print ('discrete action:', self.discrete_actions, ', action size: ', self.action_size)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.9  # discount factor
        self.learning_rate = 0.001

        self.model = self.build_model() 

        #save episode stats
        self.stats_filename = os.path.join(util.get_param('out'), 
                                           "stats_{}.csv".format(util.get_timestamp()))
        self.stats_columns = [ 'episode', 'total_reward']
        self.episode_num = 1
        print ("saving stats {} to {}".format(self.stats_columns, self.stats_filename))

        self.epilson = 1.0
        self.epilson_decay = 0.96
        self.epilson_min = 0.05

        self.learning = True
        self.reset_episode_vars()
        self.best_reward = -99999

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size))
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def preprocess_state(self, state):
        #reduce state vector to relevant dimensions
        return state[2]

    def postprocess_state(self, action):
        #return complete action vector.
        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[2] = self.discrete_actions[action]
        return complete_action

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0

    def step(self, state, reward, done):
        #reduce state vector
        state = self.preprocess_state(state)
 
        state = (state - self.state_low)/self.state_range
        state = state.reshape(1, -1)
 
        # Choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward
 
        self.last_state = np.copy(state)
        self.last_action = np.copy(action)

        if done:
            # Learn, if enough samples are available in memory
           if len(self.memory) > self.batch_size:
               experiences = self.memory.sample(self.batch_size)
               self.learn(experiences)

           #write episode stats
           self.write_stats([self.episode_num, self.total_reward])
           if self.learning and self.episode_num > 100 and self.total_reward > self.best_reward:
               self.best_reward = self.total_reward
               print ('save weight when reward is', self.best_reward)
               self.save_weights("weights.hdf5")

           self.episode_num += 1
           self.reset_episode_vars()

           if self.learning and self.episode_num > 200:
              self.learning = False #stop learning after 200 episode
              self.epilson = 0.0
              self.load_weights("weights.hdf5")

        return self.postprocess_state(action)

    def write_stats(self, stats):
        #write single episode stats to csv file
        print ('stats',stats, self.epilson)
        df_stats = pd.DataFrame([stats], columns = self.stats_columns)
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
                        header=not os.path.isfile(self.stats_filename))

    def load_weights(self, filename):
        path = os.path.join(util.get_param('out'), filename)
        self.model.load_weights(path)

    def save_weights(self, filename):
        path = os.path.join(util.get_param('out'), filename)
        self.model.save_weights(path)

    def act(self, states):
        if np.random.rand() <= self.epilson:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.vstack([states]))
        return np.argmax(act_values[0])

    def learn(self, experience):
        if not self.learning:
            return

        for e in experience:
            target = e.reward
            if not e.done:
               target = e.reward + self.gamma*np.amax(self.model.predict(np.vstack([e.next_state]))[0])

            target_f = self.model.predict(np.vstack([e.state]))
            target_f[0][e.action] = target
            self.model.fit(np.vstack([e.state]), target_f, epochs=1, verbose = 0)

        self.epilson = np.maximum(self.epilson*self.epilson_decay, self.epilson_min)
