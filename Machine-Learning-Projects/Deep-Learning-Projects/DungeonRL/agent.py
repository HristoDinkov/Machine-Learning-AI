from enums import *
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Softmax, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


class Drunkard:
    def __init__(self):
        self.q_table = None

    def get_next_action(self, state):
            return FORWARD if random.random() < 0.5 else BACKWARD

    def update(self, old_state, new_state, action, reward):
            pass


class Accountant:
    def __init__(self):
        self.q_table = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]   # Spreadsheet (Q-table) for rewards accounting

    def get_next_action(self, state):
        if self.q_table[FORWARD][state] > self.q_table[BACKWARD][state]:
            return FORWARD
        elif self.q_table[BACKWARD][state] > self.q_table[FORWARD][state]:
            return BACKWARD

        return FORWARD if random.random() < 0.5 else BACKWARD

    def update(self, old_state, new_state, action, reward):
        self.q_table[action][old_state] += reward


class Gambler:
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations=10000):
        self.q_table = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]   # Spreadsheet (Q-table) for rewards accounting
        self.learning_rate = learning_rate                  # How much we appreciate new q-value over current
        self.discount = discount                            # How much we appreciate future reward over current
        self.exploration_rate = 1.0                         # Initial exploration rate
        self.exploration_delta = 1.0 / iterations           # Shift from exploration to explotation

    def get_next_action(self, state):
        if random.random() > self.exploration_rate:
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        if self.q_table[FORWARD][state] > self.q_table[BACKWARD][state]:
            return FORWARD
        elif self.q_table[BACKWARD][state] > self.q_table[FORWARD][state]:
            return BACKWARD

        return FORWARD if random.random() < 0.5 else BACKWARD

    def random_action(self):
        return FORWARD if random.random() < 0.5 else BACKWARD

    def update(self, old_state, new_state, action, reward):
        old_value = self.q_table[action][old_state]
        future_action = self.greedy_action(new_state)
        future_reward = self.q_table[future_action][new_state]

        new_value = old_value + self.learning_rate * (reward + self.discount * future_reward - old_value)
        self.q_table[action][old_state] = new_value

        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta


class DeepGambler:
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations=10000):
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.exploration_delta = 1.0 / iterations

        self.X_n = 5    # Number of inputs
        self.Y_n = 2    # Number of ouputs(actions)

        self.model = self.define_model()

    def define_model(self):
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, self.X_n)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.Y_n, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        return model

    def to_one_hot(self, state):
        one_hot = np.zeros((1, 5))
        one_hot[0, [state]] = 1
        return one_hot

    def get_q(self, state):
        return self.model.predict(self.to_one_hot(state))

    def get_next_action(self, state):
        if random.random() > self.exploration_rate:
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        return np.argmax(self.get_q(state))

    def random_action(self):
        return FORWARD if random.random() < 0.5 else BACKWARD

    def train(self, old_state, action, reward, new_state):
        old_state_q_values = self.get_q(old_state)[0]
        new_state_q_values = self.get_q(new_state)[0]
        old_state_q_values[action] = reward + self.discount * np.amax(new_state_q_values)

        training_input = self.to_one_hot(old_state)
        target_output = [old_state_q_values.reshape(-1, 2)]

        self.model.fit(training_input, target_output, epochs=1, verbose=0)

    def update(self, old_state, new_state, action, reward):
        self.train(old_state, action, reward, new_state)

        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta
