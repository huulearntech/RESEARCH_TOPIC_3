import numpy as np
import random
import math
from itertools import combinations
from collections import deque
import tensorflow as tf
from keras.layers import LSTM, Dense, Reshape, concatenate
from keras.models import Model, save_model
from keras import Input, backend, losses, optimizers

#ENVIRONMENT CONSTANTS
NUM_USERS = 5 # number of users, 30
NUM_CHANNELS = 3  # number of channels, 20
BATTERY_CAPACITY = 10  # battery capacity of each user, identical for all users #(units ; 5dBm per unit)
TRANSMIT_POWER = 2  # transmit power between each user and access point (AP) #(units)
ACTION_SPACE_SIZE = math.comb(NUM_USERS, NUM_CHANNELS)
BANDWIDTH = int(5e6) #5 MHz
NOISE = -11
RANGE = 500



#MODEL CONSTANTS
TIME_STEPS = 1
UPDATE_RATE = 10 # update every 10 frames

WINDOW_WIDTH = 20
BATCH_SIZE = 16
NUM_UNITS =32

T = 200
TOTAL_EPISODES= 10


EPSILON_START = 0.9
EPSILON_END = 0.1
EPSILON_DECAY = 0.99

LEARNING_RATE = 1e-4
GAMMA = 0.99
BETA = 100 #penalty factor for balancing Reward and P_loss
