#!/usr/bin/env python3
import sys
import gym

from keras.models import model_from_json
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

import satellite

env = gym.make('SatelliteDrag-v0' if len(sys.argv) < 2 else sys.argv[1])

#directory = "/tmp/gym-satellite-test"
#env = gym.wrappers.Monitor(env, directory)

nb_actions = env.action_space.n

# load model and weights
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=20,
    target_model_update=1e-2,
    policy=GreedyQPolicy())
dqn.compile(Adam(), metrics=['mae'])

dqn.load_weights('dqn_Satellite_weights.h5f')

dqn.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=5000)
