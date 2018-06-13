#!/usr/bin/env python3
import sys
import gym

import satellite
import dqn_satellite

env = gym.make('SatelliteDrag-v0' if len(sys.argv) < 2 else sys.argv[1])

#directory = "/tmp/gym-satellite-test"
#env = gym.wrappers.Monitor(env, directory)

nb_actions = env.action_space.n

# load model and weights
model = dqn_satellite.create_model(nb_actions, env.observation_space.shape)
dqn = dqn_satellite.create_dqn(model, nb_actions)

dqn.load_weights('dqn_Satellite_weights.h5f')

dqn.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=5000)
