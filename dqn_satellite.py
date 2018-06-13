#!/usr/bin/env python3
"""Trains a DQN agent to control the satellite.

By default, it trains on the SatelliteDrag-v0 environment. Supply the script
with an environment name to train on a different environment.
"""

import sys
import os
import gym

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import MaxBoltzmannQPolicy
from rl.memory import SequentialMemory

import satellite


def create_model(nb_actions, observation_shape):
    model = Sequential()
    model.add(Flatten(input_shape=(1, ) + observation_shape))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    return model


def create_dqn(model, nb_actions):
    """Creates and compiles a DQN agent with an Adam optimizer."""
    memory = SequentialMemory(limit=100000, window_length=1)
    policy = MaxBoltzmannQPolicy(tau=10, eps=0.2)
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        target_model_update=1e-2,
        policy=policy,
        gamma=0.995,
        batch_size=64)
    dqn.compile(Adam(lr=5e-4, decay=0.0), metrics=['mae'])

    return dqn


def main():
    env = gym.make('SatelliteDrag-v0' if len(sys.argv) < 2 else sys.argv[1])
    env.seed(99)
    nb_actions = env.action_space.n
    #directory = "/tmp/gym-satellite-training"
    #env = gym.wrappers.Monitor(env, directory)

    model = create_model(nb_actions, env.observation_space.shape)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    dqn = create_dqn(model, nb_actions)

    # If saved weights exist, load them
    filename = "dqn_Satellite_weights.h5f"
    if os.path.isfile(filename):
        dqn.load_weights(filename)

    dqn.fit(
        env,
        nb_steps=800000,
        visualize=False,
        verbose=2,
        nb_max_episode_steps=5000)

    # Save the final weights.
    dqn.save_weights('dqn_Satellite_weights.h5f', overwrite=True)


if __name__ == "__main__":
    main()
