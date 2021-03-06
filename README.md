# Satellite environment

## Update: as of June 2018 I am no longer adding or making changes to this code. I'm leaving it online in case anyone is interested.

An environment for OpenAI gym (https://gym.openai.com) in which an agent has to
learn to keep a satellite in a given orbit around Earth by firing its engines.
To make the task more difficult, random initial perturbations and (grossly
exaggerated) atmospheric drag can be added or the satellite can be started from
rest.

Disclaimer: when I wrote this code, I was not an expert in machine learning and
only used this experiment as a practice project to familiarise myself with
OpenAI gym and to experiment with reinforcement learning.
Feedback, comments, and criticism are very welcome!

## Environments
There are currently three environments:

1. `SatelliteDrag-v0`: the satellite experiences (grossly exaggerated) atmospheric
drag which leads to orbital decay. The agent has to fire the engines as little
as possible while keeping the satellite in its orbit (see
https://youtu.be/bgc_k1CebPI for a video).

2. `SatellitePerturbation-v0`: the satellite is given a random initial
perturbation away from its orbit. The agent has to learn to return it to a
stable orbit with minimal fuel consumption.

3. `SatelliteRest-v0`: the satellite is started almost from rest. The agent
has to fire the engines to get it into orbit (since this will allow it to save
fuel in the long run). This is hard but not impossible.

In all environments the agent receives a reward for keeping the satellite close
to the correct orbital radius and gets penalised for using fuel. The game ends
when the satellite crashes into Earth or moves too far away.

No reward based on the angular velocity is given. The agent has to learn the
correct orbital angular velocity by minimising fuel consumption.

## Example solution
The file `dqn_satellite.py` implements a DQN using Max-Boltzmann exploration in
keras-rl. It can learn to solve environments 1 and 2 above.
The weights in this repository have been pre-trained for environment 1 above.
Running `test_dqn_satellite.py` will show the agent in action.
A video of environment 1 (with atmospheric drag) is available at
https://youtu.be/bgc_k1CebPI.

An agent that has been trained on environment 1 can also solve environment 3,
i.e. it can propel a satellite into the correct orbit from rest.
See https://youtu.be/Otri0fNgS3E for a video.
I have not yet been able to train an agent on environment 3 only. This is work
in progress.

In the videos, the red circle
shows the target orbit and the red indicator at the bottom of the screen shows
how close the satellite is to the ideal angular velocity. It clearly shows that
the agent learns to minimise fuel consumption by keeping the satellite at the
correct orbital angular velocity.
