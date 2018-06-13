"""
A satellite envrionment for OpenAI gym

This environment for OpenAI gym (https://gym.openai.com) consists of a satellite
orbiting a planet. The aim is to keep the satellite at a fixed orbital radius by
firing the 4 engines as rarely as is necessary. To make this task harder, the
satellite can be started from rest, a random perturbation can be applied to the
satellite, and (grossly exaggerated) atmospheric drag can be added to cause
orbital decay.

The agent is rewarded for keeping the satellite close to the desired orbital
radius. The agent is penalised for using fuel.
The game ends when the satellite crashes into Earth or moves too far away from
Earth.

This file only contains the base version: a satellite in perfect orbit around
Earth. It requires no action from the agent. To introduce a challenge, inherit
from the base class and set the perturbation, etc.

Running this file will show the behaviour without user input.

***
This file is a (substantial) modification of the lunar lander environment in
OpenAI gym. However, some ideas and a few lines of code are the same. See
LICENCE.md in this repository for the license and the copyright notice that
comes with OpenAI gym.
"""

import math
import numpy as np

import Box2D
from Box2D.b2 import (circleShape, fixtureDef, polygonShape, contactListener)

from gym.envs.classic_control import rendering

import gym
from gym import spaces
from gym.utils import seeding

FPS = 50
SCALE = 30.0

ENGINE_POWER = 0.1

SATELLITE_HEIGHT = 28 / SCALE
SATELLITE_WIDTH = 12 / SCALE

# yapf: disable
SATELLITE_POLYGON = [(-SATELLITE_WIDTH / 2, +SATELLITE_HEIGHT / 2),
                     (-SATELLITE_WIDTH / 2, -SATELLITE_HEIGHT / 2),
                     (+SATELLITE_WIDTH / 2, -SATELLITE_HEIGHT / 2),
                     (+SATELLITE_WIDTH / 2, +SATELLITE_HEIGHT / 2)]

# The solar panel
PANEL_POLYGON = [(-1.5 * SATELLITE_WIDTH, 0),
                 (-1.5 * SATELLITE_WIDTH, -SATELLITE_HEIGHT / 2.3),
                 (+1.5 * SATELLITE_WIDTH, -SATELLITE_HEIGHT / 2.3),
                 (+1.5 * SATELLITE_WIDTH, 0)]
# yapf: enable

VIEWER_SIZE = 600
DOMAIN_SIZE = VIEWER_SIZE / SCALE

GRAV_CONST = 1
EARTH_MASS = 50
EARTH_POS = (DOMAIN_SIZE / 2, DOMAIN_SIZE / 2)
EARTH_RADIUS = 30 / SCALE

ORBITAL_RADIUS = 5 * EARTH_RADIUS
ORBITAL_ANGULAR_VEL = math.sqrt(GRAV_CONST * EARTH_MASS / ORBITAL_RADIUS**3)

NUM_STATES = 8


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.satellite == contact.fixtureA.body:
            self.env.game_over = True


class SatelliteBasic(gym.Env):
    """Base version of satellite environment for OpenAI gym.

    In this base version, the satellite is in the perfect orbit and no action is
    required. To introduce a challenge, inherit from this basic version and
    change the three parameters below.
    """

    initial_perturbation = 0
    start_from_rest = False
    drag_coeff = 0

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.world.gravity = (0, 0)  # We'll have to do the gravity by hand
        self.earth = None
        self.satellite = None
        self.panel = None
        self.particles = []

        high = np.array([np.inf] * NUM_STATES)
        self.observation_space = spaces.Box(-high, high)

        # 5 actions: Noop, left, bottom, right, top
        self.action_space = spaces.Discrete(5)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.earth:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.satellite)
        self.satellite = None
        self.world.DestroyBody(self.earth)
        self.earth = None
        self.world.DestroyBody(self.panel)
        self.panel = None

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False

        self.earth = self._create_earth()
        self.satellite = self._create_satellite()
        self.panel = self._create_panel()

        self.drawlist = [self.panel] + [self.satellite] + [self.earth]

        return self.step(0)[0]

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid " % (action, type(action))

        engine_fired = False
        if action > 0:
            self._fire_engine(action)
            engine_fired = True

        # Gravitational attraction to Earth
        direction = [
            self.satellite.position[0] - EARTH_POS[0],
            self.satellite.position[1] - EARTH_POS[1]
        ]
        squared_distance = direction[0]**2 + direction[1]**2
        distance = math.sqrt(squared_distance)
        grav_force = GRAV_CONST * EARTH_MASS * self.satellite.mass / squared_distance
        self.satellite.ApplyForceToCenter(
            (-direction[0] * grav_force / distance,
             -direction[1] * grav_force / distance), True)

        # Atmospheric drag
        self.satellite.ApplyForceToCenter(
            -self.drag_coeff * self.satellite.linearVelocity, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        position = self.satellite.position
        vel = self.satellite.linearVelocity
        radial_velocity = (
            direction[0] * vel.x + direction[1] * vel.y) / distance
        orbital_velocity = (
            direction[0] * vel.y - direction[1] * vel.x) / distance
        state = [
            (position.x - EARTH_POS[0]) / (DOMAIN_SIZE / 2),
            (position.y - EARTH_POS[1]) / (DOMAIN_SIZE / 2),
            distance / (DOMAIN_SIZE / 2),
            math.atan2(direction[1], direction[0]),
            vel.x * (DOMAIN_SIZE / 2) / FPS,
            vel.y * (DOMAIN_SIZE / 2) / FPS,
            radial_velocity / FPS,
            orbital_velocity / distance / FPS,
        ]
        assert len(state) == NUM_STATES

        engine_reward = -engine_fired * 0.4  # keep fuel consumption down

        # Reward for correct radius
        orbit_reward = math.exp(-(distance - ORBITAL_RADIUS)**2 / 0.5)

        reward = engine_reward + orbit_reward

        if distance > 1.5 * ORBITAL_RADIUS:
            self.game_over = True

        done = False

        # Stop if satellite has crashed into Earth or is too far away
        if self.game_over:
            reward = -100
            done = True

        return np.array(state), reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWER_SIZE, VIEWER_SIZE)
            self.viewer.set_bounds(0, DOMAIN_SIZE, 0, DOMAIN_SIZE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl),
                          max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl),
                          max(0.2, 0.5 * obj.ttl))

        self._clean_particles(False)

        self._render_indicators()

        # The panel just follows the satellite around
        self.panel.position = self.satellite.position

        for obj in [self.drawlist[0]] + self.particles + self.drawlist[1:]:
            for f in obj.fixtures:
                trans = f.body.transform
                if isinstance(f.shape, circleShape):
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(
                        f.shape.radius, 40, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(
                        f.shape.radius,
                        40,
                        color=obj.color2,
                        filled=False,
                        linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(
                        path, color=obj.color2, linewidth=2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _fire_engine(self, engine_number):
        """Fires a given engine of the satellite."""
        fired_direction = math.pi * (1 + engine_number) / 2
        engine_position = self.satellite.position + [
            math.cos(fired_direction) * SATELLITE_WIDTH / 2,
            math.sin(fired_direction) * SATELLITE_HEIGHT / 2
        ]
        impulse_direction = (-math.cos(fired_direction) * ENGINE_POWER,
                             -math.sin(fired_direction) * ENGINE_POWER)
        self.satellite.ApplyLinearImpulse(impulse_direction,
                                          self.satellite.position, True)

        # Visualise the engine action using particles
        dispersion_angle = self.np_random.uniform(-math.pi / 8, math.pi / 8)
        x_impulse = -(impulse_direction[0] * math.cos(dispersion_angle) -
                      impulse_direction[1] * math.sin(dispersion_angle))
        y_impulse = -(impulse_direction[0] * math.sin(dispersion_angle) +
                      impulse_direction[1] * math.cos(dispersion_angle))
        particle = self._create_particle(
            density=3, position=engine_position, ttl=1.0)
        particle.linearVelocity = self.satellite.linearVelocity
        particle.ApplyLinearImpulse((x_impulse, y_impulse), particle.position,
                                    True)

    def _render_indicators(self):
        """Renders the radius and the angular velocity indicator"""
        # Orbital radius indicator
        transform_to_center = rendering.Transform(translation=EARTH_POS)
        self.viewer.draw_circle(
            ORBITAL_RADIUS,
            40,
            center=EARTH_POS,
            color=(0.5, 0., 0.),
            filled=False,
            linewidth=2,
            transform=transform_to_center).add_attr(transform_to_center)

        # Calculate angular velocity difference
        direction = [
            self.satellite.position[0] - EARTH_POS[0],
            self.satellite.position[1] - EARTH_POS[1]
        ]
        distance = math.sqrt(direction[0]**2 + direction[1]**2)
        vel = self.satellite.linearVelocity
        orbital_velocity = (
            direction[0] * vel.y - direction[1] * vel.x) / distance
        velocity_difference = (
            -orbital_velocity / distance - ORBITAL_ANGULAR_VEL)

        # Draw the angular velocity indicator
        # yapf: disable
        self.viewer.draw_polygon(
            [[DOMAIN_SIZE / 2, 1], [DOMAIN_SIZE / 2, 1.5],
             [DOMAIN_SIZE / 2 + velocity_difference * 5, 1.5],
             [DOMAIN_SIZE / 2 + velocity_difference * 5, 1]],
            color=(0.8, 0.2, 0.2),
            filled=True)
        # yapf: enable

    def _create_satellite(self):
        angle = np.random.uniform(0, 2 * math.pi)
        distance = ORBITAL_RADIUS * (1 + np.random.uniform(
            -self.initial_perturbation, self.initial_perturbation))

        satellite = self.world.CreateDynamicBody(
            position=(EARTH_POS[0] + distance * math.cos(angle),
                      EARTH_POS[1] + distance * math.sin(angle)),
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x, y)
                                             for x, y in SATELLITE_POLYGON]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x0020,  # collide only with Earth
                restitution=0.0))
        satellite.color1 = (0.5, 0.5, 0.5)
        satellite.color2 = (0., 0., 0.)

        speed_deviation = np.random.uniform(-self.initial_perturbation,
                                            self.initial_perturbation)
        speed = ORBITAL_ANGULAR_VEL * ORBITAL_RADIUS * (1 + speed_deviation)

        if self.start_from_rest:
            speed *= 0.1

        satellite.linearVelocity = (speed * math.sin(angle),
                                    -speed * math.cos(angle))

        return satellite

    def _create_panel(self):
        panel = self.world.CreateDynamicBody(
            position=self.satellite.position,
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x, y)
                                             for x, y in PANEL_POLYGON]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x0020,
                restitution=0.0))
        panel.color1 = (0.9, 0.9, 0.9)
        panel.color2 = (0., 0., 0.)

        return panel

    def _create_earth(self):
        earth = self.world.CreateStaticBody(
            position=EARTH_POS,
            angle=0.,
            fixtures=fixtureDef(
                shape=circleShape(radius=EARTH_RADIUS),
                categoryBits=0x0020,
                maskBits=0x0010))
        earth.color1 = (0., 0., 1.)
        earth.color2 = (0., 0., 1.)
        return earth

    def _create_particle(self, density, position, ttl):
        p = self.world.CreateDynamicBody(
            position=position,
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=1.5 / SCALE),
                density=density,
                categoryBits=0x1000,
                maskBits=0x0000))
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, destroy_all):
        while self.particles and (destroy_all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))


def main():
    """Runs the satellite environment without any user input"""
    env = SatelliteBasic()
    env.render()
    total_reward = 0

    while True:
        _, reward, done, _ = env.step(0)
        env.render()
        total_reward += reward
        if done:
            break

    print("Total reward: %f." % total_reward)


if __name__ == "__main__":
    main()
