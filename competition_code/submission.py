## This is course material for Introduction to Modern Artificial Intelligence
## Example code: cartpole_dqn.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020-2024. Intelligent Racing Inc. Not permitted for commercial use

## CartPole DQN with Rendering - Uses Gymnasium (the maintained successor to OpenAI Gym)
## Shows the game being played during training.
## The `import gymnasium as gym` alias is the migration path recommended by Gymnasium itself,
## so the rest of the script reads the same as the original Gym version.

import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as _tf

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.train_every = 4
        self._step_count = 0
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        act_values = self.model.predict(state, verbose=0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(act_values[0]))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([e[0] for e in minibatch])
        next_states = np.vstack([e[3] for e in minibatch])
        actions = [e[1] for e in minibatch]
        rewards = [e[2] for e in minibatch]
        dones = [e[4] for e in minibatch]

        targets = self.model.predict(states, verbose=0)
        next_qs = self.model.predict(next_states, verbose=0)
        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(next_qs[i])
            targets[i][actions[i]] = target
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

from typing import List
import roar_py_interface

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx
class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.state_size = 11
        self.action_size = 9
        self.model = DQNAgent(self.state_size, self.action_size)
        _gpus = _tf.config.list_physical_devices('GPU')
        if _gpus:
            names = ", ".join(d.name for d in _gpus)
            print(f"[device] TensorFlow sees {len(_gpus)} GPU device(s): {names}")
        else:
            print("[device] No GPU visible to TensorFlow")
        self.batch_size = 32
        self.scores = []

        self.action_map = np.array([
            [ 0.0, 1.0, 0.0],   # 0: straight + accelerate
            [ 0.0, 0.7, 0.0],   # 1: straight + cruise
            [ 0.0, 0.4, 0.0],   # 2: straight + slow
            [-0.3, 0.8, 0.0],   # 3: slight left + accelerate
            [ 0.3, 0.8, 0.0],   # 4: slight right + accelerate
            [-0.6, 0.5, 0.0],   # 5: medium left + coast
            [ 0.6, 0.5, 0.0],   # 6: medium right + coast
            [ 0.0, 0.0, 0.4],   # 7: brake straight
            [ 0.0, 0.0, 0.0],   # 8: coast
        ])

    async def initialize(self) -> None:
        self.last_waypoint = 0
        self.score = 0

        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
        #await self.vehicle.apply_action({"throttle":0.0,"steer":0.0,"brake":0.0,"hand_brake":0.0,"reverse":0,"target_gear":0})

    def compute_error(self, current_waypoint_idx):
        vector_to_waypoint = (self.maneuverable_waypoints[(current_waypoint_idx + 1) % len(self.maneuverable_waypoints)].location[:2] - self.location_sensor.get_last_gym_observation()[:2])[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])
        delta_heading = normalize_rad(heading_to_waypoint - self.rpy_sensor.get_last_gym_observation()[2])
        return delta_heading
    def euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)
    async def step(
        self
    ) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        idx = [
            (self.current_waypoint_idx + i) % len(self.maneuverable_waypoints)
            for i in range(3)
        ]
        waypoints = np.array([
            self.maneuverable_waypoints[i].location[:2] for i in idx
        ]).flatten()

        state = np.concatenate((
            waypoints,
            vehicle_velocity[:2],
            vehicle_location[:2],
            [self.rpy_sensor.get_last_gym_observation()[2]]
        )).reshape(1, self.state_size)

        action = self.model.act(state)
        steer, throttle, brake = self.action_map[action]
        throttle = 1

        control = {
            "throttle": float(throttle),
            "steer": float(steer),
            "brake": float(brake),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }

        await self.vehicle.apply_action(control)
        reward = self.current_waypoint_idx - float(brake) * 0.1 + self.euclidean_distance(vehicle_location, waypoints[self.last_waypoint])
        self.last_waypoint = self.current_waypoint_idx
        self.model.remember(state, action, reward, state, False)
        if len(self.model.memory) > self.batch_size and self.model._step_count % self.model.train_every == 0:
            self.model.replay(self.batch_size)
        self.model._step_count += 1
        return control
