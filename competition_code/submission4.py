import asyncio
import numpy as np
import carla
import roar_py_carla
import roar_py_interface
from submission_singleAgent import DQNAgent, normalize_rad, filter_waypoints

NUM_AGENTS = 4
STATE_SIZE = 11
ACTION_SIZE = 9
BATCH_SIZE = 32
TRAIN_EVERY = 2

action_map = np.array([
    [ 0.0, 1.0, 0.0],
    [ 0.0, 0.7, 0.0],
    [ 0.0, 0.4, 0.0],
    [-0.3, 0.8, 0.0],
    [ 0.3, 0.8, 0.0],
    [-0.6, 0.5, 0.0],
    [ 0.6, 0.5, 0.0],
    [ 0.0, 0.0, 0.4],
    [ 0.0, 0.0, 0.0],
])

class AgentState:
    def __init__(self, vehicle, waypoints, start_idx):
        self.vehicle = vehicle
        self.waypoints = waypoints
        self.location_sensor = vehicle.attach_location_in_world_sensor()
        self.velocity_sensor = vehicle.attach_velocimeter_sensor()
        self.rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
        self.current_waypoint_idx = filter_waypoints(
            self.location_sensor.get_last_gym_observation(),
            start_idx,
            waypoints
        )
        self.last_waypoint = self.current_waypoint_idx

    def get_state(self):
        loc = self.location_sensor.get_last_gym_observation()
        vel = self.velocity_sensor.get_last_gym_observation()
        yaw = self.rpy_sensor.get_last_gym_observation()[2]

        self.current_waypoint_idx = filter_waypoints(
            loc, self.current_waypoint_idx, self.waypoints
        )

        idx = [(self.current_waypoint_idx + i) % len(self.waypoints) for i in range(3)]
        wp = np.array([self.waypoints[i].location[:2] for i in idx]).flatten()

        return np.concatenate([wp, vel[:2], loc[:2], [yaw]]).reshape(1, STATE_SIZE)

async def main():
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(10.0)
    roar_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_instance.world
    world.set_control_steps(0.05, 0.005)
    world.set_asynchronous(False)

    waypoints = world.maneuverable_waypoints
    total_wp = len(waypoints)
    spacing = total_wp // NUM_AGENTS

    model = DQNAgent(STATE_SIZE, ACTION_SIZE)

    agents = []
    for i in range(NUM_AGENTS):
        wp_idx = (i * spacing) % total_wp
        vehicle = world.spawn_vehicle(
            "vehicle.tesla.model3",
            waypoints[wp_idx].location + np.array([0, 0, 1]),
            waypoints[wp_idx].roll_pitch_yaw,
            True,
        )
        if vehicle is None:
            print(f"Failed to spawn agent {i}")
            continue
        agents.append(AgentState(vehicle, waypoints, wp_idx))
        print(f"Spawned agent {i} at waypoint {wp_idx}")

    if not agents:
        print("No agents spawned!")
        return

    for _ in range(20):
        await world.step()

    active = [True] * len(agents)
    step_count = 0

    while True:
        await asyncio.gather(*[
            agents[i].vehicle.receive_observation()
            for i in range(len(agents)) if active[i]
        ])

        states = []
        valid_idx = []
        for i, a in enumerate(agents):
            if not active[i]:
                continue
            states.append(a.get_state())
            valid_idx.append(i)

        if not valid_idx:
            break

        states_batch = np.vstack(states)
        qvals = model.predict(states_batch)

        for j, i in enumerate(valid_idx):
            if np.random.rand() <= model.epsilon:
                action = np.random.randint(ACTION_SIZE)
            else:
                action = int(np.argmax(qvals[j]))
            steer, throttle, brake = action_map[action]
            control = {
                "throttle": float(throttle),
                "steer": float(steer),
                "brake": float(brake),
                "hand_brake": 0.0,
                "reverse": 0,
                "target_gear": 0
            }
            await agents[i].vehicle.apply_action(control)

            reward = (agents[i].current_waypoint_idx - agents[i].last_waypoint) + float(throttle) * 0.01 - float(brake) * 0.01
            agents[i].last_waypoint = agents[i].current_waypoint_idx
            model.remember(states[j], action, reward, states[j], False)

        if step_count % TRAIN_EVERY == 0 and len(model.memory) > BATCH_SIZE:
            model.replay(BATCH_SIZE)
        step_count += 1

        if step_count % 50 == 0:
            print(f"[{step_count}] agents={len(valid_idx)} eps={model.epsilon:.3f} mem={len(model.memory)}")

        await world.step()

if __name__ == "__main__":
    asyncio.run(main())
