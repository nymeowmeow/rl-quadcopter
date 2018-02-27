"""Combined task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

TAKEOFF = 1
LANDING = 2
HOVER   = 3

class Combined(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 15.0  # secs
        self.duration = 5.0
        self.target_z = 10.0
        self.state = None
        self.hover_start_time = None
        self.target_position = np.array([0.0, 0.0, self.target_z])
        self.weight_position = 0.8
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.weight_orientation = 0.0
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.weight_velocity = 0.2

    def reset(self):
        # Nothing to reset; just return initial condition
        self.state = None
        self.last_timestamp = None
        self.last_position = None

        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def load_weights(self, task):
        path = os.path.join(util.get_param('out'), 'weights', '{}_dqn_weights.h5'.format(task))
        self.agent.load_weights(path)

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose, orientation, velocity ; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position)/max(timestamp - self.last_timestamp, 1e-3)
        state = np.concatenate([position, orientation, velocity])
        self.last_timestamp = timestamp
        self.last_position = position

        # Compute reward/penalty and check if this episode is complete
        done = False
        error_position = np.linalg.norm(self.target_position - state[0:3])
        error_orientation = np.linalg.norm(self.target_orientation - state[3:7])
        error_velocity = np.linalg.norm(self.target_velocity - state[7:10])

        reward = 0.0
        if self.state is None:
            self.state = TAKEOFF
            self.agent.load_weights("takeoff_weights.hdf5")
            print ('load takeoff weight')
            self.agent.learning = False
            self.agent.epilson = 0.0
        elif self.state == TAKEOFF:
            reward = -min(abs(self.target_z - pose.position.z), 20.0)
            if pose.position.z >= self.target_z: 
               reward += 10.0
               self.agent.load_weights("hover_weights.hdf5")
               self.state = HOVER
               self.hover_start_time = timestamp
               #hover is trained with initial speed 0, which is not the case, so
               #retrain the model to hover
               if self.agent.episode_num < 200:
                   self.agent.learning = True
                   self.agent.epilson = 1.0
               print ('load hover weights, takeoff -> hover')
        elif self.state == HOVER:
            reward = -(self.weight_position * error_position + self.weight_orientation * error_orientation
                       + self.weight_velocity * error_velocity)
            reward -= abs(linear_acceleration.z)
            if error_position < 1.5:
                reward += 10.0

            if (timestamp - self.hover_start_time) > self.duration:
                reward += 50.0
                self.agent.load_weights("landing_weights.hdf5")
                self.state = LANDING
                self.agent.learning = False
                self.agent.epilson = 0.0
                print ('load landing weights, hover -> landing')
        elif self.state == LANDING:
            reward = -(self.weight_position * error_position + self.weight_orientation * error_orientation)
            if error_position <= 0.1:
               reward += 10.0
               done = True
               print ('landing done')

        if timestamp > self.max_duration:  # agent has run out of time
            reward -= 50.0  # extra penalty
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
