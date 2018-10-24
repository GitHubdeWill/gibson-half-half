from gibson.core.physics.robot_bases import BaseRobot
import numpy as np
import pybullet as p
import os
import gym, gym.spaces
from transforms3d.euler import euler2quat, euler2mat
from transforms3d.quaternions import quat2mat, qmult
import transforms3d.quaternions as quat
import sys

OBSERVATION_EPS = 0.01


class WalkerBase(BaseRobot):
    """ Built on top of BaseRobot
    Handles action_dim, sensor_dim, scene
    base_position, apply_action, calc_state
    reward
    """
        
    def __init__(self, 
        filename,           # robot file name 
        robot_name,         # robot name
        action_dim,         # action dimension
        power,
        initial_pos,
        target_pos,
        scale,
        sensor_dim=None,
        resolution=512,
        control = 'torque',
        env = None
    ):
        BaseRobot.__init__(self, filename, robot_name, scale, env)
        self.control = control
        self.resolution = resolution
        self.obs_dim = None
        self.obs_dim = [self.resolution, self.resolution, 0]

        if "rgb_filled" in self.env.config["output"]:
            self.obs_dim[2] += 3
        if "depth" in self.env.config["output"]:
            self.obs_dim[2] += 1

        assert type(sensor_dim) == int, "Sensor dimension must be int, got {}".format(type(sensor_dim))
        assert type(action_dim) == int, "Action dimension must be int, got {}".format(type(action_dim))

        action_high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-action_high, action_high)
        obs_high = np.inf * np.ones(self.obs_dim) + OBSERVATION_EPS
        self.observation_space = gym.spaces.Box(-obs_high, obs_high)
        sensor_high = np.inf * np.ones([sensor_dim])
        self.sensor_space = gym.spaces.Box(-sensor_high, sensor_high)

        self.power = power
        self.camera_x = 0
        self.target_pos = target_pos
        self.initial_pos = initial_pos
        self.body_xyz=[0, 0, 0]
        self.action_dim = action_dim
        self.scale = scale
        self.angle_to_target = 0

    def robot_specific_reset(self):
        for j in self.ordered_joints:
            j.reset_joint_state(self.np_random.uniform(low=-0.1, high=0.1), 0)

        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)

        self.scene.actor_introduce(self)
        self.initial_z = None

    def get_position(self):
        '''Get current robot position
        '''
        return self.robot_body.get_position()

    def get_orientation(self):
        '''Return robot orientation
        '''
        return self.robot_body.get_orientation()

    def set_position(self, pos):
        self.robot_body.reset_position(pos)
        
    def set_orient(self, orientation):  
#         print("ok")
        self.robot_body.reset_orientation(orientation)

    def move_by(self, delta):
        new_pos = np.array(delta) + self.get_position()
        self.robot_body.reset_position(new_pos)

    def move_forward(self, forward=0.05):
        x, y, z, w = self.robot_body.get_orientation()
        self.move_by(quat2mat([w, x, y, z]).dot(np.array([forward, 0, 0])))
        
    def move_backward(self, backward=0.05):
        x, y, z, w = self.robot_body.get_orientation()
        self.move_by(quat2mat([w, x, y, z]).dot(np.array([-backward, 0, 0])))

    def turn_left(self, delta=0.03):
        orn = self.robot_body.get_orientation()
        new_orn = qmult((euler2quat(-delta, 0, 0)), orn)
        self.robot_body.set_orientation(new_orn)

    def turn_right(self, delta=0.03):
        orn = self.robot_body.get_orientation()
        new_orn = qmult((euler2quat(delta, 0, 0)), orn)
        self.robot_body.set_orientation(new_orn)
        
    def get_rpy(self):
        return self.robot_body.bp_pose.rpy()

    def apply_action(self, a):
        #print(self.ordered_joints)
        if self.control == 'torque':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
        elif self.control == 'velocity':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_velocity(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
        elif self.control == 'position':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_position(a[n])
        elif type(self.control) is list or type(self.control) is tuple: #if control is a tuple, set different control
        # type for each joint
            for n, j in enumerate(self.ordered_joints):
                if self.control[n] == 'torque':
                    j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
                elif self.control[n] == 'velocity':
                    j.set_motor_velocity(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
                elif self.control[n] == 'position':
                    j.set_motor_position(a[n])
        else:
            pass

    def get_target_position(self):
        return self.target_pos

    def set_target_position(self, pos):
        self.target_pos = pos

    def calc_state(self):
        j = np.array([j.get_joint_relative_state() for j in self.ordered_joints], dtype=np.float32).flatten()
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
        parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z == None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(self.target_pos[1] - self.body_xyz[1],
                                            self.target_pos[0] - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.target_pos[1] - self.body_xyz[1], self.target_pos[0] - self.body_xyz[0]])
        self.walk_target_dist_xyz = np.linalg.norm(
            [self.target_pos[2] - self.body_xyz[2], self.target_pos[0] - self.body_xyz[1], self.target_pos[0] - self.body_xyz[0]])
        angle_to_target = self.walk_target_theta - yaw
        self.angle_to_target = angle_to_target

        self.walk_height_diff = np.abs(self.target_pos[2] - self.body_xyz[2])

        self.dist_to_start = np.linalg.norm(np.array(self.body_xyz) - np.array(self.initial_pos))

        debugmode= 0
        if debugmode:
            print("Robot dsebug mode: walk_height_diff", self.walk_height_diff)
            print("Robot dsebug mode: walk_target_z", self.target_pos[2])
            print("Robot dsebug mode: body_xyz", self.body_xyz[2])

        rot_speed = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw), np.cos(-yaw), 0],
             [        0,             0, 1]]
        )
        vx, vy, vz = np.dot(rot_speed, self.robot_body.speed())  # rotate speed back to body point of view

        debugmode=0
        if debugmode:
            print("Robot state", self.target_pos[1] - self.body_xyz[1], self.target_pos[0] - self.body_xyz[0])

        more = np.array([ z-self.initial_z,
            np.sin(angle_to_target), np.cos(angle_to_target),
            0.3* vx , 0.3* vy , 0.3* vz ,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r, p], dtype=np.float32)

        if debugmode:
            print("Robot more", more)


        if not 'nonviz_sensor' in self.env.config["output"]:
            j.fill(0)
            more.fill(0)

        return np.clip( np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0 (hzyjerry) ==> make rewards similar scale
        debugmode=0
        if (debugmode):
            print("calc_potential: self.walk_target_dist x y", self.walk_target_dist)
            print("robot position", self.body_xyz, "target position", [self.target_pos[0], self.target_pos[1], self.target_pos[2]])
        return - self.walk_target_dist / self.scene.dt


    def calc_goalless_potential(self):
        return self.dist_to_start / self.scene.dt

    def dist_to_target(self):
        return np.linalg.norm(np.array(self.body_xyz) - np.array(self.get_target_position()))

    def _is_close_to_goal(self):
        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
        parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        dist_to_goal = np.linalg.norm([self.body_xyz[0] - self.target_pos[0], self.body_xyz[1] - self.target_pos[1]])
        return dist_to_goal < 2

    def _get_scaled_position(self):
        '''Private method, please don't use this method outside
        Used for downscaling MJCF models
        '''
        return self.robot_body.get_position() / self.mjcf_scaling

class Husky(WalkerBase):
    foot_list = ['front_left_wheel_link', 'front_right_wheel_link', 'rear_left_wheel_link', 'rear_right_wheel_link']
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 0.6
    
    def __init__(self, config, env=None):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        
        WalkerBase.__init__(self, "husky.urdf", "base_link", action_dim=4, 
                            sensor_dim=23, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            target_pos=config["target_pos"], 
                            resolution=config["resolution"], 
                            env = env)
        self.is_discrete = config["is_discrete"]

        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)        
            self.torque = 0.03
            self.action_list = [[self.torque, self.torque, self.torque, self.torque],
                                [-self.torque, -self.torque, -self.torque, -self.torque],
                                [self.torque, -self.torque, self.torque, -self.torque],
                                [-self.torque, self.torque, -self.torque, self.torque],
                                [0, 0, 0, 0]]

            self.setup_keys_to_action()
        else:
            action_high = 0.02 * np.ones([4])
            self.action_space = gym.spaces.Box(-action_high, action_high)
        
    def apply_action(self, action):
        if self.is_discrete:
            realaction = self.action_list[action]
        else:
            realaction = action
        WalkerBase.apply_action(self, realaction)

    def steering_cost(self, action):
        if not self.is_discrete:
            return 0
        if action == 2 or action == 3:
            return -0.1
        else:
            return 0

    def angle_cost(self):
        angle_const = 0.2
        diff_to_half = np.abs(self.angle_to_target - 1.57)
        is_forward = self.angle_to_target > 1.57
        diff_angle = np.abs(1.57 - diff_to_half) if is_forward else 3.14 - np.abs(1.57 - diff_to_half)
        debugmode = 0
        if debugmode:
            print("is forward", is_forward)
            print("diff to half", diff_to_half)
            print("angle to target", self.angle_to_target)
            print("diff angle", diff_angle)
        return -angle_const* diff_angle


    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'), ): 0, ## forward
            (ord('s'), ): 1, ## backward
            (ord('d'), ): 2, ## turn right
            (ord('a'), ): 3, ## turn left
            (): 4
        }

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)

        angular_speed = self.robot_body.angular_speed()
        return np.concatenate((base_state, np.array(angular_speed)))
