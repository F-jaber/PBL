###### code adapted from Softbank Robotics Research's qi-gym repository found here: https://github.com/softbankrobotics-research ######
import sys

import os
import gym
import time
import argparse
from datetime import datetime
import numpy as np
from gym import spaces
import pybullet
import pybullet_data
from qibullet import PepperVirtual
from qibullet import SimulationManager


class RobotEnv(gym.Env):
    def __init__(self, gui=False, rec=False):
        self.r_kinematic_chain = [
            "RShoulderPitch",
            "RShoulderRoll",
            "RElbowRoll",
            "RElbowYaw",
            "RWristYaw"]

        self.initial_stand = [
            1.207,
            -0.129,
            1.194,
            1.581,
            1.632]

        # if you want to visualize the simulation set to true
        self.gui = gui
        # if you want to record, set to true
        self.rec = rec
        self.joints_initial_pose = list()

        # Passed to True at the end of an episode
        self.episode_start_time = None
        self.episode_over = False
        self.episode_reward = 0.0
        self.episode_number = 0
        self.episode_steps = 0
        self.file_name = 'Throw_env-v0'
        self.simulation_manager = SimulationManager()
        self.states = []
        #radius of the ball
        self.projectile_radius = 0.03

        self._setupScene()

        lower_limits = list()
        upper_limits = list()

        #create observation space limits
        lower_limits.extend([-5, -5, -5, -5, -5, -5])
        upper_limits.extend([5, 5, 5, 5, 5, 5])


        # Define the observation space
        self.observation_space = spaces.Box(
            low=np.array(lower_limits),
            high=np.array(upper_limits))

        # Define the action space
        velocity_limits = [
            self.pepper.joint_dict[joint].getMaxVelocity() for
            joint in self.r_kinematic_chain]
        velocity_limits.extend([
            -self.pepper.joint_dict[joint].getMaxVelocity() for
            joint in self.r_kinematic_chain])

        normalized_limits = self.normalize(velocity_limits)

        self.max_velocities = normalized_limits[:len(self.r_kinematic_chain)]
        self.min_velocities = normalized_limits[len(self.r_kinematic_chain):]
        self.action_space = spaces.Box(
            low=np.array(self.min_velocities),
            high=np.array(self.max_velocities))

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        try:
            action = list(action)
            assert len(action) == len(self.action_space.high.tolist())

        except AssertionError:
            print("Incorrect action")
            return None, None, None, None

        self.episode_steps += 1
        np.clip(action, self.min_velocities, self.max_velocities)
        #throw ball
        self._setVelocities(self.r_kinematic_chain, action)

        obs, reward = self._getState()
        self.states.append(obs.tolist())
        return obs, reward, self.episode_over, {}

    def reset(self):
        """
        Resets the environment for a new episode
        """
        self.episode_over = False
        self.episode_reward = 0.0
        self.episode_steps = 0
        self._resetScene()
        self.states = []
        #record episode
        if self.gui and self.rec:
            self.record(self.episode_number)
        # Reset the start time for the current episode
        self.episode_start_time = time.time()

        # Fill and return the observation
        return self._getObservation()

    def render(self, mode='rgb_array', close=False):
        pass

    def _setVelocities(self, angles, normalized_velocities):
        """
        Sets velocities on the robot joints
        """
        for angle, velocity in zip(angles, normalized_velocities):
            # Unnormalize the velocity
            velocity *= self.pepper.joint_dict[angle].getMaxVelocity()

            position = self.pepper.getAnglesPosition(angle)
            lower_limit = self.pepper.joint_dict[angle].getLowerLimit()
            upper_limit = self.pepper.joint_dict[angle].getUpperLimit()

            if position <= lower_limit and velocity < 0.0:
                velocity = 0.0
            elif position >= upper_limit and velocity > 0.0:
                velocity = 0.0

            pybullet.setJointMotorControl2(
                self.pepper.robot_model,
                self.pepper.joint_dict[angle].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=velocity,
                force=self.pepper.joint_dict[angle].getMaxEffort())

    def _getBucketPosition(self):
        """
        Returns the position of the target bucket in the world
        """
        # Get the position of the bucket (goal) in the world
        bucket_pose, bucket_qrot = pybullet.getBasePositionAndOrientation(
            self.bucket)

        return bucket_pose, bucket_qrot



    def _getProjectilePosition(self):
        """
        Returns the position of the ball in the world
        """
        # Get the position of the projectile in the world
        project_pose, project_qrot = pybullet.getBasePositionAndOrientation(
            self.projectile)

        return project_pose, project_qrot

    def _getLinkPosition(self, link_name):
        """
        Returns the position of the link in the world frame
        """
        link_state = pybullet.getLinkState(
            self.pepper.robot_model,
            self.pepper.link_dict[link_name].getIndex())

        return link_state[0], link_state[1]

    def _computeProjectileSpawnPose(self):
        """
        Returns the ideal position for the projectile (in the robot's hand)
        """
        r_wrist_pose, _ = self._getLinkPosition("r_wrist")
        return [
            r_wrist_pose[0] + 0.04,
            r_wrist_pose[1] - 0.01,
            r_wrist_pose[2] + 0.064]

    def _computeBucket1SpawnPose(self):
        """
        Returns a spawning pose for the targeted bin
        """
        return [0.3, -0.22, 0.0]



    def normalize(self, values, range_min=-1.0, range_max=1.0):
        """
        Normalizes values (list) according to a specific range
        """
        zero_bound = [x - min(values) for x in values]
        range_bound = [
            x * (range_max - range_min) / (max(zero_bound) - min(zero_bound))
            for x in zero_bound]

        return [x - max(range_bound) + range_max for x in range_bound]

    def _getState(self, convergence_norm=0.15):
        """
        Gets the observation and computes the current reward. Will also
        determine if the episode is over or not, by filling the episode_over
        boolean.
        """
        reward = 0.0

        # Get position of the object and gripper pose in the odom frame
        projectile_pose, _ = self._getProjectilePosition()
        bucket1_pose, _ = self._getBucket1Position()

        # Finish episode after 4 time steps
        if self.episode_steps == 4:
            self.episode_over = True
            reward += 0

        # Fill the observation
        obs = self._getObservation()

        w1 = -0.47247753  # weight of left-right direction of ball
        w2 = -0.88134272  # weight of l2norm between ball and right bucket

        # Compute the reward
        if self.episode_over:
            states = np.array([state for state in self.states])
            l_r = states[:, 1].mean()
            x1, y1, z1 = states[:, 0], states[:, 1], states[:, 2]
            x2, y2, z2 = states[:, 3], states[:, 4], states[:, 5]
            l2normbuck1 = np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 - z2) ** 2)) #calculate l2-norm at every time step
            slope1 = list((l2normbuck1[1:] / l2normbuck1[:-1] - 1)) # calculate change in l2-norm over entire trajectory
            slope1_sum = np.array(slope1).sum() # calculate sum of change in l2-norm


            reward += (w1 * l_r + w2 * slope1_sum) # assign reward


            initial_proj_footprint = [
                self.initial_projectile_pose[0],
                self.initial_projectile_pose[1],
                0.0]

        # Add the reward to the episode reward
        self.episode_reward += reward

        # Update the previous projectile pose
        self.prev_projectile_pose = projectile_pose

        if self.episode_over:
            self.episode_number += 1

        return obs, reward

    def _getObservation(self):
        """
        Returns the observation

        Returns:
            obs - the list containing the observations
        """
        # Get position of the projectile and bucket in the odom frame
        projectile_pose, _ = self._getProjectilePosition()
        bucket1_pose, _ = self._getBucket1Position()


        # Get the position of the r_gripper in the odom frame (base footprint
        # is on the origin of the odom frame in the xp)
        # gripper_pose, gripper_rot = self._getLinkPosition("r_gripper")

        # Fill and return the observation
        return np.array([pose for pose in projectile_pose] + \
                        [pose for pose in bucket1_pose])

    def _setupScene(self):
        """
        Setup a scene environment within the simulation
        """
        self.client = self.simulation_manager.launchSimulation(gui=self.gui)
        self.pepper = self.simulation_manager.spawnPepper(
            self.client,
            spawn_ground_plane=True)

        self.pepper.goToPosture("Stand", 1.0)
        self.pepper.setAngles("RHand", 0.7, 1.0)
        self.pepper.setAngles(
            self.r_kinematic_chain,
            self.initial_stand,
            1.0)

        time.sleep(1.0)
        self.joints_initial_pose = self.pepper.getAnglesPosition(
            self.pepper.joint_dict.keys())

        pybullet.setAdditionalSearchPath("environment")
        self.bucket = pybullet.loadURDF(
            "trash.urdf",
            self._computeBucket1SpawnPose(),
            flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)



        # The initial pose of the projectile
        self.initial_projectile_pose = self._computeProjectileSpawnPose()

        self.projectile = pybullet.loadURDF(
            "ball.urdf",
            self.initial_projectile_pose)

        time.sleep(0.2)

        # Get position of the projectile in the odom frame
        self.prev_projectile_pose, _ = self._getProjectilePosition()

    def _resetScene(self):
        """
        Resets the scene for a new scenario
        """
        pybullet.resetBasePositionAndOrientation(
            self.pepper.robot_model,
            posObj=[0.0, 0.0, 0.0],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)

        self._hardResetJointState()

        # The initial pose of the projectile
        self.initial_projectile_pose = self._computeProjectileSpawnPose()

        pybullet.resetBasePositionAndOrientation(
            self.projectile,
            posObj=self.initial_projectile_pose,
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)
        pybullet.resetBasePositionAndOrientation(
            self.bucket,
            posObj=self._computeBucket1SpawnPose(),
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)



        # Get position of the object and gripper pose in the odom frame
        self.prev_projectile_pose, _ = self._getProjectilePosition()

    def _hardResetJointState(self):
        """
        Performs a hard reset on the joints of the robot, avoiding the robot to
        get stuck in a position
        """
        for joint, position in \
                zip(self.pepper.joint_dict.keys(), self.joints_initial_pose):
            pybullet.setJointMotorControl2(
                self.pepper.robot_model,
                self.pepper.joint_dict[joint].getIndex(),
                pybullet.VELOCITY_CONTROL,
                targetVelocity=0.0)
            pybullet.resetJointState(
                self.pepper.robot_model,
                self.pepper.joint_dict[joint].getIndex(),
                position)

    def close(self):
        """
        Terminates the environment
        """
        self.simulation_manager.stopSimulation(self.client)

    def record(self, epi):
        """
        Records and saves each episode into a mp4 file
        """
        pybullet.startStateLogging(
            pybullet.STATE_LOGGING_VIDEO_MP4,
            'aprel_trajectories/clips/' + self.file_name + '_' + str(epi) + '.mp4')
