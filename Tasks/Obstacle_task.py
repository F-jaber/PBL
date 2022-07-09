###### code adapted from Softbank Robotics Research's qi-gym repository found here: https://github.com/softbankrobotics-research ######

import sys
import math
import os
import gym
import time
import argparse
from datetime import datetime
import numpy as np
from gym import spaces
import random
import pybullet
import pybullet_data
from qibullet import PepperVirtual
from qibullet import SimulationManager
from urllib.request import Request, urlopen


class RobotEnv(gym.Env):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2

    def __init__(self, gui=False, rec = False):
        # if you want to visualize the simulation set to true
        self.gui = gui
        # if you want to record, set to true
        self.rec = rec

        self.episode_start_time = None
        self.episode_over = False
        self.evaded_obstacle = False
        self.episode_reward = 0.0
        self.episode_number = 0
        self.episode_steps = 0
        self.file_name = 'Nav_env-v0'
        self.simulation_manager = SimulationManager()
        #save the states in this list to calculate the reward at the end
        self.states = []


        #set up environment
        self._setupScene()
        # observation space limits
        self.obs_lim = [-5, -5, -5, 5, 5, 5]

        self.observation_space = spaces.Box(
            low=np.array(self.obs_lim[:3]),
            high=np.array(self.obs_lim[3:]))


        #discrete action space
        self.action_space = spaces.Discrete(3)


    def step(self, action):
        """

        Parameters
        ----------
        action : list of len 4: [x,y,theta, dif of position between robot and object]

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

        self.episode_steps += 1

        self._setMovement(action)

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
        self.evaded_obstacle = False
        self._resetScene()
        if self.gui and self.rec:
            self.record(self.episode_number)
        # Reset the start time for the current episode
        self.episode_start_time = time.time()
        self.states = []
        # Fill and return the observation
        return self._getObservation()

    def render(self, mode='rgb_array', close=False):
        self.gui = True

    def _setMovement(self, action):
        """
        Sets distance for the robot to move
        """

        if action == self.STRAIGHT:
            action = [1,0,0]
        elif action == self.LEFT:
            action = [2,2,.5]
        else:
            action = [2,-2,-.5]

        self.pepper.moveTo(action[0], action[1], action[2], frame=PepperVirtual.FRAME_ROBOT,
                           _async=True)

    def _getObstaclePosition(self):
        """
        Returns the position of the obstacle in the world
        """
        # Get the position of the bucket (goal) in the world
        obstacle_pose, obstacle_qrot = pybullet.getBasePositionAndOrientation(
            self.obstacle)

        return obstacle_pose, obstacle_qrot

    def _getPositionDif(self):
        #calculates position difference between robot and obstacle
        obs_pos, _ = self._getObstaclePosition()
        pepper_posx, pepper_posy, pepper_orient = self.pepper.getPosition()
        dif_posx = obs_pos[0] - pepper_posx
        return dif_posx

    def _ObstacleSpawnPose(self):
        """
        Returns a spawning pose for the obstacle
        """

        # return [radius * np.cos(angle), radius * np.sin(angle), 0.0]
        return [2.0, 0.0, 0.0]



    def _getState(self, convergence_norm=0.15):
        """
        Gets the observation and computes the current reward. Will also
        determine if the episode is over or not, by filling the episode_over
        boolean.
        """
        reward = 0.0

        if self.episode_steps == 60:
            self.episode_over = True
            reward += 0
        # Fill the observation
        obs = self._getObservation()

        # Compute the reward
        w1 = -0.81376396 # pos dif weight
        w2 = -0.58119551 # left_right weight
        if self.episode_over:
            states = np.array([state for state in self.states])
            l_r = states[:, 1].mean()
            pos_dif = states[:, 0].min() -0.65
            reward += (w1 * pos_dif) + (w2 * l_r)

        # Add the reward to the episode reward
        self.episode_reward += reward

        if self.episode_over:
            self.episode_number += 1

        return obs, reward

    def _getObservation(self):
        """
        Returns the observation

        Returns:
            obs - the list containing the observations
        """
        # get pepper's and the obstacle's psotion
        obstacle_pose, _ = pybullet.getBasePositionAndOrientation(
            self.obstacle)
        pepper_pos = self.pepper.getPosition()
        # calculate relative postion vector
        relative_pos = np.subtract(obstacle_pose, pepper_pos)

        return relative_pos

    def _setupScene(self):
        """
        Setup a scene environment within the simulation
        """
        #spawn pepper
        self.client = self.simulation_manager.launchSimulation(gui=self.gui)
        self.pepper = self.simulation_manager.spawnPepper(
            self.client,
            spawn_ground_plane=True)

        self.pepper.goToPosture("Stand", 0.6)

        #spawn obstacle
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.obstacle = pybullet.loadURDF(
            "block.urdf",
            basePosition=self._ObstacleSpawnPose(),
            globalScaling=10.0,
            flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.pepper.moveTo(10, 0, 0, frame=PepperVirtual.FRAME_ROBOT, _async=True)

    def _resetScene(self):
        """
        Resets the scene for a new scenario
        """
        pybullet.resetBasePositionAndOrientation(
            self.pepper.robot_model,
            posObj=[0.0, 0.0, 0.0],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)

        pybullet.resetBasePositionAndOrientation(
            self.obstacle,
            posObj=self._ObstacleSpawnPose(),
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)


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
            self.file_name + '_' + str(epi) + '.mp4')
