import numpy as np
from random import uniform, choice, gauss 
from scipy.constants import pi
from math import cos, sin


class MotionModel:
    def __init__(self, model: str):
        """
        motion models

        Parameters
        ----
        model: str
            options: 'no_movement', 'gauss_pedestrian', 'walking_pedestrian',
            'fast_vehicle', 'random'

        Attributes
        ----
        model: str
            motion model identification
        min_theta: float
            minimum direction angle change, in rad
        max_theta: float
            maximum direction angle change, in rad
        speed: float
            speed, in m/s
        direction: float
            direction angle, in rad
        """

        self.model = model
        self.min_theta = -pi
        self.max_theta = pi
        self.direction = 0
        self.speed = 0
        self.manage_model()

    def manage_model(self):
        self.direction = uniform(-pi, pi)
        if self.model == 'random':
            self.model = choice([
                'no_movement', 'gauss_pedestrian', 
                'walking_pedestrian'
            ])
        if self.model == 'no_movement':
            self.min_theta = 0
            self.max_theta = 1
            self.speed = 0
            self.scale = 0
        elif self.model == 'gauss_pedestrian':
            self.min_theta = -pi/2
            self.max_theta = pi/2
            self.speed = .7
            self.scale = 0.05
        elif self.model == 'walking_pedestrian':
            self.min_theta = -pi/18
            self.max_theta = pi/18
            self.speed = 1.37
            self.scale= .01
        elif self.model == 'fast_vehicle':
            self.min_theta = -pi/18
            self.max_theta = pi/18
            self.speed = 1000
            self.scale= .01
        else:
            raise Exception('Invalid motion model option.')

    def step(self, position, direction, dt, *kargs, **kwargs):
        dp1 = gauss(mu=direction, sigma=self.scale)
        pp1 = np.array(position)
        aux = np.array(pp1[:2]) + dt * self.speed * np.array((cos(dp1), sin(dp1)))
        pp1[:2] = aux
        pp1 = tuple(pp1.tolist())
        return pp1, dp1

    # def step(self, position, direction, dt, *kargs, **kwargs):
        # dp1 = direction + uniform(self.min_theta, self.max_theta)
        # pp1 = np.array(position) + dt * self.speed * np.array((cos(dp1), sin(dp1)))
        # pp1 = tuple(pp1.tolist())
    #     return pp1, dp1
