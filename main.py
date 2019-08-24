import time

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class World(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self):
        self.viewer = None
        self.state = None

        self.done = False

        self.scr_wid = 600
        self.scr_hgt = 600
        # todo: put in actual coordinates in points
        na = Node('A', Point(1.000, 3.35))
        nb = Node('B', Point(1.000, 3.35))

        tb = Node('Bravo', Point(1.000, 3.35))
        tc = Node('Charlie', Point(1.000, 3.35))
        td = Node('Dingo', Point(1.000, 3.35))
        tf = Node('Foxtrot', Point(1.000, 3.35))
        tt = Node('Tango', Point(1.000, 3.35))
        tw = Node('Whiskey', Point(1.000, 3.35))

        ra = Track(tc, td, 5)
        rb = Track(td, tb, 5)

        rc = Track(tb, nb, 3)
        rd = Track(td, nb, 3)
        re = Track(tc, nb, 3)

        rf = Track(na, nb, 3)

        rg = Track(na, tf, 2)
        rh = Track(na, tt, 2)
        ri = Track(na, tw, 2)

        train = Train(track=re, dest=tb, dist=1.0, direction=-1)

        self.nodes = [na, nb]


        self.tracks = [
            ra, rb, rc, rd, re, rf, rg, rh, ri,
        ]

    def step(self, action):

        return self.state, 1.0, self.done, {}

    def reset(self):
        self.state = None
        self.done = False
        return self.state

    def render(self, mode='human'):

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class Track:
    def __init__(self, a, b, track_length):
        self.begins_at = a
        self.ends_at = b
        self.track_length = track_length


# class RoutingTable:
#     def __init__(self):
#         pass


class Train:
    def __init__(self, track, dest, dist, direction):
        self.on_track = track
        self.destination = dest
        self.distance_from_beginning_of_track = dist
        self.direction = direction  # -1 = from end to beginning, 1 = from beginning to end


class Node:
    def __init__(self, name, pos):
        self.name = name
        self.positon = pos


if __name__ == '__main__':
    render_fps = 20



    gym.envs.registration.register(
        id='world-v0',
        entry_point='main:World',
    )
    world = gym.make('world-v0')
    world.reset()
    for x in range(100000000):
        world.step(None)

        time.sleep(1 / render_fps)

