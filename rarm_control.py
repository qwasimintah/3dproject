#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions
from transform import translate, rotate, scale, vec, identity
#from h_loader import *
from transform import Trackball, identity
from solution_shared_2 import * 


# class Node:
#     """ Scene graph transform and parameter broadcast node """
#     def __init__(self, name='', children=(), transform=identity(), **param):
#         self.transform, self.param, self.name = transform, param, name
#         self.children = list(iter(children))

#     def add(self, *drawables):
#         """ Add drawables to this node, simply updating children list """
#         self.children.extend(drawables)

#     def draw(self, projection, view, win, model, **param):
#         """ Recursive draw, passing down named parameters & model matrix. """
#         # merge named parameters given at initialization with those given here
#         param = dict(param, **self.param)
#         model = self.transform @ model
#         for child in self.children:
#             child.draw(projection, view, model, **param)


class Cylinder(Node):
    """ Very simple tyronasaurus based on practical 2 load function """
    def __init__(self):
        Node.__init__(self)
        self.add(*load('cylinder.obj'))  # just load the cylinder from file



class RotationControlNode(Node):
    def __init__(self, key_up, key_down, axis, angle=0, **param):
        super().__init__(**param)   # forward base constructor named arguments
        self.angle, self.axis = angle, axis
        self.key_up, self.key_down = key_up, key_down

    def draw(self, projection, view, model, win=None, **param):
        assert win is not None
        self.angle += 2 * int(glfw.get_key(win, self.key_up) == glfw.PRESS)
        self.angle -= 2 * int(glfw.get_key(win, self.key_down) == glfw.PRESS)
        self.transform = identity() @ self.angle

        # call Node's draw method to pursue the hierarchical tree calling
        super().draw(projection, view, model,**param)



# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    #viewer.add(*[mesh for file in sys.argv[1:] for mesh in load(file)])
    #viewer.add(*[mesh for file in ["trex.obj"] for mesh in load(file)])

    #cylinder_node = Node(name='my_cylinder', transform=translate(-1, 0, 0), color=(1, 0, 0.5, 0))
    cylinder = Cylinder()             # re-use same cylinder instance
    limb_shape = Node(transform=translate(0, 1, 0) @ rotate(vec(1,0,0), -30))  # make a thin cylinder
    limb_shape.add(cylinder)          # common shape of arm and forearm
    arm_node = RotationControlNode(glfw.KEY_LEFT, glfw.KEY_RIGHT, vec(0, 1, 0), children=[cylinder])
    #arm_node = RoataNode(transform=translate(0, 2, 0) @ rotate(vec(1,0,0), 0) @  scale(0.1, 0.4, 0.1) , children=[cylinder])    # robot arm rotation with phi angle
    arm_node.add(limb_shape)

    # make a flat cylinder
    base_shape = Node(transform=translate(0, 0, 0) @ scale(0.3, 0.3, 0.3), children=[cylinder])
    base_node = Node(transform=rotate(vec(1,0,0),0) @ translate(0,0,0))   # robot base rotation with theta angle
    base_node.add(base_shape, arm_node)
    viewer.add(base_node)


    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
