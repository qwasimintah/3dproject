#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status


# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
from transform import translate, rotate, scale, vec, identity
from h_loader import *
from transform import Trackball, identity
from PIL import Image               # load images for textures
from itertools import cycle
from lib import * 

class Cylinder(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self):
        Node.__init__(self)
        self.add(*load('cylinder.obj'))  # just load the cylinder from file


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    files = os.listdir("sky")
    print(files)

    #cylinder_node = Node(name='my_cylinder', transform=translate(-1, 0, 0), color=(1, 0, 0.5, 1))
    #cylinder_node.add(Cylinder())
    viewer.add(TexturedPlane(files))

    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
