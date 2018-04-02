#!/usr/bin/env python3

"""
Python OpenGL practical application.
"""
# Python built-in modules
import os
# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions
import numpy as np                  # all matrix manipulations & OpenGL args
from transform import translate, rotate, scale, vec

class ColorMesh:

    def __init__(self, attributes, index=None):
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []
        self.index_size = 0
        nb_primitives, size = 0, 0
        print(index)
        for layout_index, buffer_data in enumerate(attributes):
            self.buffers += [GL.glGenBuffers(1)]
            nb_primitives, size = buffer_data.shape
            GL.glEnableVertexAttribArray(layout_index)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[layout_index])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, buffer_data, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(layout_index, size, GL.GL_FLOAT, False, 0, None)

        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index, GL.GL_STATIC_DRAW)
            self.index_size = len(index)
            self.arguments = (index.size, GL.GL_UNSIGNED_INT, None)
            print("LEN INDEX:",len(index))

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)


    def draw(self, primitive, color_shader):
    #def draw(self, projection, view, model, color_shader=None, color=(1, 1, 1, 1), **params):
        GL.glBindVertexArray(self.glid)
        #self.draw_command(primitive, *self.arguments)

        #if self.index_size != 0:
        if True:
            #GL.glDrawElements(primitive, self.index_size, GL.GL_UNSIGNED_INT, None)
            GL.glDrawElements(primitive, *self.arguments)

        else:
            GL.glDrawArrays(primitive, 0, 3)  # 3 <- ???
            GL.glBindVertexArray(0)

    def __del__(self):
        print("_del_ called")
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)


# -------------- 3D ressource loader -----------------------------------------
def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []     # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes
