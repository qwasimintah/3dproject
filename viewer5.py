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
from transform import translate, rotate, scale, vec
from loader import *




# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object


# ------------  Simple color shaders ------------------------------------------
COLOR_VERT = """#version 330 core
//uniform mat4 matrix;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
smooth out vec3 outsmooth;
void main() {
    //gl_Position = matrix * vec4(position, 1);
    gl_Position = vec4(position, 1);
    outsmooth = color;

}"""

COLOR_FRAG = """#version 330 core
out vec4 outColor;
smooth in vec3 outsmooth;
void main() {
    ///outColor = vec4(1, 0, 0, 1);
    outColor = vec4(outsmooth, 1);

}"""

# ------------  Scene object classes ------------------------------------------
class VertexArray:

    def __init__(self, attributes, index=None):
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []
        self.index_size = 0
        for layout_index, buffer_data in enumerate(attributes):
            self.buffers += [GL.glGenBuffers(1)]
            GL.glEnableVertexAttribArray(layout_index)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[layout_index])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, buffer_data, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(layout_index, 3, GL.GL_FLOAT, False, 0, None)

        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index, GL.GL_STATIC_DRAW)
            self.index_size = len(index)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)


    def draw(self, primitive, color_shader):

        # GL.glUseProgram(color_shader.glid)

        GL.glBindVertexArray(self.glid)
        #transform = (translate(-.5) @ scale(.5) @ rotate(vec(1,0,0), 45))
        #if transform is not None:
        #    matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        #    GL.glUniformMatrix4fv(matrix_location, 1, True,  transform)

        if self.index_size != 0:
            #print("Index size : ", self.index_size)
            GL.glDrawElements(primitive, self.index_size, GL.GL_UNSIGNED_INT, None)

        else:
            GL.glDrawArrays(primitive, 0, 3)  # 3 <- ???
        GL.glBindVertexArray(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)


class SimplePyramid:

    def __init__(self):

        # triangle position buffer
        position = np.array(((-.5,0,.5), (.5,0,.5), (.5,0,-.5), (-.5,0,-.5), (0,1,0)), 'f')

        color = np.array(((1,0,0),(1,1,0),(0,0,1),(1,0,1),(0,1,0)), 'f')

        index = np.array((4,0,1,4,1,2,4,2,3,4,3,0,0,4,1,1,4,2),np.uint32)


        self.vertex_array = VertexArray([position, color], index)
        """
        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below

        self.buffers = [GL.glGenBuffers(1)]  # create buffer for position attrib

        # bind the vbo, upload position data to GPU, declare its size and type
        GL.glEnableVertexAttribArray(0)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, position, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)



        self.buffers += [GL.glGenBuffers(1)]
        GL.glEnableVertexAttribArray(1)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[1])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, color, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)



        self.buffers += [GL.glGenBuffers(1)]
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        """

    def draw(self, projection, view, model, color_shader):
        GL.glUseProgram(color_shader.glid)

        transform = (translate(-.5) @ scale(.5) @ rotate(vec(1,0,0), 45))
        if transform is not None:
            matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
            GL.glUniformMatrix4fv(matrix_location, 1, True,  transform)

        self.vertex_array.draw(GL.GL_TRIANGLES, color_shader)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        #GL.glBindVertexArray(self.glid)

        #matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        #GL.glUniformMatrix4fv(matrix_location, 1, True,  (translate(-.5) @ scale(.5) @ rotate(vec(1,0,0), 45))) 
        #GL.glDrawElements(GL.GL_TRIANGLES, 18, GL.GL_UNSIGNED_INT, None)
        #GL.glBindVertexArray(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)


# ------------  Viewer class & window management ------------------------------
class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        # compile and initialize shader programs once globally

        # initially empty list of object to draw
        self.drawables = []

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # draw our scene objects
            for (i, drawable) in enumerate(self.drawables):
                self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
                # drawable.draw(None, None, None, self.color_shader)
                GL.glUseProgram(self.color_shader.glid)

                #transform = (translate(-.5) @ scale(.5) @ rotate(vec(1,0,0), 45))
                #matrix_location = GL.glGetUniformLocation(self.color_shader.glid, 'matrix')
                #GL.glUniformMatrix4fv(matrix_location, 1, True,  transform)

                #drawable.vertex_array.draw(GL.GL_TRIANGLES)
                #drawable.draw(None, None, None, self.color_shader)
                drawable.draw(GL.GL_TRIANGLES, self.color_shader)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)


class SimplePyramidPlain:

    def __init__(self):

        # triangle position buffer
        """
        position = np.array(((-.5,0,.5), (.5,0,.5), (-.5,0,-.5), (0,1,0),
                             (.5,0,.5), (.5,0,-.5), (-.5,0,-.5), (0,1,0),
                             (-.5,0,.5), (.5,0,.5), (-.5,0,-.5), (0,1,0),
                             (.5,0,.5), (.5,0,-.5), (-.5,0,-.5), (0,1,0),
                             ), 'f')

        color = np.array(((1,0,0),(1,1,0),(1,0,1),(0,1,0),
                          (1,1,0),(0,0,1),(1,0,1),(0,1,0),
                          (1,0,0),(1,1,0),(1,0,1),(0,1,0),
                          (1,1,0),(0,0,1),(1,0,1),(0,1,0),
                          ), 'f')

        index = np.array((4,0,1,4,1,2,4,2,3,4,3,0,0,4,1,1,4,2),np.uint32)
        """


        position = np.array(((-.5,0,.5),(-.5,0,.5),(-.5,0,.5),(.5,0,.5), (.5,0,.5), (.5,0,.5), (.5,0,.5),(.5,0,-.5),(.5,0,-.5),(.5,0,-.5),(.5,0,-.5),(-.5,0,-.5),(-.5,0,-.5),(-.5,0,-.5),(0,1,0),(0,1,0),(0,1,0),(0,1,0)), 'f')

        b = (0,0,1)
        g = (0,1,0)
        r = (1,0,0)
        y = (1,1,0)
        m = (1,0,1)
        c = (0,1,1)
        color = np.array((b,y,m,b,g,m,c,g,r,c,r,y,m,c,b,g,r,y), 'f')
        index = np.array((0,3,14,4,7,15,8,10,16,11,1,17,25,12,6,13,9), np.uint32);


        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below

        self.buffers = [GL.glGenBuffers(1)]  # create buffer for position attrib

        # bind the vbo, upload position data to GPU, declare its size and type
        GL.glEnableVertexAttribArray(0)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, position, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)



        self.buffers += [GL.glGenBuffers(1)]
        GL.glEnableVertexAttribArray(1)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[1])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, color, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)



        self.buffers += [GL.glGenBuffers(1)]
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def draw(self, projection, view, model, color_shader):
        GL.glUseProgram(color_shader.glid)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)

        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        GL.glUniformMatrix4fv(matrix_location, 1, True,  (translate(.5) @ scale(.5) @ rotate(vec(1,0,0), -45))) 
        #GL.glDrawElements(GL.GL_TRIANGLES, index.size, GL.GL_UNSIGNED_INT, None)
        GL.glDrawElements(GL.GL_TRIANGLES, 18, GL.GL_UNSIGNED_INT, None)
        #GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # place instances of our basic objects
    #viewer.add(SimplePyramid())
    viewer.add(load("suzanne.obj")[0])
    #viewer.add(SimplePyramidPlain())

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
