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
from transform import translate, rotate, scale, vec , identity
from transform import frustum, perspective


from transform import translate, rotate, scale, vec



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
<<<<<<< HEAD:viewer.py
COLOR_VERT = """#version 330 core
uniform mat4 matrix;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
smooth out vec3 outcolour;
void main() {
    gl_Position = matrix * vec4(position, 1);
    outcolour = color;
=======
COLOR_VERT = """#version 330 
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
smooth out vec3 outposition;
smooth out vec3 outcolors;
uniform mat4 matrix;
uniform mat4 projection;

void main() {
    gl_Position = projection* matrix * vec4(position, 1);
    outposition = position;
    outcolors = color;
>>>>>>> 3daed8aadd8ffbd395badb1fcffd181c9e5eaf08:viewer_den.py

}"""

COLOR_FRAG = """#version 330 core
out vec4 outColor;
<<<<<<< HEAD:viewer.py
//uniform float red;
//uniform float blue;
//uniform float green;
//smooth in vec3 outposition;
smooth in vec3 outcolour;
void main() {
    //outColor = vec4(outposition, 1);
    outColor = vec4(outcolour, 1);
=======
uniform vec3 color;
smooth in vec3 outposition;
smooth in vec3 outcolors;
void main() {
    outColor = vec4(color,1);
    //outColor = vec4(outposition,1);
>>>>>>> 3daed8aadd8ffbd395badb1fcffd181c9e5eaf08:viewer_den.py

}"""

# ------------  Scene object classes ------------------------------------------
class SimpleTriangle:
    """Hello triangle object"""

    def __init__(self):

<<<<<<< HEAD:viewer.py
        # triangle position buffer
        #position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')
        position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')
        color = np.array(((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)), 'f')

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below

=======
        offset = .0
        position = np.array(((0, .5+offset, 0), (.5+offset, -.5+offset, 0), (-.5+offset, -.5+offset, 0)), 'f')

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below
        self.buffers = [GL.glGenBuffers(1)]  # create buffer for position attrib
        print(self.buffers)
>>>>>>> 3daed8aadd8ffbd395badb1fcffd181c9e5eaf08:viewer_den.py
        # bind the vbo, upload position data to GPU, declare its size and type
        self.buffers = [GL.glGenBuffers(1)]  # create buffer for position attrib
        GL.glEnableVertexAttribArray(0)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, position, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

<<<<<<< HEAD:viewer.py
        ###

        #self.buffers = [GL.glGenBuffers(1)]  # create buffer for position attrib
        self.buffers_c = [GL.glGenBuffers(1)]  # create buffer for position attrib
        GL.glEnableVertexAttribArray(1)      # assign to layout = 0 attribute
=======



        # triangle color buffer
        color = np.array(((1, 0, 0), (0, 1, .0), (0, 0, 1)), 'f')
        #self.glid_c = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        #GL.glBindVertexArray(self.glid_c)      # activate to receive state below
        self.buffers_c = [GL.glGenBuffers(1)]  # create buffer for position attrib
        print(self.buffers_c)
        # bind the vbo, upload position data to GPU, declare its size and type
        GL.glEnableVertexAttribArray(1)      # assign to layout = 1 attribute
>>>>>>> 3daed8aadd8ffbd395badb1fcffd181c9e5eaf08:viewer_den.py
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers_c[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, color, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)

<<<<<<< HEAD:viewer.py
=======

>>>>>>> 3daed8aadd8ffbd395badb1fcffd181c9e5eaf08:viewer_den.py
        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 1)




    def draw(self, projection, view, model, color_shader):

        ###

<<<<<<< HEAD:viewer.py

        GL.glUseProgram(color_shader.glid)
=======
>>>>>>> 3daed8aadd8ffbd395badb1fcffd181c9e5eaf08:viewer_den.py
        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        


    def __del__(self):
<<<<<<< HEAD:viewer.py
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, [self.buffers, self.buffers_c])
=======
        c = []
        c.append(self.glid)

        buffer_list =[]
        buffer_list.append(self.buffers);
        buffer_list.append(self.buffers_c)
>>>>>>> 3daed8aadd8ffbd395badb1fcffd181c9e5eaf08:viewer_den.py

        GL.glDeleteVertexArrays(1, c)
        GL.glDeleteBuffers(1, buffer_list)

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


        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)


        # initially empty list of object to draw
        self.drawables = []

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(None, None, None, self.color_shader)


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

        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_LEFT:
                color_loc = GL.glGetUniformLocation(self.color_shader.glid, 'color')
                GL.glUniform3fv(color_loc, 1, (0.6, 0.6, 0.9))

            if key == glfw.KEY_RIGHT:
                color_loc = GL.glGetUniformLocation(self.color_shader.glid, 'color')
                GL.glUniform3fv(color_loc, 1, (1, 1, 1))


            if key == glfw.KEY_DOWN:
                color_loc = GL.glGetUniformLocation(self.color_shader.glid, 'color')
                GL.glUniform3fv(color_loc, 1, (0, 1, 0))

            if key == glfw.KEY_UP:
                color_loc = GL.glGetUniformLocation(self.color_shader.glid, 'color')
                GL.glUniform3fv(color_loc, 1, (1, 0, 0))

            if key == glfw.KEY_R:
                matrix_location = GL.glGetUniformLocation(self.color_shader.glid, 'matrix')
<<<<<<< HEAD:viewer.py
                GL.glUniformMatrix4fv(matrix_location, 1, True, rotate(vec(0, 1, 0), 45))
                """
                red_location = GL.glGetUniformLocation(self.color_shader.glid, 'red')
                green_location = GL.glGetUniformLocation(self.color_shader.glid, 'green')
                blue_location = GL.glGetUniformLocation(self.color_shader.glid, 'blue')
                GL.glUniform1f(red_location, 1.0)
                GL.glUniform1f(green_location, 0.0)
                GL.glUniform1f(blue_location, 0.0)
                """
=======
                GL.glUniformMatrix4fv(matrix_location, 1, True, rotate(vec(0, 1, 0), 60))

            if key == glfw.KEY_W:
                matrix_location = GL.glGetUniformLocation(self.color_shader.glid, 'matrix')
                projection_location = GL.glGetUniformLocation(self.color_shader.glid, 'projection')
                persp = perspective(10, 1, 0.01, 10000)
                GL.glUniformMatrix4fv(matrix_location, 1, True, identity())
                GL.glUniformMatrix4fv(projection_location, 1, True, persp)




>>>>>>> 3daed8aadd8ffbd395badb1fcffd181c9e5eaf08:viewer_den.py


# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # place instances of our basic objects
    viewer.add(SimpleTriangle())

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
