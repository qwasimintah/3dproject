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
from PIL import Image               # load images for textures


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
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 fragColor;

void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    fragColor = color;
}"""


COLOR_FRAG = """#version 330 core
in vec3 fragColor;
out vec4 outColor;
void main() {
    outColor = vec4(fragColor, 1);
}"""


class ColorMesh:

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, color_shader):

        names = ['view', 'projection', 'model']
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
        GL.glUseProgram(color_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.draw(GL.GL_TRIANGLES)


# ------------  Scene object classes ------------------------------------------
class VertexArray:
    """helper class to create and self destroy vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            attribs should be list of arrays with dim(0) indexed by vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load a buffer per initialized vertex attribute (=dictionary)
        for loc, data in enumerate(attributes):
            if data is None:
                continue

            # bind a new vbo, upload its data to GPU, declare its size and type
            self.buffers += [GL.glGenBuffers(1)]
            data = np.array(data, np.float32, copy=False)
            print(data)
            nb_primitives, size = data.shape
            print(nb_primitives,size)
            GL.glEnableVertexAttribArray(loc)  # activates for current vao only
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    #def draw(self, primitive):
    def draw(self, projection, view, model, color_shader):
        """draw a vertex array, either as direct array or indexed array"""
        GL.glBindVertexArray(self.glid)


        names = ['view', 'projection', 'model']
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
        GL.glUseProgram(color_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        #self.vertex_array.draw(GL.GL_TRIANGLES)
        #print(*self.arguments)
        self.draw_command(GL.GL_TRIANGLES, *self.arguments)
        #GL.glDrawArrays(GL.GL_TRIANGLES, self.arguments[0], GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)



    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)


class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, name='', children=(), transform=identity(), **param):
        self.transform, self.param, self.name = transform, param, name
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model, **param):
        """ Recursive draw, passing down named parameters & model matrix. """
        # merge named parameters given at initialization with those given here
        param = dict(param, **self.param)
        model = self.transform @ model
        for child in self.children:
            child.draw(projection, view, model, **param)

class ColorMesh:

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, color_shader):

        names = ['view', 'projection', 'model']
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
        GL.glUseProgram(color_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.draw(GL.GL_TRIANGLES)

class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
        # helper array stores texture format for every pixel size 1..4
        format = [GL.GL_LUMINANCE, GL.GL_LUMINANCE_ALPHA, GL.GL_RGB, GL.GL_RGBA]
        try:
            # imports image as a numpy array in exactly right format
            tex = np.array(Image.open(file))
            format = format[0 if len(tex.shape) == 2 else tex.shape[2] - 1]
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, format, GL.GL_UNSIGNED_BYTE, tex)

            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)


class CubeTexture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, files=[], wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.glid)
        # helper array stores texture format for every pixel size 1..4
        format = [GL.GL_LUMINANCE, GL.GL_LUMINANCE_ALPHA, GL.GL_RGB, GL.GL_RGBA]
        sky_faces = ['morning_lf.png', 'morning_rt.png', 'morning_up.png', 'morning_dn.png', 'morning_ft.png', 'morning_bk.png']
        try:
            for i,file in enumerate(sky_faces):
            # imports image as a numpy array in exactly right format
                tex = np.array(Image.open("hw_morning/"+file))
                #print(i)
                format1 = format[0 if len(tex.shape) == 2 else (tex.shape[2] - 1)]
                GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_RGBA, tex.shape[1],
                                tex.shape[0], 0, format1, GL.GL_UNSIGNED_BYTE, tex)

                # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
                # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
                # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
                # GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
                # GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
                message = 'Loaded texture %s\t(%s, %s, %s, %s)'
                print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))

            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
            
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)
        #GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)



# -------------- Example texture plane class ----------------------------------
TEXTURE_VERT = """#version 330 core
uniform mat4 modelviewprojection;
layout(location = 0) in vec3 position;
//out vec3 fragTexCoord;
varying mediump vec3 fragTexCoord;
void main() {
    gl_Position = modelviewprojection * vec4(position, 1);
    //gl_Position = gl_Position.xyww;
    //fragTexCoord = normalize(position.xyz);
    fragTexCoord = position;
}"""

TEXTURE_FRAG = """#version 330 core
uniform samplerCube diffuseMap;
varying mediump vec3 fragTexCoord;
//in vec3 fragTexCoord;
out vec4 outColor;
void main() {
    outColor = texture(diffuseMap, fragTexCoord);
}"""


class TexturedPlane:
    """ Simple first textured object """

    def __init__(self, file):
        # feel free to move this up in the viewer as per other practicals
        self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        # triangle and face buffers
        vertices = 100 * np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)

        #print(vertices)
        #faces = np.array(((0, 1, 2), (0, 2, 3)), np.uint32)
        #self.vertex_array = VertexArray([vertices], faces)
        skybox_right = iter([1, -1, -1, 1, -1,  1, 1,  1,  1, 1,  1,  1, 1,  1, -1, 1, -1, -1])
        skybox_left =iter([-1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1])
        skybox_top = iter([-1,  1, -1, 1,  1, -1, 1,  1,  1, 1,  1,  1, -1,  1,  1, -1,  1, -1])
        skybox_bottom = iter( [-1, -1, -1, -1, -1,  1, 1, -1, -1, 1, -1, -1, -1, -1,  1, 1, -1,  1])
        skybox_back = iter([-1,  1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1,  1, -1, -1,  1, -1])
        skybox_front = iter([-1, -1,  1, -1,  1,  1, 1,  1,  1, 1,  1,  1, 1, -1,  1, -1, -1,  1])
        skybox = iter(SKY_BOX_VERTICES)
        sk_right_triangle = [(x,next(skybox), next(skybox)) for x in skybox]
        #sk_right_triangle = [(x, next(y), next(y)) for y in [skybox_right, skybox_left, skybox_top, skybox_bottom, skybox_back, skybox_front] for x in y ]
        print(sk_right_triangle)
        #skybox_vertices = np.array([skybox_right, skybox_left, skybox_top, skybox_bottom, skybox_back, skybox_front], dtype=np.float32)
        #print(skybox_vertices)
        self.vertex_array = VertexArray([sk_right_triangle])

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.file = file

        # setup texture and upload it to GPU
        self.texture = CubeTexture(file, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, win=None, **_kwargs):

        # some interactive elements
        if win!=None:
            if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
                self.wrap_mode = next(self.wrap)
                self.texture = CubeTexture(self.file, self.wrap_mode, *self.filter_mode)

            if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
                self.filter_mode = next(self.filter)
                self.texture = CubeTexture(self.file, self.wrap_mode, *self.filter_mode)
        GL.glUseProgram(self.shader.glid)

        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP,self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.draw(projection, view, model, self.shader)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
        GL.glUseProgram(0)


def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes




class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])


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
        self.trackball = GLFWTrackball(self.win)

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
        #GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        #GL.glFrontFace ( GL.GL_CW )
        #GL.glEnable(GL.GL_TEXTURE_CUBE_MAP)
        #GL.glEnable(GL.GL_TEXTURE_2D)    
        #GL.glEnable(GL.GL_RGB);
        GL.glCullFace( GL.GL_BACK)
        GL.glDepthFunc(GL.GL_LEQUAL);
        #GL.glDepthFunc(GL.GL_LESS);
        GL.glDepthMask(GL.GL_TRUE);
        #glDepthFunc(GL_LEQUAL);
        # compile and initialize shader programs once globally

        # initially empty list of object to draw
        self.drawables = []

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # draw our scene objects
            self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
            for (i, drawable) in enumerate(self.drawables):
                winsize = glfw.get_window_size(self.win)
                view = self.trackball.view_matrix()
                projection = self.trackball.projection_matrix(winsize)
                #self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
                # drawable.draw(None, None, None, self.color_shader)
                GL.glUseProgram(self.color_shader.glid)

                #transform = (translate(-.5) @ scale(.5) @ rotate(vec(1,0,0), 45))
                #matrix_location = GL.glGetUniformLocation(self.color_shader.glid, 'matrix')
                #GL.glUniformMatrix4fv(matrix_location, 1, True,  transform)

                #drawable.vertex_array.draw(GL.GL_TRIANGLES)
                drawable.draw(projection, view, identity(), win=self.win, color_shader=self.color_shader)
                #drawable.draw(GL.GL_TRIANGLES, self.color_shader)

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

class Trex(Node):
    """ Very simple tyronasaurus based on practical 2 load function """
    def __init__(self):
        Node.__init__(self)
        self.add(*load('trex.obj'))  # just load the cylinder from file


SKY_BOX_VERTICES = [-1.0,  1.0, -1.0,
    -1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
     1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0, -1.0,  1.0,
    -1.0, -1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0,  1.0,
    -1.0, -1.0,  1.0,
     1.0, -1.0, -1.0,
     1.0, -1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0, -1.0,
     1.0, -1.0, -1.0,
    -1.0, -1.0,  1.0,
    -1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0, -1.0,  1.0,
    -1.0, -1.0,  1.0,
    -1.0,  1.0, -1.0,
     1.0,  1.0, -1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
    -1.0,  1.0, -1.0,
    -1.0, -1.0, -1.0,
    -1.0, -1.0,  1.0,
     1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
    -1.0, -1.0,  1.0,
     1.0, -1.0,  1.0]

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    #viewer.add(*[mesh for file in sys.argv[1:] for mesh in load(file)])
    #viewer.add(*[mesh for file in ["trex.obj"] for mesh in load(file)])

    #cylinder_node = Node(name='my_cylinder', transform=translate(-1, 0, 0), color=(1, 0, 0.5, 0))
    cylinder_node = Node(name='my_cylinder', transform=translate(-1, 0, 0))
    cylinder_node.add(Trex())
    #viewer.add(cylinder_node)
    viewer.add(TexturedPlane('grass.png'))




    #if len(sys.argv) < 2:
    if False: 
        print('Usage:\n\t%s [3dfile]*\n\n3dfile\t\t the filename of a model in'
              ' format supported by pyassimp.' % (sys.argv[0],))

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
