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
from transform import *
from itertools import cycle

import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions
from bisect import bisect_left
import copy

class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, name='', children=(), transform=identity(), **param):
        self.transform, self.param, self.name = transform, param, name
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, local_model, **param):
        """ Recursive draw, passing down named parameters & model matrix. """
        # merge named parameters given at initialization with those given here
        param.update(self.param)
        model =  local_model @ self.transform
        for child in self.children:
            child.draw(projection, view, model, **param)

class RotationControlNode(Node):
    def __init__(self, key_up, key_down, axis, angle=0, **param):
        super().__init__(**param)   # forward base constructor named arguments
        self.angle, self.axis = angle, axis
        self.key_up, self.key_down = key_up, key_down

    def draw(self, projection, view, model, win=None, **param):
        assert win is not None
        self.angle += 2 * int(glfw.get_key(win, self.key_up) == glfw.PRESS)
        self.angle -= 2 * int(glfw.get_key(win, self.key_down) == glfw.PRESS)
        #self.transform = rotate(self.axis, self.angle) @ self.transform
        self.transform =  self.transform @ rotate(self.axis, self.angle)
        self.angle = 0

        # call Node's draw method to pursue the hierarchical tree calling
        super().draw(projection, view, model, win=win, **param)

class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        #print("[", self.times[0]," - ", self.times[len(self.times) - 1], "]")
        if (time <= self.times[0]):
            return self.values[0]
        if (time > self.times[len(self.times) - 1]):
            return self.values[len(self.times) - 1]
        # 2. search for closest index entry in self.times, using bisect_left function
        insert_index = bisect_left(self.times, time)
        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        fraction = (time - self.times[insert_index - 1])/(self.times[insert_index] - self.times[insert_index - 1])
        return self.interpolate(self.values[insert_index - 1], self.values[insert_index], fraction)

class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        #print(type(translate_keys[0]))
        self.translate_keys = KeyFrames(translate_keys, interpolation_function=lerp)
        #print("stk : ", np.shape(self.translate_keys.values[0]))
        self.scale_keys = KeyFrames(scale_keys, interpolation_function=lerp)
        self.rotate_keys = KeyFrames(rotate_keys, interpolation_function=quaternion_slerp)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        translate_matrix = translate(*self.translate_keys.value(time))
        scale_matrix = scale(self.scale_keys.value(time))
        rotate_matrix = quaternion_matrix(self.rotate_keys.value(time))
        return translate_matrix @ rotate_matrix @ scale_matrix

class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, translate_keys, rotate_keys, scale_keys, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model, **param)


class Sphere(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self):
        super().__init__()
        self.add(*load('sphere.obj', (1,1,1)))  # just load the cylinder from file

class Sun(Node):
    """ Very simple tyranosaurus based on practical 2 load function """
    def __init__(self):
        Node.__init__(self)
        self.add(*load('sun.obj', (1, 1, 0)))  # just load the cylinder from file

class FlyingDinosaur(Node):
    """ Very simple tyranosaurus based on practical 2 load function """
    def __init__(self):
        Node.__init__(self)
        self.add(*load('Pterodactyl/Pterodactyl.obj', (.4, .4, .4)))  # just load the cylinder from file


def newSphereLoad(file):
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []     # error reading => return empty list

    # lenght=0
    # for m in scene.meshes:
    #     new_vertices = m.vertices.copy()
    #     for vertex in new_vertices:
    #         vertex[1] = 1

    #     # new_vertices = new_vertices + 50
    #     m.vertices = np.concatenate((m.vertices, new_vertices))

    #     new_normals = m.normals.copy()
    #     m.normals = np.concatenate((m.normals, new_normals))

    #     new_faces = m.faces.copy()

    #     m.faces = np.concatenate((m.faces, new_faces))


    # print (scene.meshes[0].vertices[0])
    # l =len(scene.meshes[0].vertices)/2
    # print(l)
    # print (scene.meshes[0].vertices[int(l)])
        # vertices.append(m.vertices)
        # normals.append(m.normals)
        # faces.append(m.faces)


    for m in scene.meshes:
        copied_m = copy.deepcopy(m)
        for vertex in copied_m.vertices:
            vertex[1] = 1

        scene.meshes.append(copied_m)


    # for m in new_meshes:
    #     new_meshes.append(m)
    #     new_mesh =  copy.deepcopy(m)
    #     new_mesh.vertices = new_mesh.vertices + 50
    #     new_meshes.append(new_mesh)

    # meshes = VertexArray([vertices, normals], faces)
    meshes = [VertexArray([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes

# -------------- 3D ressource loader -----------------------------------------
def load(file, color=None):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []     # error reading => return empty list

    meshes = [VertexArray([m.vertices, m.normals], m.faces, color) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes

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
out vec3 colors;
uniform mat4 P;
uniform mat4 V;
uniform mat4 M;
uniform vec3 temp2;
void main() {
    gl_Position = P * V * M * vec4(position, 1);
    colors = temp2;
    //colors = vec3(1,1,1);

}"""

COLOR_FRAG = """#version 330 core
out vec4 outColor;
in vec3 colors;
void main() {
    outColor = vec4(colors, 1);
}"""

class VertexArray:
    def __init__(self, attributes, index=None, color=None):
        self.index = index
        self.attributes = attributes
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []
        self.color = color
        for layout_index, buffer_data in enumerate(attributes):
            self.buffers += [GL.glGenBuffers(1)]
            GL.glEnableVertexAttribArray(layout_index)      # assign to layout = 0 attribute
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[layout_index])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, buffer_data, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(layout_index, 3, GL.GL_FLOAT, False, 0, None)

        # if index != None:
        self.buffers += [GL.glGenBuffers(1)]
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[len(attributes)])
        print (len(attributes))
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index, GL.GL_STATIC_DRAW)


        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def draw(self, projection, view, model, color_shader, **param):
        GL.glUseProgram(color_shader.glid)
        
        P = GL.glGetUniformLocation(color_shader.glid, 'P')
        GL.glUniformMatrix4fv(P, 1, True, projection)
        
        V = GL.glGetUniformLocation(color_shader.glid, 'V')
        GL.glUniformMatrix4fv(V, 1, True, view)
        
        M = GL.glGetUniformLocation(color_shader.glid, 'M')
        GL.glUniformMatrix4fv(M, 1, True, model)

        if (self.color):
            temp = GL.glGetUniformLocation(color_shader.glid, 'temp2')
            GL.glUniform3fv(temp, 1, self.color)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
                
        GL.glBindVertexArray(self.glid)   # activate our vertex array
        GL.glDrawElements(GL.GL_TRIANGLES, self.index.size, GL.GL_UNSIGNED_INT, None)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)


class SimplePyramid:
    def __init__(self):

        # triangle position buffer
        self.position = np.array(
            (
                (0, 1, 0), 
                (-.5, 0, -.5), 
                (-.5, 0, .5), 
                (.5, 0, .5), 
                (.5, 0, -.5)
            ), 
            np.float32
        )
        self.index = np.array((
            0, 1, 2, 
            0, 2, 3, 
            0, 3, 4,
            0, 4, 1,
            1, 3, 4,
            1, 2, 3
        ), np.uint32)

        color = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'f')

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below
        self.buffers = [GL.glGenBuffers(1), GL.glGenBuffers(1), GL.glGenBuffers(1)]  # create buffer for position attrib

        # bind the vbo, upload position data to GPU, declare its size and type
        GL.glEnableVertexAttribArray(0)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.position, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

        # bind the vbo, upload position data to GPU, declare its size and type

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[1])                  # make it active to receive
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.index, GL.GL_STATIC_DRAW) 

        GL.glEnableVertexAttribArray(1)      # assign to layout = 1 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[2])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, color, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)
        

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def draw(self, projection, view, model, color_shader, **param):
        GL.glUseProgram(color_shader.glid)
        
        P = GL.glGetUniformLocation(color_shader.glid, 'P')
        GL.glUniformMatrix4fv(P, 1, True, projection)
        
        V = GL.glGetUniformLocation(color_shader.glid, 'V')
        GL.glUniformMatrix4fv(V, 1, True, view)
        
        M = GL.glGetUniformLocation(color_shader.glid, 'M')
        GL.glUniformMatrix4fv(M, 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        
        GL.glBindVertexArray(self.glid)   # activate our vertex array
        GL.glDrawElements(GL.GL_TRIANGLES, self.index.size, GL.GL_UNSIGNED_INT, None)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)

# ------------  Scene object classes ------------------------------------------
class SimpleTriangle:
    """Hello triangle object"""

    def __init__(self):

        # triangle position buffer
        global position
        position= np.array((
            (0, 1, 0),
            (0, -1, -1),
            (-1, -1, 1),  
            (0, 1, 0),
            (0, -1, -1), 
            (1, -1, 1), 
            (-1, -1, 1), 
            (0, -1, -1), 
            (1, -1, 1)
        ), np.float32)
        color = np.array(((0, 0, 0), (0, 1, 0), (0, 0, 1)), 'f')

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below
        self.buffers = [GL.glGenBuffers(1), GL.glGenBuffers(1)]  # create buffer for position attrib

        # bind the vbo, upload position data to GPU, declare its size and type
        GL.glEnableVertexAttribArray(0)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, position, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

        GL.glEnableVertexAttribArray(1)      # assign to layout = 1 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[1])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, color, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)
        

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def draw(self, projection, view, model, color_shader, **param):
        GL.glUseProgram(color_shader.glid)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)
        # GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, position.shape[0])             # draw 9 vertices = 3 triangles
        GL.glBindVertexArray(0)

        P = GL.glGetUniformLocation(color_shader.glid, 'P')
        GL.glUniformMatrix4fv(P, 1, True, projection)
        
        V = GL.glGetUniformLocation(color_shader.glid, 'V')
        GL.glUniformMatrix4fv(V, 1, True, view)

        M = GL.glGetUniformLocation(color_shader.glid, 'M')
        GL.glUniformMatrix4fv(M, 1, True, model)


        # my_color_location    = GL.glGetUniformLocation(color_shader.glid, 'color')
        # GL.glUniform3fv(my_color_location, 1, (1, 1, 1))

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
        self.trackball = GLFWTrackball(self.win)
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)


        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)

        # initially empty list of object to draw
        self.drawables = []


    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)
            

            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, identity(), color_shader = self.color_shader)

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
            # elif key == glfw.KEY_C:
            #     red = GL.glGetUniformLocation(self.color_shader.glid, 'red')
            #     green = GL.glGetUniformLocation(self.color_shader.glid, 'green')
            #     blue = GL.glGetUniformLocation(self.color_shader.glid, 'blue')
            #     GL.glUniform1f(red, random())
            #     GL.glUniform1f(green, random())
            #     GL.glUniform1f(blue, random())
            elif key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))



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

# -------------- main program and scene setup --------------------------------
def main():
 
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    sphere = Sphere()             # re-use same cylinder instance

    """
        Cloud
    """
    sphere_level3_1 = Node(transform=translate(100, -50, 0) @ rotate(axis=vec(1, 0, 0), angle=0, radians=None) @ scale(.5, .5, .5), children=[sphere])
    sphere_level3_2 = Node(transform=translate(100, 50, 0) @ rotate(axis=vec(1, 0, 0), angle=0, radians=None) @ scale(.4, .6, 1), children=[sphere])
    sphere_level3_3 = Node(transform=translate(50, 100, 0) @ rotate(axis=vec(1, 0, 0), angle=0, radians=None) @ scale(.21, .25, .3), children=[sphere])
    
    sphere_level2_1 = Node(transform=translate(100, 0, 0) @ rotate(axis=vec(1, 0, 0), angle=0, radians=None) @ scale(.7, .7, .7), children=[sphere])
    sphere_level2_1.add(sphere_level3_1)
    sphere_level2_1.add(sphere_level3_2)
    sphere_level2_1.add(sphere_level3_3)

    sphere_level2_2 = Node(transform=translate(0, 100, 0) @ rotate(axis=vec(0, 1, 0), angle=90, radians=None) @ scale(.7, .7, .7), children=[sphere])    
    sphere_level2_2.add(sphere_level3_1)
    sphere_level2_2.add(sphere_level3_2)
    sphere_level2_2.add(sphere_level3_3)

    sphere_level2_3 = Node(transform=translate(0, -10, 100) @ rotate(axis=vec(0, 0, 1), angle=90, radians=None) @ scale(.7, .7, .7), children=[sphere])    
    sphere_level2_3.add(sphere_level3_1)
    sphere_level2_3.add(sphere_level3_2)
    sphere_level2_3.add(sphere_level3_3)

    sphere_level2_4 = Node(transform=translate(0, -10, -100) @ rotate(axis=vec(0, 1, 1), angle=90, radians=None) @ scale(.7, .7, .7), children=[sphere])    
    sphere_level2_4.add(sphere_level3_1)
    sphere_level2_4.add(sphere_level3_2)
    sphere_level2_4.add(sphere_level3_3)



    sphere_level1_1 = Node(transform=translate(1, 0, 0) @ rotate(axis=vec(1, 0, 0), angle=90, radians=None) @ scale(.005, .005, .005), children=[sphere])
    sphere_level1_1.add(sphere_level2_1)
    sphere_level1_1.add(sphere_level2_2)
    sphere_level1_1.add(sphere_level2_3)
    sphere_level1_1.add(sphere_level2_4)
    
    sphere_level1_2 = Node(transform=translate(-.3, 0, 0) @ rotate(axis=vec(0, 1, 0), angle=90, radians=None) @ scale(.005, .005, .005), children=[sphere])
    sphere_level1_2.add(sphere_level2_1)
    sphere_level1_2.add(sphere_level2_2)
    sphere_level1_2.add(sphere_level2_3)
    sphere_level1_2.add(sphere_level2_4)

    sphere_level1_3 = Node(transform=translate(-1, 0, 0) @ rotate(axis=vec(0, 1, 1), angle=180, radians=None) @ scale(.005, .005, .005), children=[sphere])
    sphere_level1_3.add(sphere_level2_1)
    sphere_level1_3.add(sphere_level2_2)
    sphere_level1_3.add(sphere_level2_3)
    sphere_level1_3.add(sphere_level2_4)

    translate_keys3 = {0 : vec(0, .65, 0), 50 : vec(3, 0, 0)}
    translate_keys1 = {0 : vec(-1, .6, 0), 50 : vec(3, 0, 0)}
    translate_keys2 = {0 : vec(-2, .7, 0), 50 : vec(4, 0, 0)}

    rotate_keys1 = {0: quaternion(1, 0, 0), 50: quaternion(0,1,1)}
    rotate_keys2 = {0: quaternion(1, 0, 0), 50: quaternion(1,.5,0)}
    rotate_keys3 = {0: quaternion(0, 0, 0), 50: quaternion(0,1,0)}
    scale_keys1 = {0: .25}
    scale_keys2 = {0: .3}
    scale_keys3 = {0: .2}

    keynode1 = KeyFrameControlNode(translate_keys1, rotate_keys3, scale_keys1)
    keynode2 = KeyFrameControlNode(translate_keys1, rotate_keys2, scale_keys3)
    keynode3 = KeyFrameControlNode(translate_keys2, rotate_keys3, scale_keys2)

    keynode4 = KeyFrameControlNode(translate_keys2, rotate_keys2, scale_keys1)
    keynode5 = KeyFrameControlNode(translate_keys2, rotate_keys2, scale_keys2)
    keynode6 = KeyFrameControlNode(translate_keys3, rotate_keys3, scale_keys1)

    keynode7 = KeyFrameControlNode(translate_keys3, rotate_keys1, scale_keys1)
    keynode8 = KeyFrameControlNode(translate_keys1, rotate_keys1, scale_keys3)
    keynode9 = KeyFrameControlNode(translate_keys3, rotate_keys2, scale_keys1)

    keynode1.add(sphere_level1_1)
    keynode2.add(sphere_level1_2)
    keynode3.add(sphere_level1_3)

    keynode4.add(sphere_level1_1)
    keynode5.add(sphere_level1_2)
    keynode6.add(sphere_level1_3)

    keynode7.add(sphere_level1_1)
    keynode8.add(sphere_level1_2)
    keynode9.add(sphere_level1_3)

    viewer.add(keynode1, keynode2, keynode3, keynode4, keynode5, keynode6, keynode7, keynode8, keynode9)


    """
        Sun
    """
    dim = 1.8
    sun_translate_keys = {2*i : -1 * dim * vec(math.cos(math.radians(i*10)) ,  math.sin(math.radians(-i*10))+.6 ) for i in range(0,17)}
    sun_rotate_keys = {0: quaternion(), 2: quaternion(), 3: quaternion(), 4: quaternion()}
    sun_scale_keys = {0: .1}
    sun_keynode = KeyFrameControlNode(sun_translate_keys, sun_rotate_keys, sun_scale_keys)

    sun_keynode.add(Sun())
    viewer.add(sun_keynode)

    """
        Flying Dinosaur
    """
    flying_dinosaur_translate_keys = {
        0 : vec(-2, .4, 0), 7.5 : vec(-.15, -.2, .5), 8 : vec(0, -.21, .5), 8.5 : vec(.15, -.2, .5), 16 : vec(2, .4, 0)
    }
    flying_dinosaur_rotate_keys = {
        0: quaternion(.7,1,0), 7: quaternion(0,.6,0), 8: quaternion(-.1,.7,0) ,9: quaternion(-.2, .8,0), 16: quaternion(0.7,-.8,0)
    }
    flying_dinosaur_scale_keys = {0: .0005, 7.5: .002, 8: .002, 8.5: .002, 16: .0005}  
    flying_dinosaur_keynode = KeyFrameControlNode(flying_dinosaur_translate_keys, flying_dinosaur_rotate_keys, flying_dinosaur_scale_keys)

    flying_dinosaur_keynode.add(FlyingDinosaur())
    viewer.add(flying_dinosaur_keynode)

    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
