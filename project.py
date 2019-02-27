#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys
from bisect import bisect_left      # search sorted keyframe lists

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions
from transform import translate, rotate, scale, vec, normalized, vec
from transform import Trackball, identity
from transform import (lerp, quaternion_slerp, quaternion_matrix, quaternion,
                       quaternion_from_euler)
#from grid_texture import generate_perlin_grid, gen_sphere, geizer
from generate_assets import grid, sphere, geizer, fountain
from PIL import Image               # load images for textures
from math import cos, sin, pi, sqrt, radians
import lib_bk

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


TEXTURE_VERT_SKY = """#version 330 core
uniform mat4 modelviewprojection;
layout(location = 0) in vec3 position;
//out vec3 fragTexCoord;
out mediump vec3 fragTexCoord;
void main() {
    gl_Position = modelviewprojection * vec4(position, 1);
    //gl_Position = gl_Position.xyww;
    fragTexCoord = normalize(position.xyz);
    //fragTexCoord = position;
}"""

TEXTURE_FRAG_SKY = """#version 330 core
uniform samplerCube diffuseMap;
in mediump vec3 fragTexCoord;
//in vec3 fragTexCoord;
out vec4 outColor;
void main() {
    outColor = texture(diffuseMap, fragTexCoord);
}"""


SKY_BOX_VERTICES = [
    -1.0,  1.0, -1.0,
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





class CubeTexture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, files=[], wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.glid)
        # helper array stores texture format for every pixel size 1..4
        format = [GL.GL_LUMINANCE, GL.GL_LUMINANCE_ALPHA, GL.GL_RGB, GL.GL_RGBA]
        sky_faces = ['beach_ft.JPG', 'beach_bk.JPG', 'beach_up.JPG', 'beach_dn.JPG', 'beach_rt.JPG', 'beach_lf.JPG']
        try:
            for i,file in enumerate(sky_faces):
            # imports image as a numpy array in exactly right format
                tex = np.array(Image.open("sor_beach/"+file))
                format1 = format[0 if len(tex.shape) == 2 else (tex.shape[2] - 1)]
                GL.glTexImage2D(GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL.GL_RGBA, tex.shape[1],
                                tex.shape[0], 0, format1, GL.GL_UNSIGNED_BYTE, tex)

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

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)



class CubeTexturedPlane:

    def __init__(self, file):
        self.shader = Shader(TEXTURE_VERT_SKY, TEXTURE_FRAG_SKY)
        skybox = iter(SKY_BOX_VERTICES)
        sk_right_triangle = [(x,next(skybox), next(skybox)) for x in skybox]
        self.vertex_array = lib_bk.VertexArray([sk_right_triangle])

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
        view2 = view.copy()
        view2[0][3] = 0
        view2[1][3] = 0
        view2[2][3] = 0
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view2 @ model)

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP,self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.draw(projection, view, model, self.shader)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
        GL.glUseProgram(0)


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
            nb_primitives, size = data.shape
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


    def draw(self, primitive):
        """draw a vertex array, either as direct array or indexed array"""
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)
        GL.glBindVertexArray(0)


    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)


class VertexArrayInst:
    """helper class to create and self destroy vertex array objects."""
    def __init__(self, nb_instances, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            attribs should be list of arrays with dim(0) indexed by vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        self.nb_instances = nb_instances

        # load a buffer per initialized vertex attribute (=dictionary)
        for loc, data in enumerate(attributes):
            if data is None:
                continue

            # bind a new vbo, upload its data to GPU, declare its size and type
            self.buffers += [GL.glGenBuffers(1)]
            data = np.array(data, np.float32, copy=False)
            nb_primitives, size = data.shape
            GL.glEnableVertexAttribArray(loc)  # activates for current vao only
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArraysInstanced
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElementsInstanced
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self, primitive):
        """draw a vertex array, either as direct array or indexed array"""
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments, self.nb_instances)
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
        model = model @ self.transform
        for child in self.children:
            child.draw(projection, view, model, **param)


# -------------- Keyframing Utilities TP6 ------------------------------------
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
        if (time <= self.times[0]):
            return self.values[0]
        if (time > self.times[len(self.times) - 1]):
            return self.values[len(self.times) - 1]

        # 2. search for closest index entry in self.times, using bisect_left
        insert_index = bisect_left(self.times, time)

        # 3. using the retrieved index, interpolate between the two neighboring
        # values in self.values, using the stored self.interpolate function
        fraction = (time - self.times[insert_index - 1])/(self.times[insert_index] - self.times[insert_index - 1])

        return self.interpolate(self.values[insert_index - 1], self.values[insert_index], fraction)


class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translate_keys = KeyFrames(translate_keys, interpolation_function=lerp)
        self.scale_keys = KeyFrames(scale_keys, interpolation_function=lerp)
        self.rotate_keys = KeyFrames(rotate_keys, interpolation_function=quaternion_slerp)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        translate_matrix = translate(*self.translate_keys.value(time))
        scale_matrix = scale(self.scale_keys.value(time))
        rotate_matrix = quaternion_matrix(self.rotate_keys.value(time))
        return translate_matrix @ rotate_matrix @ scale_matrix

    def last_frame(self):
        tlast = self.translate_keys.times[-1]
        slast = self.scale_keys.times[-1]
        rlast = self.rotate_keys.times[-1]
        return max(tlast, slast, rlast)

# -------------- Linear Blend Skinning : TP7 ---------------------------------
MAX_VERTEX_BONES = 4
MAX_BONES = 128

# new shader for skinned meshes, fully compatible with previous color fragment
SKINNING_VERT = """#version 330 core
// ---- camera geometry
uniform mat4 projection, view;

// ---- skinning globals and attributes
const int MAX_VERTEX_BONES=%d, MAX_BONES=%d;
uniform mat4 boneMatrix[MAX_BONES];

// ---- vertex attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 tex;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec4 bone_ids;
layout(location = 4) in vec4 bone_weights;

uniform vec3 light;
out vec3 nout;
out vec3 l;
uniform vec3 ks;
out vec3 ks_out;

// ----- interpolated attribute variables to be passed to fragment shader
//out vec3 fragColor;
out vec2 fragTexCoord;

void main() {

    // ------ creation of the skinning deformation matrix

    mat4 skinMatrix = mat4(0.);
    int i;
    float s = 0.0;
    for(i=0; i < MAX_VERTEX_BONES; i++){
        s = s + bone_weights[i];
    }
    for(i=0; i < MAX_VERTEX_BONES; i++){
        skinMatrix = skinMatrix + bone_weights[i]*boneMatrix[int(bone_ids[i])];
    }

    // ------ compute world and normalized eye coordinates of our vertex
    vec4 wPosition4 = (1./s) * skinMatrix * vec4(position, 1.0);
    gl_Position = projection * view * wPosition4;

    l = normalize(light);
    nout = normal;
    ks_out = ks;
    fragTexCoord = tex;

    //fragColor = color;
}
""" % (MAX_VERTEX_BONES, MAX_BONES)

class SkinnedMesh:
    """class of skinned mesh nodes in scene graph """
    def __init__(self, texture, attributes, bone_nodes, bone_offsets, index=None):

        # setup shader attributes for linear blend skinning shader
        self.vertex_array = VertexArray(attributes, index)

        self.shader = Shader(SKINNING_VERT, TEXTURE_FRAG)

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = bone_offsets

        self.texture = texture

    def draw(self, projection, view, _model, **_kwargs):
        """ skinning object draw method """

        shid = self.shader.glid
        GL.glUseProgram(shid)

        # setup camera geometry parameters
        loc = GL.glGetUniformLocation(shid, 'projection')
        GL.glUniformMatrix4fv(loc, 1, True, projection)
        loc = GL.glGetUniformLocation(shid, 'view')
        GL.glUniformMatrix4fv(loc, 1, True, view)

        # projection geometry
        ks_location = GL.glGetUniformLocation(self.shader.glid, 'ks')
        GL.glUniform3fv(ks_location, 1, (0.01,0.01,0.01))

        t = glfw.get_time()
        l_location = GL.glGetUniformLocation(self.shader.glid, 'light')
        GL.glUniform3fv(l_location, 1, (SUN_POSITION[0], SUN_POSITION[1], SUN_POSITION[2]))

        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)

        # bone world transform matrices need to be passed for skinning
        for bone_id, node in enumerate(self.bone_nodes):
            bone_matrix = node.world_transform @ self.bone_offsets[bone_id]

            bone_loc = GL.glGetUniformLocation(shid, 'boneMatrix[%d]' % bone_id)
            GL.glUniformMatrix4fv(bone_loc, 1, True, bone_matrix)

        # draw mesh vertex array
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave with clean OpenGL state, to make it easier to detect problems
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)

class SkinningControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, keys_dict, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = {}
        for act, keys in keys_dict.items():
            self.keyframes[act] = TransformKeyFrames(*keys) if keys[0] else None
        self.world_transform = identity()
        self.time = 0

        self.action = "Idle"

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        if self.action in self.keyframes.keys() and self.keyframes[self.action] is not None:  # no keyframe update should happens if no keyframes
            self.transform = self.keyframes[self.action].value(self.time)

        # store world transform for skinned meshes using this node as bone
        self.world_transform = model @ self.transform

        # default node behaviour (call children's draw method)
        super().draw(projection, view, model, **param)

    def set_time_all(self, value, add=False):
        if add:
            self.time += value
        else:
            self.time = value

        for c in self.children:
            if isinstance(c, SkinningControlNode):
                c.set_time_all(value, add)

    def set_action_all(self, value):
        self.action = value
        for c in self.children:
            if isinstance(c, SkinningControlNode):
                c.set_action_all(value)

    def set_tmax(self):

        def get_tmax(x, act):
            if not isinstance(x, SkinningControlNode):
                return 0
            if act in x.tmax:
                tself = x.tmax[act]
            else:
                tself = None
            if tself is None:
                tself = 0
            if len(x.children) > 0:
                tchildren = max([get_tmax(c, act) for c in x.children])
            else:
                tchildren = 0
            return max(tself, tchildren)

        for act in self.keyframes.keys():
            self.tmax[act] = max(get_tmax(self, act), 1.0)


class Controlled():
    """ The class representing the unique controlled object """
    def __init__(self, node):
        self.node = node
        self.translation = np.zeros(shape=(3), dtype=np.float)
        self.alpha = 0
        self.scaling = np.array([1.,1.,1.], dtype=np.float)
        self.in_action = False
        self.last_frame = 0
        self.move = "Walk"

    def draw(self, projection, view, model, **param):

        translate_matrix = translate(*self.translation[:3])
        scale_matrix = scale(self.scaling)
        rotate_matrix = rotate(axis=vec(0,1,0), angle=self.alpha)

        t = glfw.get_time()
        self.node.set_time_all(t - self.last_frame)
        if t - self.last_frame > self.node.tmax[self.node.action]:
            self.node.set_action_all("Idle")
            self.in_action = False
            self.node.set_time_all(0)
            self.last_frame = t
        model = translate_matrix @ rotate_matrix @ model
        self.node.draw(projection, view, model, **param)

#------------------------------------------------------------------
def load_skinned(files, actions=None):
    """load resources from file using pyassimp, return node hierarchy """

    if actions is None or len(actions)<len(files):
        print("Actions for animations were not defined or some are missing.")
        if actions == None:
            actions = ["Walk"]*len(files)
        else:
            for i in range(len(files) - len(actions)):
                actions.append("Walk")
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scenes = dict()
        for i,f in enumerate(files):
            print("Loading models : ", f)
            scenes[actions[i]] = pyassimp.load(f, option)

    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []

    # ----- load animations
    def conv(assimp_keys, ticks_per_second):
        """ Conversion from assimp key struct to our dict representation """
        return {key.time / ticks_per_second: key.value for key in assimp_keys}

    # load first animation in scene file (could be a loop over all animations)
    transform_keyframes_dict = {}

    for act, scene in scenes.items():
        transform_keyframes = {}
        if scene.animations:
            anim = scene.animations[0]
            for channel in anim.channels:
                # for each animation bone, store trs dict with {times: transforms}
                # (pyassimp name storage bug, bytes instead of str => convert it)
                transform_keyframes[channel.nodename.data.decode('utf-8')] = (
                    conv(channel.positionkeys, anim.tickspersecond),
                    conv(channel.rotationkeys, anim.tickspersecond),
                    conv(channel.scalingkeys, anim.tickspersecond)
                )
        transform_keyframes_dict[act] = transform_keyframes

    nodes = {}  # nodes: string name -> node dictionary

    def make_nodes(pyassimp_node):
        """ Recursively builds nodes for our graph, matching pyassimp nodes """

        trs_keyframes_dict = {}
        tmax = {}
        for act, trs in transform_keyframes_dict.items():
            trs_keyframes_dict[act] = trs.get(pyassimp_node.name, (None,))
            if len(trs_keyframes_dict[act]) > 0 and trs_keyframes_dict[act][0] is not None:
                tmax[act] = max([max(trs_keyframes_dict[act][i].keys()) for i in range(len(trs_keyframes_dict[act]))])

        node = SkinningControlNode(trs_keyframes_dict, name=pyassimp_node.name,
                                   transform=pyassimp_node.transformation)
        node.tmax = tmax
        nodes[pyassimp_node.name] = node, pyassimp_node
        node.add(*(make_nodes(child) for child in pyassimp_node.children))
        return node

    root_node = make_nodes(scenes["Walk"].rootnode)

    path = os.path.dirname(files[0])
    for mat in scenes["Walk"].materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))
        if 'file' in mat.tokens:  # texture file token
            print("file in math tokens")
            tname = mat.tokens['file'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            tname = [os.path.join(d[0], f) for d in os.walk(path) for f in d[2]
                     if tname.startswith(f) or f.startswith(tname)]
            if tname:
                print("tname = %s found" % tname[0])
                mat.texture = Texture(tname[0])
            else:
                print('Failed to find texture:', tname)


    # ---- create SkinnedMesh objects
    for mesh in scenes["Walk"].meshes:
        # -- skinned mesh: weights given per bone => convert per vertex for GPU
        # first, populate an array with MAX_BONES entries per vertex
        v_bone = np.array([[(0, 0)]*MAX_BONES] * mesh.vertices.shape[0],
                          dtype=[('weight', 'f4'), ('id', 'u4')])
        for bone_id, bone in enumerate(mesh.bones[:MAX_BONES]):
            for entry in bone.weights:  # weight,id pairs necessary for sorting
                v_bone[entry.vertexid][bone_id] = (entry.weight, bone_id)

        v_bone.sort(order='weight')             # sort rows, high weights last
        v_bone = v_bone[:, -MAX_VERTEX_BONES:]  # limit bone size, keep highest

        # prepare bone lookup array & offset matrix, indexed by bone index (id)
        bone_nodes = [nodes[bone.name][0] for bone in mesh.bones]
        bone_offsets = [bone.offsetmatrix for bone in mesh.bones]


        if mesh.materialindex == 2:
            texture = scenes["Walk"].materials[2].texture
        if mesh.materialindex == 1:
            texture = scenes["Walk"].materials[1].texture

        tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                  if mesh.texturecoords.size else None)

        # initialize skinned mesh and store in pyassimp_mesh for node addition
        mesh.skinned_mesh = SkinnedMesh(texture,
                [mesh.vertices, tex_uv, mesh.normals, v_bone['id'], v_bone['weight']],
                bone_nodes, bone_offsets, mesh.faces
        )

    # ------ add each mesh to its intended nodes as indicated by assimp
    for final_node, assimp_node in nodes.values():
        final_node.add(*(_mesh.skinned_mesh for _mesh in assimp_node.meshes))

    nb_triangles = sum((mesh.faces.shape[0] for mesh in scenes["Walk"].meshes))
    print('Loaded', files[0], '\t(%d meshes, %d faces, %d nodes, %d animations)' %
          (len(scenes["Walk"].meshes), nb_triangles, len(nodes), len(scenes["Walk"].animations)))

    for scene in scenes.values():
        pyassimp.release(scene)
    return [root_node]


COLOR_VERT_SUN = """#version 330 core
layout(location = 0) in vec3 position;
out vec3 colors;
uniform mat4 P;
uniform mat4 V;
uniform mat4 M;
uniform vec3 color;
void main() {
    gl_Position = P * V * M * vec4(position, 1);
    colors = color;
    //colors = vec3(1,1,1);

}"""

COLOR_FRAG_SUN = """#version 330 core
out vec4 outColor;
in vec3 colors;
void main() {
    outColor = vec4(colors, 1);
}"""


class SunMesh:

    def __init__(self, attributes, color, index=None):

        self.vertex_array = VertexArray(attributes, index)
        self.color_shader = Shader(COLOR_VERT_SUN, COLOR_FRAG_SUN)
        self.color = color

    def draw(self, projection, view, model, color_shader,**param):

        names = ['P', 'V', 'M', 'color']
        loc = {n: GL.glGetUniformLocation(self.color_shader.glid, n) for n in names}
        GL.glUseProgram(self.color_shader.glid)

        GL.glUniformMatrix4fv(loc['P'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['V'], 1, True, view)
        GL.glUniformMatrix4fv(loc['M'], 1, True, model)
        GL.glUniform3fv(loc['color'], 1, self.color)

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


# -------------- Example texture plane class ----------------------------------
TEXTURE_VERT = """#version 330 core


uniform mat4 modelviewprojection;
uniform vec3 light;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex;

out vec3 nout;
out vec3 l;
uniform vec3 ks;
out vec3 ks_out;

out vec2 fragTexCoord;
void main() {
    gl_Position = modelviewprojection * vec4(position, 1);
    l = normalize(light);
    nout = normal;
    ks_out = ks;
    fragTexCoord = tex;
}"""

NB_TEXT_INSTANCES = 1000

TEXTURE_VERT_INST = """#version 330 core


uniform mat4 modelviewprojection;
uniform vec3 light;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex;

uniform vec3 offsets[%d];

out vec3 nout;
out vec3 l;
uniform vec3 ks;
out vec3 ks_out;

out vec2 fragTexCoord;
void main() {
    vec3 offset = offsets[gl_InstanceID];
    gl_Position = modelviewprojection * vec4(position + offset, 1);
    l = normalize(light);
    nout = normal;
    ks_out = ks;
    fragTexCoord = tex;
}""" % (NB_TEXT_INSTANCES)

TEXTURE_FRAG = """#version 330 core
in vec3 nout;
in vec3 l;
in vec3 ks_out;

float shine;
out vec4 outColor;

uniform sampler2D diffuseMap;
in vec2 fragTexCoord;
void main() {
    float scalar_prod = dot(nout,l);
    if (scalar_prod < 0) {
        scalar_prod = 0;
    }
    shine = 1.0 * 128.0;
    vec3 ref = reflect(normalize(l), normalize(nout));
    float dot_ref = dot(ref, vec3(0,0,1));
    if (dot_ref< 0) {
        dot_ref = 0;
    }
    float r = scalar_prod*texture(diffuseMap, fragTexCoord)[0] + pow(dot_ref, shine)*ks_out[0];
    float g = scalar_prod*texture(diffuseMap, fragTexCoord)[1] + pow(dot_ref, shine)*ks_out[1];
    float b = scalar_prod*texture(diffuseMap, fragTexCoord)[2] + pow(dot_ref, shine)*ks_out[2];
    outColor = vec4(r, g, b, 1);
}"""

class TexturedMesh:
    """ Simple first textured object """

    def __init__(self, texture, attributes, index):
        self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        # triangle and face buffers
        vertices = attributes[0]
        self.tex_uv = attributes[1]
        self.normals = (attributes[2])
        self.vertex_array = VertexArray([vertices, self.normals, self.tex_uv], index)

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.file = texture
        self.transform = identity()

        # setup texture and upload it to GPU
        self.texture = texture

    def draw(self, projection, view, model, win=None, **_kwargs):

        # some interactive elements
        if win!=None:
            if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
                self.wrap_mode = next(self.wrap)
                self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

            if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
                self.filter_mode = next(self.filter)
                self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)
        GL.glUseProgram(self.shader.glid)

        # projection geometry
        ks_location = GL.glGetUniformLocation(self.shader.glid, 'ks')
        GL.glUniform3fv(ks_location, 1, (0.01,0.01,0.01))

        t = glfw.get_time()
        l_location = GL.glGetUniformLocation(self.shader.glid, 'light')
        GL.glUniform3fv(l_location, 1, (-SUN_POSITION[0], -SUN_POSITION[1], -SUN_POSITION[2]))

        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        tex_loc = GL.glGetUniformLocation(self.shader.glid, 'tex_uv')
        model = model @ self.transform
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)
        GL.glUniform2fv(tex_loc, 0, self.tex_uv)
        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)



class TexturedMeshInst:
    """ Simple first textured object """

    def __init__(self, texture, nb_instances, attributes, index):
        self.shader = Shader(TEXTURE_VERT_INST, TEXTURE_FRAG)

        # triangle and face buffers
        vertices = attributes[0]
        self.tex_uv = attributes[1]
        self.normals = (attributes[2])
        self.vertex_array = VertexArrayInst(nb_instances, [vertices, self.normals, self.tex_uv], index)

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.file = texture
        self.transform = identity()

        # setup texture and upload it to GPU
        self.texture = texture
        self.positions = None

    def draw(self, projection, view, model, win=None, **_kwargs):

        # some interactive elements
        if win!=None:
            if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
                self.wrap_mode = next(self.wrap)
                self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

            if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
                self.filter_mode = next(self.filter)
                self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)
        GL.glUseProgram(self.shader.glid)

        # offset the instance
        for i, pos in enumerate(self.positions):
            loc_offset = GL.glGetUniformLocation(self.shader.glid, 'offsets[%d]' % i)
            GL.glUniform3fv(loc_offset, 1, pos)

        # projection geometry
        ks_location = GL.glGetUniformLocation(self.shader.glid, 'ks')
        GL.glUniform3fv(ks_location, 1, (0.01,0.01,0.01))

        t = glfw.get_time()
        l_location = GL.glGetUniformLocation(self.shader.glid, 'light')
        GL.glUniform3fv(l_location, 1, (-SUN_POSITION[0], -SUN_POSITION[1], -SUN_POSITION[2]))

        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        tex_loc = GL.glGetUniformLocation(self.shader.glid, 'tex_uv')
        model = model @ self.transform
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)
        GL.glUniform2fv(tex_loc, 0, self.tex_uv)
        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)

#-----------------------------------------------------------------------------
def get_elevation(ground, size, x, y):
    xl = int(x)
    yl = int(y)
    xu = int(x) + 1
    yu = int(y) + 1
    try:
        if xu - x > y - yl:
            a1 = ground[xl*size + yl][1]
            a2 = ground[xu*size + yl][1]
            a3 = ground[xl*size + yu][1]
            w2 = x - xl
            w3 = yu - y
            w1 = 1 - w2 - w3
        else:
            a1 = ground[xu*size + yu][1]
            a2 = ground[xl*size + yu][1]
            a3 = ground[xu*size + yl][1]
            w2 = xu - x
            w3 = y - yl
            w1 = 1 - w2 - w3
        return w1*a1 + w2*a2 + w3*a3
    except IndexError:
        return 0
#-----------------------------------------------------------------------------

def load_textured_n_instances(file, nb_instances):
    """ load resources using pyassimp, return list of TexturedMeshes """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file)
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))
        if 'file' in mat.tokens:  # texture file token
            tname = mat.tokens['file'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            tname = [os.path.join(d[0], f) for d in os.walk(path) for f in d[2]
                     if tname.startswith(f) or f.startswith(tname)]
            if tname:
                mat.texture = Texture(tname[0])
            else:
                print('Failed to find texture:', tname)

    # prepare textured mesh
    meshes = []
    for mesh in scene.meshes:
        texture = scene.materials[mesh.materialindex].texture
        # tex coords in raster order: compute 1 - y to follow OpenGL convention
        tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                  if mesh.texturecoords.size else None)
        # create the textured mesh object from texture, attributes, and indices
        meshes.append(TexturedMeshInst(texture, nb_instances, [mesh.vertices, tex_uv, mesh.normals], mesh.faces))

        size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
        print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))



    pyassimp.release(scene)
    return meshes

#-----------------------------------------------------------------------------

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

SUN_POSITION=[0]*3

class KeyFrameControlNode2(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, translate_keys, rotate_keys, scale_keys,isSun=False ,**kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)
        self.isSun = isSun
        if isSun:
            self.time = glfw.get_time()
        else:
            self.time = 0
    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """

        SUN_POSITION[1] = self.transform[0][3]
        SUN_POSITION[1] = self.transform[1][3]
        SUN_POSITION[2] = self.transform[2][3]
        if not self.isSun:
            self.time  = (self.time+0.05) %50
            self.transform = self.keyframes.value(self.time)
        else:
            self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model, **param)

#-----------------------------------------------------------------------------

class Sphere(Node):
    def __init__(self):
        Node.__init__(self)
        vertices, normals, faces = sphere(5)
        meshes = SunMesh([100*vertices, normals], (1,1,1), faces)
        self.add(meshes)

class Geizer(Node):
    def __init__(self):
        Node.__init__(self)
        vertices, normals, faces = geizer()
        meshes = SunMesh([vertices, normals], (0.6,0.53,0.16), faces)
        self.add(meshes)

class Sun(Node):
    def __init__(self):
        Node.__init__(self)
        self.add(*load('sun.obj', (1, 1, 0)))

class FlyingDinosaur(Node):
    def __init__(self):
        Node.__init__(self)
        self.add(*load('Pterodactyl/Pterodactyl.obj', (.4, .4, .4)))


def load(file, color=None):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []     # error reading => return empty list

    if color == None:
        color =(1,1,1)

    meshes = [SunMesh([m.vertices, m.normals],color, m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))
    pyassimp.release(scene)
    return meshes

#------------------------- fountain like effect -------------------------------
PARTICLES = 200

TEXTURE_VERT1 = """#version 330 core
uniform mat4 modelviewprojection;
layout(location = 0) in vec3 position;
uniform vec3 offsets[%d];
uniform mat4 scale;

out vec2 fragTexCoord;
void main() {
   vec3 offset = offsets[gl_InstanceID];
   gl_Position = modelviewprojection *scale * (vec4(position ,1) + vec4(offset,1)) ;
   fragTexCoord = position.xy;
}""" %(PARTICLES)

TEXTURE_FRAG1 = """#version 330 core
uniform sampler2D diffuseMap;
in vec2 fragTexCoord;
out vec4 outColor;
void main() {
   outColor = texture(diffuseMap, fragTexCoord);
}"""


class TexturedPlane1:
    """Instanced particle"""

    def __init__(self, file):
        self.shader = Shader(TEXTURE_VERT1, TEXTURE_FRAG1)

        # triangle and face buffers
        vertices = np.array(((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)), np.float32)
        faces = np.array(((0, 1, 3), (3, 1, 2)), np.uint32)
        self.vertex_array = VertexArrayPlaneInst([vertices], PARTICLES , faces)

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.file = file

        # setup texture and upload it to GPU
        self.texture = Texture(file, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, win=None, **_kwargs):

        # some interactive elements
        if win!=None:
            if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
                self.wrap_mode = next(self.wrap)
                self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

            if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
                self.filter_mode = next(self.filter)
                self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)
        GL.glUseProgram(self.shader.glid)
        t = glfw.get_time()

        #offsets for instances
        translations = fountain(t, PARTICLES)

        for i in range(PARTICLES):
            lco = GL.glGetUniformLocation(self.shader.glid, 'offsets[%d]' % i)
            GL.glUniform3fv(lco, 1, translations[i])
        l = GL.glGetUniformLocation(self.shader.glid, 'scale')
        GL.glUniformMatrix4fv(l, 1, True, scale(0.1))


        for i in range(10):
            lco = GL.glGetUniformLocation(self.shader.glid, 'offsets[%d]' % i)
            GL.glUniform3fv(lco, 1, translations[i])
        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)



class VertexArrayPlaneInst:
    """helper class to create and self destroy vertex array objects."""
    def __init__(self, attributes,number, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            attribs should be list of arrays with dim(0) indexed by vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0
        self.instances = number
        # load a buffer per initialized vertex attribute (=dictionary)
        for loc, data in enumerate(attributes):
            if data is None:
                continue

            # bind a new vbo, upload its data to GPU, declare its size and type
            self.buffers += [GL.glGenBuffers(1)]
            data = np.array(data, np.float32, copy=False)
            nb_primitives, size = data.shape
            GL.glEnableVertexAttribArray(loc)  # activates for current vao only
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArraysInstanced
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElementsInstanced
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self, primitive):
        """draw a vertex array, either as direct array or indexed array"""
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments, self.instances)
        GL.glBindVertexArray(0)


    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)



# ------------  Viewer class & window management ------------------------------
class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):
        self.mov_speed = 0.5

        self.camera = np.array([0,-5,-10], dtype=np.float32) # default camera position relative to the focus posisiton
        self.campos = np.array([0,-5,-10], dtype=np.float32) # current camera position relative to the focus posisiton
        self.dist = sqrt(self.camera[0]**2 + self.camera[2]**2)

        # unused
        self.dir = np.array([0,0,0], dtype=np.float32)
        self.rotation = identity() # orientation of the view matrix
        self.alpha = 0

        self.trackball_view = False

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
        GL.glEnable(GL.GL_DEPTH_TEST)
        #GL.glEnable(GL.GL_CULL_FACE)
        GL.glFrontFace ( GL.GL_CW )
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        GL. glEnable(GL.GL_BLEND);

        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
        # compile and initialize shader programs once globally

        # initially empty list of object to draw
        self.drawables = []
        self.trackball = GLFWTrackball(self.win)
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])


    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            winsize = glfw.get_window_size(self.win)

            camera_position = (self.pos + self.campos)

            if self.trackball_view:
                view = self.trackball.view_matrix()
                view = translate(*camera_position) @ view
            else:
                self.pos = -1*self.controlled.translation
                for x in range(20):
                    zt = get_elevation(self.ground, self.ground_size, camera_position[0], camera_position[2])
                    if zt > -camera_position[1]:
                        # we're under the surface - that's not what we want
                        self.campos[0] *= (27-x)/27.0
                        self.campos[2] *= (27-x)/27.0
                        camera_position = self.pos + self.campos
                    else:
                        break
                view = rotate(axis=vec(0,1,0), angle=self.alpha) @ translate(*camera_position)
            projection = self.trackball.projection_matrix(winsize)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, identity(), win=self.win,
                              color_shader=self.color_shader)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables, controlled=False, spawn=(0,0,0)):
        """ add objects to draw in this window """
        if controlled:
            self.controlled = Controlled(drawables[0])
            self.controlled.translation += spawn
            self.controlled.node.set_tmax()
            self.pos = -1*self.controlled.translation
            self.drawables.extend([self.controlled])
        else:
            self.drawables.extend(drawables)


    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:

            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_P:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key == glfw.KEY_SPACE:
                glfw.set_time(0)

            if key == glfw.KEY_I:
                self.pos += np.array([0,0,self.mov_speed], dtype=np.float32)
            if key == glfw.KEY_K:
                self.pos -= np.array([0,0,self.mov_speed], dtype=np.float32)
            if key == glfw.KEY_L:
                self.pos -= np.array([self.mov_speed,0,0], dtype=np.float32)
            if key == glfw.KEY_J:
                self.pos += np.array([self.mov_speed,0,0], dtype=np.float32)
            if key == glfw.KEY_N:
                self.pos += np.array([0,self.mov_speed,0], dtype=np.float32)
            if key == glfw.KEY_U:
                self.pos -= np.array([0,self.mov_speed,0], dtype=np.float32)
            if key == glfw.KEY_X:
                if self.controlled.move == "Run":
                    self.controlled.move = "Walk"
                elif self.controlled.move == "Walk":
                    self.controlled.move = "Run"
            if key == glfw.KEY_V:
                self.trackball_view = not self.trackball_view
            if key == glfw.KEY_R:
                if not self.controlled.in_action:
                    d = self.controlled.node
                    d.set_action_all("Roar")
                    self.controlled.in_action = True
                    self.controlled.last_frame = glfw.get_time()
            if key == glfw.KEY_F:
                if not self.controlled.in_action:
                    d = self.controlled.node
                    d.set_action_all("Eat")
                    self.controlled.in_action = True
                    self.controlled.last_frame = glfw.get_time()
            if key == glfw.KEY_ENTER:
                if not self.controlled.in_action:
                    d = self.controlled.node
                    d.set_action_all("Attack")
                    self.controlled.in_action = True
                    self.controlled.last_frame = glfw.get_time()

            if key == glfw.KEY_W:
                if not self.controlled.in_action:
                    d = self.controlled.node
                    d.set_action_all(self.controlled.move)
                    self.controlled.in_action = True
                    self.controlled.last_frame = glfw.get_time()
                elif self.controlled.node.action == "Run" or self.controlled.node.action == "Walk":
                    if self.controlled.move == "Walk":
                        tr = (0,0,-0.1,1)
                    else:
                        tr = (0,0,-0.2,1)
                    trd = (rotate(axis=vec(0,1,0), angle=self.controlled.alpha) @ tr)[:3]

                    cur_pos = self.controlled.translation
                    new_pos = cur_pos + trd
                    newz = get_elevation(self.ground, self.ground_size, new_pos[0], new_pos[2])
                    trd[1] = newz - cur_pos[1]
                    self.controlled.translation += trd
                    self.pos -= trd

            if key == glfw.KEY_A:
                d = self.controlled.node
                pos = self.controlled.translation
                self.controlled.alpha = (self.controlled.alpha + 1) % 360

                alpha = self.alpha * pi/180.0
                self.alpha -= 1
                alpha = self.alpha * pi/180.0
                self.campos = self.camera + (self.dist * sin(alpha), 0, self.dist * (1-cos(alpha)) )

            if key == glfw.KEY_D:
                d = self.controlled.node
                pos = self.controlled.translation
                self.controlled.alpha = (self.controlled.alpha - 1) % 360
                self.alpha += 1
                alpha = self.alpha * pi/180.0
                self.campos = self.camera + (self.dist * sin(alpha), 0, self.dist * (1-cos(alpha)) )

            if key == glfw.KEY_E:
                self.alpha -= 1
                alpha = self.alpha * pi/180.0
                self.campos = self.camera + (self.dist * sin(alpha), 0, self.dist * (1-cos(alpha)) )
            if key == glfw.KEY_Q:
                self.alpha += 1
                alpha = self.alpha * pi/180.0
                self.campos = self.camera + (self.dist * sin(alpha), 0, self.dist * (1-cos(alpha)) )

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()
    files = os.listdir("sor_beach")

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

    translate_keys3 = {0 : vec(50, 20, -6), 50 : vec(110, 20, -1)}
    translate_keys1 = {0 : vec(60, 30, -6), 50 : vec(140, 30, -1)}
    translate_keys2 = {0 : vec(50, 20, -6), 50 : vec(130, 20, -1)}

    rotate_keys1 = {0: quaternion(1, 0, 0), 50: quaternion(0,1,1)}
    rotate_keys2 = {0: quaternion(1, 0, 0), 50: quaternion(1,.5,0)}
    rotate_keys3 = {0: quaternion(0, 0, 0), 50: quaternion(0,1,0)}
    scale_keys1 = {0: 4}
    scale_keys2 = {0: 5}
    scale_keys3 = {0: 6}

    keynode1 = KeyFrameControlNode2(translate_keys1, rotate_keys3, scale_keys1)
    keynode2 = KeyFrameControlNode2(translate_keys1, rotate_keys2, scale_keys3)
    keynode3 = KeyFrameControlNode2(translate_keys2, rotate_keys3, scale_keys2)

    keynode4 = KeyFrameControlNode2(translate_keys2, rotate_keys2, scale_keys1)
    keynode5 = KeyFrameControlNode2(translate_keys2, rotate_keys2, scale_keys2)
    keynode6 = KeyFrameControlNode2(translate_keys3, rotate_keys3, scale_keys1)

    keynode7 = KeyFrameControlNode2(translate_keys3, rotate_keys1, scale_keys1)
    keynode8 = KeyFrameControlNode2(translate_keys1, rotate_keys1, scale_keys3)
    keynode9 = KeyFrameControlNode2(translate_keys3, rotate_keys2, scale_keys1)

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
    dim = 200
    sun_translate_keys = {4*i : -1 * dim * vec(cos(radians(i*10)) - .3,  .8 * sin(radians(-i*10))+.55, .4 ) for i in range(0,17)}
    sun_rotate_keys = {0: quaternion(), 2: quaternion(), 3: quaternion(), 4: quaternion()}
    sun_scale_keys = {0: 5}
    sun_keynode = KeyFrameControlNode2(sun_translate_keys, sun_rotate_keys, sun_scale_keys)

    sun_keynode.add(Sun())
    viewer.add(sun_keynode)

    """
        Flying Dinosaur
    """
    flying_dinosaur_translate_keys = {
        0 : vec(70, 25, 20), 7.5 : vec(90, 10, 31), 8 : vec(95, 9, 31), 8.5 : vec(100, 10, 31), 16 : vec(120, 30, 20)
    }
    flying_dinosaur_rotate_keys = {
        0: quaternion(.7,1,0), 7: quaternion(0,.6,0), 8: quaternion(-.1,.7,0) ,9: quaternion(-.2, .8,0), 16: quaternion(0.7,-.8,0)
    }
    flying_dinosaur_scale_keys = {0: .005, 7.5: .02, 8: .02, 8.5: .02, 16: .005}
    flying_dinosaur_keynode = KeyFrameControlNode2(flying_dinosaur_translate_keys, flying_dinosaur_rotate_keys, flying_dinosaur_scale_keys)

    flying_dinosaur_keynode.add(FlyingDinosaur())
    viewer.add(flying_dinosaur_keynode)

    # Sky and ground
    size, step = 200, 25
    global SKY_BOX_VERTICES

    SKY_BOX_VERTICES = list(map(lambda x:x*size, SKY_BOX_VERTICES))
    viewer.add(CubeTexturedPlane(files))

    new = False
    textured = False
    if len(sys.argv) >= 2:
        new = (sys.argv[1] == "-n")
    if len(sys.argv) >= 3:
        textured = (sys.argv[2] == "-t")

    vertices, normals, faces, tname = grid(size, step=step, new=new, textured=textured)
    for i,x in enumerate(normals):
        normals[i] = -1*x

    tex_uv = np.zeros(shape=((size+1)*(size+1), 2))
    for i in range(len(tex_uv)):
        tex_uv[i][0] = (i // (size + 1))/(size + 1)
        tex_uv[i][1] = (i % (size + 1))/(size + 1)

    viewer.ground = vertices
    viewer.ground_size = size + 1


    tree_bases = []
    for i, x in enumerate(normals):
        if abs(x[1]) > 0.96:
            p = i // (size + 1)
            q = i % (size + 1)
            tree_bases.append((p, vertices[i][1], q))
    number_trees = len(tree_bases)
    if number_trees > 20:
        tree_bases = [tree_bases[i] for i in range(1, number_trees, number_trees//50)]


    global NB_TEXT_INSTANCES
    NB_TEXT_INSTANCES = len(tree_bases)

    trees = load_textured_n_instances("Tree/Tree.obj", len(tree_bases))

    for t in trees:
        t.positions = tree_bases
        viewer.add(t)
    viewer.add(TexturedMesh(Texture(tname), [vertices, tex_uv, normals], faces))
    dino = load_skinned(["dino/Dinosaurus_idle.dae",
                         "dino/Dinosaurus_walk.dae",
                         "dino/Dinosaurus_run.dae",
                         "dino/Dinosaurus_attack.dae",
                         "dino/Dinosaurus_roar.dae",
                         "dino/Donosaurus_eat.dae"], actions=["Idle", "Walk", "Run", "Attack", "Roar", "Eat"])
    spawn_dino = vertices[int(size/2)*(size+1)+int(size/2)]
    viewer.add(*dino, controlled=True, spawn=spawn_dino)

    # Geizer
    geizer = Geizer()
    geizer_pos = spawn_dino.copy()
    geizer_pos += [0,0,-10]
    geizer_pos[1] = get_elevation(vertices, size + 1, geizer_pos[0], geizer_pos[2])
    geizer.transform = translate(*geizer_pos)
    viewer.add(geizer)
    fountain = Node(transform=translate(*geizer_pos), children=[TexturedPlane1("droplet3.png")])
    viewer.add(fountain)


    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
