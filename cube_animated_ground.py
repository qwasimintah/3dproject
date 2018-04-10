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
from transform import translate, rotate, scale, vec, identity, lerp
#from h_loader import *
from transform import Trackball, identity
#from grid_normals import generate_grid, generate_perlin_grid 
from grid_texture import generate_perlin_grid

from PIL import Image               # load images for textures
from transform import (lerp, quaternion_slerp, quaternion_matrix, quaternion,
                       quaternion_from_euler)
#import copy

import math
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
#

#class ColorMesh:
#
#    def __init__(self, attributes, index=None):
#        self.vertex_array = VertexArray(attributes, index)
#
#    def draw(self, projection, view, model, color_shader):
#
#        names = ['view', 'projection', 'model']
#        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
#        GL.glUseProgram(color_shader.glid)
#
#        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
#        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
#        GL.glUniformMatrix4fv(loc['model'], 1, True, model)
#
#        # draw triangle as GL_TRIANGLE vertex array, draw array call
#        self.vertex_array.draw(GL.GL_TRIANGLES)


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

    #def draw(self, primitive):
    #def draw(self, projection, view, model, color_shader):
    #    """draw a vertex array, either as direct array or indexed array"""
    #    GL.glBindVertexArray(self.glid)


    #    names = ['view', 'projection', 'model']
    #    loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
    #    GL.glUseProgram(color_shader.glid)

    #    GL.glUniformMatrix4fv(loc['view'], 1, True, view)
    #    GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
    #    GL.glUniformMatrix4fv(loc['model'], 1, True, model)

    #    # draw triangle as GL_TRIANGLE vertex array, draw array call
    #    #self.vertex_array.draw(GL.GL_TRIANGLES)
    #    self.draw_command(GL.GL_TRIANGLES, *self.arguments)
    #    #GL.glDrawArrays(GL.GL_TRIANGLES, self.arguments[0], GL.GL_UNSIGNED_INT, None)
    #    GL.glBindVertexArray(0)


    def draw(self, primitive):
        """draw a vertex array, either as direct array or indexed array"""
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)
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


class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, trans_keys, rotat_keys, scale_keys, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(trans_keys, rotat_keys, scale_keys)

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model, **param)


# -------------- Linear Blend Skinning : TP7 ---------------------------------
MAX_VERTEX_BONES = 4
MAX_BONES = 128

# new shader for skinned meshes, fully compatible with previous color fragment
# TODO: complete the loop for TP7 exercise 1
SKINNING_VERT = """#version 330 core
// ---- camera geometry
uniform mat4 projection, view;

// ---- skinning globals and attributes
const int MAX_VERTEX_BONES=%d, MAX_BONES=%d;
uniform mat4 boneMatrix[MAX_BONES];

// ---- vertex attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 tex;
//layout(location = 2) in vec3 color;
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

        # feel free to move this up in Viewer as shown in previous practicals
        #self.skinning_shader = Shader(SKINNING_VERT, COLOR_FRAG)
        #self.skinning_shader = Shader(SKINNING_VERT, TEXTURE_FRAG)
        self.shader = Shader(SKINNING_VERT, TEXTURE_FRAG)

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = bone_offsets

        self.texture = texture

    def draw(self, projection, view, _model, **_kwargs):
        """ skinning object draw method """

        #shid = self.skinning_shader.glid
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
        GL.glUniform3fv(l_location, 1, (math.cos(t), math.sin(t), 0))

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


# -------- Skinning Control for Keyframing Skinning Mesh Bone Transforms ------
class SkinningControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, *keys, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(*keys) if keys[0] else None
        self.world_transform = identity()

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        if self.keyframes:  # no keyframe update should happens if no keyframes
            self.transform = self.keyframes.value(glfw.get_time())

        # store world transform for skinned meshes using this node as bone
        self.world_transform = model @ self.transform

        # default node behaviour (call children's draw method)
        super().draw(projection, view, model, **param)


# -------------- Deformable Cylinder Mesh  ------------------------------------
class SkinnedCylinder(SkinningControlNode):
    """ Deformable cylinder """
    def __init__(self, sections=11, quarters=20, **params):

        # this "arm" node and its transform serves as control node for bone 0
        # we give it the default identity keyframe transform, doesn't move
        super().__init__({0: (0, 0, 0)}, {0: quaternion()}, {0: 1}, **params)

        # we add a son "forearm" node with animated rotation for the second
        # part of the cylinder
        self.add(SkinningControlNode(
            {0: (0, 0, 0)},
            {0: quaternion(), 2: quaternion_from_euler(90), 4: quaternion()},
            {0: 1}))

        # there are two bones in this animation corresponding to above noes
        bone_nodes = [self, self.children[0]]

        # these bones have no particular offset transform
        bone_offsets = [identity(), identity()]

        # vertices, per vertex bone_ids and weights
        vertices, faces, bone_id, bone_weights = [], [], [], []
        for x_c in range(sections+1):
            for angle in range(quarters):
                # compute vertex coordinates sampled on a cylinder
                z_c, y_c = sincos(360 * angle / quarters)
                vertices.append((x_c - sections/2, y_c, z_c))

                # the index of the 4 prominent bones influencing this vertex.
                # since in this example there are only 2 bones, every vertex
                # is influenced by the two only bones 0 and 1
                bone_id.append((0, 1, 0, 0))

                # per-vertex weights for the 4 most influential bones given in
                # a vec4 vertex attribute. Not using indices 2 & 3 => 0 weight
                # vertex weight is currently a hard transition in the middle
                # of the cylinder
                # TODO: modify weights here for TP7 exercise 2
                #weight = 1 if x_c <= sections/2 else 0
                #if x_c <= sections/4:
                #    weight = 1
                #elif x_c >= 3*sections/4:
                #    weight = 0
                #else:
                #    weight = lerp(3*sections/4, sections/4, 2*x_c/sections)

                #weight = lerp(1, 0, x_c/sections)

                weight = 1/(1 + 2*(-(x_c- sections)/sections))


                bone_weights.append((weight, 1 - weight, 0, 0))

        # face indices
        faces = []
        for x_c in range(sections):
            for angle in range(quarters):

                # indices of the 4 vertices of the current quad, % helps
                # wrapping to finish the circle sections
                ir0c0 = x_c * quarters + angle
                ir1c0 = (x_c + 1) * quarters + angle
                ir0c1 = x_c * quarters + (angle + 1) % quarters
                ir1c1 = (x_c + 1) * quarters + (angle + 1) % quarters

                # add the 2 corresponding triangles per quad on the cylinder
                faces.extend([(ir0c0, ir0c1, ir1c1), (ir0c0, ir1c1, ir1c0)])

        # the skinned mesh itself. it doesn't matter where in the hierarchy
        # this is added as long as it has the proper bone_node table
        self.add(SkinnedMesh([vertices, bone_weights, bone_id, bone_weights],
                             bone_nodes, bone_offsets, faces))


# -------------- 3D resource loader -------------------------------------------
def load_skinned(file):
    """load resources from file using pyassimp, return node hierarchy """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []

    # ----- load animations
    def conv(assimp_keys, ticks_per_second):
        """ Conversion from assimp key struct to our dict representation """
        return {key.time / ticks_per_second: key.value for key in assimp_keys}

    # load first animation in scene file (could be a loop over all animations)
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

    # ---- prepare scene graph nodes
    # create SkinningControlNode for each assimp node.
    # node creation needs to happen first as SkinnedMeshes store an array of
    # these nodes that represent their bone transforms
    nodes = {}  # nodes: string name -> node dictionary

    def make_nodes(pyassimp_node):
        """ Recursively builds nodes for our graph, matching pyassimp nodes """
        trs_keyframes = transform_keyframes.get(pyassimp_node.name, (None,))

        node = SkinningControlNode(*trs_keyframes, name=pyassimp_node.name,
                                   transform=pyassimp_node.transformation)
        nodes[pyassimp_node.name] = node, pyassimp_node
        node.add(*(make_nodes(child) for child in pyassimp_node.children))
        return node

    root_node = make_nodes(scene.rootnode)

    path = os.path.dirname(file)
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))
        if 'file' in mat.tokens:  # texture file token
            print("file in math tokens")
            tname = mat.tokens['file'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            tname = [os.path.join(d[0], f) for d in os.walk(path) for f in d[2]
                     if tname.startswith(f) or f.startswith(tname)]
            print(tname)
            if tname:
                print("tname = %s found" % tname[0])
                mat.texture = Texture(tname[0])
            else:
                print('Failed to find texture:', tname)


    # ---- create SkinnedMesh objects
    for mesh in scene.meshes:
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

        # python, can I please load some textures as well, please?
        print("mat index : ", mesh.materialindex, "all: ", [mesh.materialindex for mesh in scene.meshes])
        print("nb mats ", len(scene.materials), "mats : ", scene.materials)
        #texture = scene.materials[mesh.materialindex].texture

        # for dinosaurs (retarded indices)
        #texture = scene.materials[int(mesh.materialindex/2)].texture
        if mesh.materialindex == 2:
            texture = scene.materials[2].texture
        if mesh.materialindex == 1:
            texture = scene.materials[1].texture

        tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                  if mesh.texturecoords.size else None)
        ##meshes.append(TexturedMesh(texture, [mesh.vertices, tex_uv, mesh.normals], mesh.faces))

        # initialize skinned mesh and store in pyassimp_mesh for node addition
        ##mesh.skinned_mesh = SkinnedMesh(
        ##        [mesh.vertices, mesh.normals, v_bone['id'], v_bone['weight']],
        ##        bone_nodes, bone_offsets, mesh.faces
        ##)
        mesh.skinned_mesh = SkinnedMesh(texture,
                [mesh.vertices, tex_uv, mesh.normals, v_bone['id'], v_bone['weight']],
                bone_nodes, bone_offsets, mesh.faces
        )

    # ------ add each mesh to its intended nodes as indicated by assimp
    for final_node, assimp_node in nodes.values():
        final_node.add(*(_mesh.skinned_mesh for _mesh in assimp_node.meshes))

    nb_triangles = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded', file, '\t(%d meshes, %d faces, %d nodes, %d animations)' %
          (len(scene.meshes), nb_triangles, len(nodes), len(scene.animations)))
    pyassimp.release(scene)
    return [root_node]

# ----------------Probably-useless--------------------------------------------
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
        #sky_faces = ['morning_rt.png', 'morning_lf.png', 'morning_up.png', 'morning_dn.png', 'morning_bk.png', 'morning_ft.png']
        sky_faces = ['morning_ft.png', 'morning_bk.png', 'morning_up.png', 'morning_dn.png', 'morning_rt.png', 'morning_lf.png']
        #sky_faces = ['hills_ft.png', 'hills_bk.png', 'hills_up.png', 'hills_dn.png', 'hills_rt.png', 'hills_lf.png']
        try:
            for i,file in enumerate(sky_faces):
            # imports image as a numpy array in exactly right format
                tex = np.array(Image.open("hw_morning/"+file))
                #tex = np.array(Image.open("sky/"+file))
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



class CubeTexturedPlane:
    """ Simple first textured object """

    def __init__(self, file):
        # feel free to move this up in the viewer as per other practicals
        self.shader = Shader(TEXTURE_VERT_SKY, TEXTURE_FRAG_SKY)
        skybox = iter(SKY_BOX_VERTICES)
        sk_right_triangle = [(x,next(skybox), next(skybox)) for x in skybox]
        #sk_right_triangle = [(x, next(y), next(y)) for y in [skybox_right, skybox_left, skybox_top, skybox_bottom, skybox_back, skybox_front] for x in y ]
        print(sk_right_triangle)
        #skybox_vertices = np.array([skybox_right, skybox_left, skybox_top, skybox_bottom, skybox_back, skybox_front], dtype=np.float32)
        #print(skybox_vertices)
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


# -------------- Example texture plane class ----------------------------------
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




# -------------- Example texture plane class ----------------------------------
#TEXTURE_VERT = """#version 330 core
#uniform mat4 modelviewprojection;
#layout(location = 0) in vec3 position;
#out vec2 fragTexCoord;
#void main() {
#    gl_Position = modelviewprojection * vec4(position, 1);
#    fragTexCoord = position.xy;
#}"""
#
#TEXTURE_FRAG = """#version 330 core
#uniform sampler2D diffuseMap;
#in vec2 fragTexCoord;
#out vec4 outColor;
#void main() {
#    outColor = texture(diffuseMap, fragTexCoord);
#}"""

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
        # feel free to move this up in the viewer as per other practicals
        self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        # triangle and face buffers
        vertices = attributes[0]
        self.tex_uv = attributes[1]
        self.normals = (attributes[2]) # why put them in self?
        #print("THIS IS NORMALS", self.normals)
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
        #GL.glUniform3fv(l_location, 1, (math.cos(t), 0, math.sin(t)))
        GL.glUniform3fv(l_location, 1, (math.cos(t), math.sin(t), 0))

        #GL.glUniform3fv(l_location, 1, (0.5, 0, -0.75))
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
        #self.vertex_array.draw(projection, view, model, self.shader)
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)




class TexturedPlane:
    """ Simple first textured object """

    def __init__(self, file, size=200, step=25):
        # feel free to move this up in the viewer as per other practicals
        self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        [vertices, normals], faces = generate_perlin_grid(size, step=step)
        tex_uv = np.zeros(shape=((size+1)*(size+1), 2))
        for i in range(len(tex_uv)):
            # remember scale for later
            tex_uv[i][0] = (i // (size + 1))/(size + 1) # why +3????
            tex_uv[i][1] = (i % (size + 1))/(size + 1)
            #tex_uv[i][0] = (i // (size + 3)) # why +3????
            #tex_uv[i][1] = (i % (size + 3))
        #randomize_height(vertices)
        self.vertex_array = VertexArray([vertices, normals, tex_uv], faces)

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

        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)

	# lighting coeffs	
        ks_location = GL.glGetUniformLocation(self.shader.glid, 'ks')
        GL.glUniform3fv(ks_location, 1, (0.01,0.01,0.01))

        t = glfw.get_time()
        l_location = GL.glGetUniformLocation(self.shader.glid, 'light')
        GL.glUniform3fv(l_location, 1, (math.cos(t), 0, math.sin(t)))
        #GL.glUniform3fv(l_location, 1, (0.5, 0, -0.75))

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        #self.vertex_array.draw(projection, view, model, self.shader)
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
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


def load_textured(file):
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
        #print(texture)
        # tex coords in raster order: compute 1 - y to follow OpenGL convention
        tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                  if mesh.texturecoords.size else None)
        #print(tex_uv)
        # create the textured mesh object from texture, attributes, and indices
        meshes.append(TexturedMesh(texture, [mesh.vertices, tex_uv, mesh.normals], mesh.faces))

    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes

##################################################################
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
    instances = []
    for inst in range(nb_instances):
        meshes = []
        for mesh in scene.meshes:
            texture = scene.materials[mesh.materialindex].texture
            #print(texture)
            # tex coords in raster order: compute 1 - y to follow OpenGL convention
            tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                      if mesh.texturecoords.size else None)
            #print(tex_uv)
            # create the textured mesh object from texture, attributes, and indices
            meshes.append(TexturedMesh(texture, [mesh.vertices, tex_uv, mesh.normals], mesh.faces))

        size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
        print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))


        instances.append(meshes)

    pyassimp.release(scene)
    return instances 
##################################################################


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
        GL.glEnable(GL.GL_CULL_FACE)
        #GL.glFrontFace ( GL.GL_CW )

        #GL.glCullFace( GL.GL_BACK)
        #GL.glDepthFunc(GL.GL_LEQUAL);
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
        # compile and initialize shader programs once globally

        # initially empty list of object to draw
        self.drawables = []
        self.trackball = GLFWTrackball(self.win)
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

    #def run(self):
    #    """ Main render loop for this OpenGL window """
    #    while not glfw.window_should_close(self.win):
    #        # clear draw buffer
    #        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    #        # draw our scene objects
    #        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
    #        for (i, drawable) in enumerate(self.drawables):
    #            winsize = glfw.get_window_size(self.win)
    #            view = self.trackball.view_matrix()
    #            projection = self.trackball.projection_matrix(winsize)
    #            #self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)
    #            # drawable.draw(None, None, None, self.color_shader)
    #            GL.glUseProgram(self.color_shader.glid)

    #            #transform = (translate(-.5) @ scale(.5) @ rotate(vec(1,0,0), 45))
    #            #matrix_location = GL.glGetUniformLocation(self.color_shader.glid, 'matrix')
    #            #GL.glUniformMatrix4fv(matrix_location, 1, True,  transform)

    #            #drawable.vertex_array.draw(GL.GL_TRIANGLES)
    #            drawable.draw(projection, view, identity(), win=self.win, color_shader=self.color_shader)
    #            #drawable.draw(GL.GL_TRIANGLES, self.color_shader)

    #        # flush render commands, and swap draw buffers
    #        glfw.swap_buffers(self.win)

    #        # Poll for and process events
    #        glfw.poll_events()

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, identity(), win=self.win,
                              color_shader=self.color_shader)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    #def on_key(self, _win, key, _scancode, action, _mods):
    #    """ 'Q' or 'Escape' quits """
    #    if action == glfw.PRESS or action == glfw.REPEAT:
    #        if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
    #            glfw.set_window_should_close(self.win, True)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key == glfw.KEY_SPACE:
                glfw.set_time(0)

class Trex(Node):
    """ Very simple tyranosaurus based on the natural selection and lots of slow prey"""
    def __init__(self):
        Node.__init__(self)
        self.add(*load('trex.obj'))  # just load the cylinder from file

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


SKY_BOX_VERTICES = list(map(lambda x:x*10, SKY_BOX_VERTICES))

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    #viewer.add(*[mesh for file in sys.argv[1:] for mesh in load(file)])
    #viewer.add(*[mesh for file in ["trex.obj"] for mesh in load(file)])

    #cylinder_node = Node(name='my_cylinder', transform=translate(-1, 0, 0), color=(1, 0, 0.5, 0))

    #cylinder_node = Node(name='my_cylinder', transform=translate(-1, 0, 0))
    #cylinder_node.add(Trex())

    #viewer.add(cylinder_node)
    #viewer.add(TexturedPlane('grass_green.png'))


    #------------------Multiple Trees----------------------------------------
    #size, step = 200, 25
    #[vertices, normals], faces = generate_perlin_grid(size, step=step)
    #for i,x in enumerate(normals):
    #    normals[i] = -1*x

    #tex_uv = np.zeros(shape=((size+1)*(size+1), 2))
    #for i in range(len(tex_uv)):
    #    tex_uv[i][0] = (i // (size + 1))/(size + 1)
    #    tex_uv[i][1] = (i % (size + 1))/(size + 1)

    ## THIS IS STUPID REMOVE IT LATER
    #for x in vertices:
    #    x[1], x[2] = x[2], x[1]
    #for x in normals:
    #    x[1], x[2] = x[2], x[1]


    #tree_bases = []
    #for i, x in enumerate(normals):
    #    #print(x[1])
    #    if abs(x[1]) > 0.96:
    #        p = i // (size + 1)
    #        q = i % (size + 1)
    #        tree_bases.append((p, vertices[i][1], q))
    #number_trees = len(tree_bases)
    #print("nb_trees: ", number_trees)
    #if number_trees > 20:
    #    #tree_bases = tree_bases[:20]
    #    tree_bases = [tree_bases[i] for i in range(1, number_trees, number_trees//150)]
    ##viewer.add(TexturedPlane('grass.png'))

    ##trees = load_textured_n_instances("Tree/Tree.obj", 25)
    #trees = load_textured_n_instances("Tree/Tree.obj", len(tree_bases))
    ##tree1, tree2 = load_textured2("Tree/Tree.obj")
    ##tree2 = load_textured("Tree/Tree.obj")
    #viewer.add(TexturedMesh(Texture('ground_tex.png'), [vertices, tex_uv, normals], faces))
    #for i, tree in  enumerate(trees):
    #    for m in tree:
    #        #m.transform = identity()
    #        #m.transform = translate(3*(i//5), 0, 3*(i%5))
    #        m.transform = translate(*(tree_bases[i]))
    #    viewer.add(*tree)
    #        #m.transform = rotate(axis=vec(0,1,0),angle=90.0) 
    ##tree2 = copy.deepcopy(tree1)
    ##for m in tree2:
    ##    m.transform = translate(1,1,1)
    ##viewer.add(*tree1)
    ##viewer.add(*tree2)
    #------------------------------------------------------------------------
    files = os.listdir("sky")
    print(files)

    #cylinder_node = Node(name='my_cylinder', transform=translate(-1, 0, 0), color=(1, 0, 0.5, 1))
    #cylinder_node.add(Cylinder())
    viewer.add(CubeTexturedPlane(files))
    #------------------Textured And Animated Dinosaurs----------------------
    viewer.add(*load_skinned("dino/Dinosaurus_attack.dae"))
    #viewer.add(*load_skinned("ninja/ninja.ms3d"))
    #-----------------------------------------------------------------------
    size, step = 200, 25
    [vertices, normals], faces = generate_perlin_grid(size, step=step)
    for i,x in enumerate(normals):
        normals[i] = -1*x

    tex_uv = np.zeros(shape=((size+1)*(size+1), 2))
    for i in range(len(tex_uv)):
        tex_uv[i][0] = (i // (size + 1))/(size + 1)
        tex_uv[i][1] = (i % (size + 1))/(size + 1)

    # THIS IS STUPID REMOVE IT LATER
    for x in vertices:
        x[1], x[2] = x[2], x[1]
    for x in normals:
        x[1], x[2] = x[2], x[1]


    tree_bases = []
    for i, x in enumerate(normals):
        #print(x[1])
        if abs(x[1]) > 0.96:
            p = i // (size + 1)
            q = i % (size + 1)
            tree_bases.append((p, vertices[i][1], q))
    number_trees = len(tree_bases)
    print("nb_trees: ", number_trees)
    if number_trees > 20:
        #tree_bases = tree_bases[:20]
        tree_bases = [tree_bases[i] for i in range(1, number_trees, number_trees//150)]
    #viewer.add(TexturedPlane('grass.png'))

    #trees = load_textured_n_instances("Tree/Tree.obj", 25)
    trees = load_textured_n_instances("Tree/Tree.obj", len(tree_bases))
    #tree1, tree2 = load_textured2("Tree/Tree.obj")
    #tree2 = load_textured("Tree/Tree.obj")
    viewer.add(TexturedMesh(Texture('ground_tex.png'), [vertices, tex_uv, normals], faces))
    for i, tree in  enumerate(trees):
        for m in tree:
            #m.transform = identity()
            #m.transform = translate(3*(i//5), 0, 3*(i%5))
            m.transform = translate(*(tree_bases[i]))
        viewer.add(*tree)
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
