import numpy as np
from transform import lerp, normalized
import random
import pickle
from gen_texture import generate_texture
from math import cos, sin, pi, floor, sqrt

def make_vertex(vertices, n, x, y, height, width, centered=False):
    if centered:
        vertices[n][0] = x - (height + 1)/2
        vertices[n][2] = y - (width + 1)/2
    else:
        vertices[n][0] = x
        vertices[n][2] = y

def generate_grid(height, width=None, scale=1, centered=False):
    if width==None:
        width = height
    vertices = np.zeros(shape=((height + 1)*(width + 1), 3), dtype=np.float32)
    faces = np.zeros(shape=(height*width*2, 3), dtype=np.uint32)
    for i in range(height):
        for j in range(0, width):
            n = i * (width + 1) + j
            # generate one vertex
            make_vertex(vertices, n, i, j, height, width, centered)
            # generate two faces
            faces[i*2*width + 2*j][0] = n
            faces[i*2*width + 2*j][1] = n + 1
            faces[i*2*width + 2*j][2] = n + width + 1
            faces[i*2*width + 2*j + 1] [0] = n + 1
            faces[i*2*width + 2*j + 1][1] = n + 2 + width
            faces[i*2*width + 2*j + 1][2] = n + width + 1
        # last vertex in the row
        make_vertex(vertices, i*(width + 1) + width, i, j, height, width, centered)
    # last row of vertices
    for j in range(width + 1):
        make_vertex(vertices, height*(width + 1) + j, i, j, height, width, centered)
    print("scale : ", scale)
    return scale*vertices, faces

def randomize_gradient(width, height, step):
    gradient = np.zeros(shape=((width + 1)*(height + 1), 2), dtype=float)
    for p in range(0, height + 1, step):
        for q in range(0, width + 1, step):
            gradient[p*(width + 1) + q][0]=random.uniform(-1,1)
            gradient[p*(width + 1) + q][1]=random.uniform(-1,1)
    return gradient

def dotGridGradient(gradient, width, ix, iy, x, y):
    dx = x - ix;
    dy = y - iy;
    return (dx*gradient[iy + (width + 1)*ix][0] + dy*gradient[iy + (width + 1)*ix][1]);

def get_normal(a, b):
    cr = np.cross(a, b)
    return normalized(cr)

def avg_neighbours_grid(array, x, y, height, width, normals_proj):
    neighbours = []
    o = array[x*(width + 1) + y]
    if x > 0:
        b = array[(x-1)*(width + 1) + y]
    if x < height:
        u = array[(x+1)*(width + 1) + y]
    if y > 0:
        l = array[x*(width + 1) + y-1]
    if y < width:
        r = array[x*(width + 1) + y+1]

    if x > 0:
        if y > 0:
             neighbours.append(get_normal(l - o, b - o))
        if y < width:
            neighbours.append(get_normal(b - o, r - o))
    if x < height:
        if y < width:
            nrm = get_normal(r - o, u - o)
            neighbours.append(nrm)
            normals_proj[x*width + y] = abs(nrm[1])
        if y > 0:
            nrm = get_normal(u - o, l - o)
            neighbours.append(nrm)
    res = neighbours[0]
    N = len(neighbours)
    for n in range(1, len(neighbours)):
        res = lerp(res, neighbours[n], (N - n)/N)
    res = normalized(res)
    norm = sqrt(sum(res*res))
    if norm < 0.1:
        res[0] = 0
        res[1] = 1
        res[2] = 0
    return normalized(res)


def get_normals(vertices, height, width):
    normals = np.zeros(shape=((width + 1)*(height + 1), 3), dtype=float)
    normals_proj = np.zeros(shape=(height * width))
    for p in range(height + 1):
        for q in range(width + 1):
            normals[p*(width + 1) + q] = avg_neighbours_grid(vertices, p, q, height, width, normals_proj)
    return normals, normals_proj

def lerp_smooth(a, b, fraction):
    f = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    return f(1-fraction)*a + f(fraction)*b

def generate_perlin_grid(height, width=None, step=25, scale=1, centered=False):
    print("Generating the ground")

    if width==None:
        width = height

    magnitude = 0.5
    vertices, faces = generate_grid(height, width, scale, centered)
    for i in range(3):
        gradient = randomize_gradient(width + step, height + step, step)
        for p in range(0, height):
            for q in range(0, width):
                p0, q0 = step*(p // step), step*(q // step)
                p1, q1 = p0 + step, q0 + step
                sp, sq = p % step, q % step
                n0 = dotGridGradient(gradient, width + step, p0, q0, p, q)
                n1 = dotGridGradient(gradient, width + step, p1, q0, p, q)
                ix0 = lerp_smooth(n0, n1, sp/step)
                n0 = dotGridGradient(gradient, width + step, p0, q1, p, q)
                n1 = dotGridGradient(gradient, width + step, p1, q1, p, q)
                ix1 = lerp_smooth(n0, n1, sp/step)
                value = lerp_smooth(ix0, ix1, sq/step)
                vertices[p*(width + 1) + q][1] += magnitude*value
        step = step // 2 + 1
        magnitude *= 0.7

    print("Generation of the ground mesh complete. Calculating the normals.")
    normals, normals_proj = get_normals(vertices, height, width)
    print("Calculation of normals complete.")

    return (vertices, normals, faces)
