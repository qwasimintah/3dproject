import numpy as np
import math
from math import floor
from transform import lerp, normalized
import numpy as np
import random

def make_vertex(vertices, n, x, y, height, width, centered=False):
    if centered:
        vertices[n][0] = x - (height + 1)/2
        vertices[n][1] = y - (width + 1)/2
    else:
        vertices[n][0] = x
        vertices[n][1] = y

def generate_grid(height, width=None, scale=1, centered=False):
    if width==None:
        width = height
    #vertices = np.array([], np.float32)
    #faces = np.array([], np.uint32)
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
    return scale*vertices, faces


def randomize_height(height_map, width, height, step):
    for p in range(0, height, step):
        for q in range(0, width, step):
            height_map[p*width + q][2]=random.gauss(0, 0.50)

defined_grad = []
used_grad = []

def randomize_gradient(width, height, step):
    #gradient = [0]*(width*height)
    gradient = np.zeros(shape=((width + 1)*(height + 1), 2), dtype=float)

    #global defined_grad

    for p in range(0, height + 1, step):
        for q in range(0, width + 1, step):
            #gradient[p*width + q][0]=random.gauss(0, 0.05)
            #gradient[p*width + q][1]=random.gauss(0, 0.05)
            #defined_grad.append(p*width + q)
            gradient[p*width + q][0]=random.uniform(-1,1)
            gradient[p*width + q][1]=random.uniform(-1,1)
    #return 0.08*gradient
    return gradient

def dotGridGradient(gradient, width, ix, iy, x, y):
    dx = x - ix;
    dy = y - iy;

    #global used_grad
    #if iy + width*ix not in used_grad:
    #    used_grad.append(iy + width*ix)

    return (dx*gradient[iy + width*ix][0] + dy*gradient[iy + width*ix][1]);


#def perlin(x, y, vertices):
#    x0 = floor(x)
#    x1 = x0 + 1
#    y0 = floor(y)
#    y1 = y0 + 1
#
#    sx = x - x0
#    sy = y - y0
#
#    n0 = dotGridGradient(x0, y0, x, y)
#    n1 = dotGridGradient(x1, y0, x, y)
#    ix0 = lerp(n0, n1, sx)
#    n0 = dotGridGradient(x0, y1, x, y)
#    n1 = dotGridGradient(x1, y1, x, y)
#    ix1 = lerp(n0, n1, sx)
#    value = lerp(ix0, ix1, sy)
#
#    return value

def get_normal(a, b):
    cr = np.cross(a, b)
    return normalized(cr)

def avg_neighbours_grid(array, x, y, height, width):
    neighbours = []
    if x > 0:
        if y > 0:
             neighbours.append(get_normal(array[x*(width + 1) + y-1] - array[x*(width + 1) + y], array[(x-1)*(width + 1) + y] - array[x*(width + 1) + y]))  
        if y < width:
            neighbours.append(get_normal(array[(x-1)*(width + 1) + y] - array[x*(width + 1) + y], array[x*(width + 1) + y+1] - array[x*(width + 1) + y]))  
    if x < height:
        if y < width:
            neighbours.append(get_normal(array[x*(width + 1) + y+1] - array[x*(width + 1) + y], array[(x+1)*(width + 1) + y] - array[x*(width + 1) + y]))  
        if y > 0:
            neighbours.append(get_normal(array[(x+1)*(width + 1) + y] - array[x*(width + 1) + y], array[x*(width + 1) + y-1] - array[x*(width + 1) + y]))  
    res = 0*neighbours[0]
    N = len(neighbours)
    for n in range(len(neighbours)):
        res = lerp(res, neighbours[n], (N - n)/N)
    res = normalized(res)
    norm = math.sqrt(sum(res*res))
    if norm < 0.1:
        res[0] = 0
        res[1] = 0
        res[2] = -1
    return normalized(res)


def get_normals(vertices, height, width):
    normals = np.zeros(shape=((width + 1)*(height + 1), 3), dtype=float)
    for p in range(height + 1):
        for q in range(width + 1):
            normals[p*(width + 1) + q] = avg_neighbours_grid(vertices, p, q, height, width)
    return normals
	    	

def generate_perlin_grid(height, width=None, step=25, scale=1, centered=False):
    if width==None:
        width = height
    vertices, faces = generate_grid(height, width, scale, centered)
    gradient = randomize_gradient(width, height, step)
    for p in range(0, height - step):
        for q in range(0, width - step):
            #if p%step != 0 and q%step != 0:
            if True:
                p0, q0 = step*(p // step), step*(q // step)
                #p1, q1 = p0 + 1, q0 + 1
                p1, q1 = p0 + step, q0 + step
                sp, sq = p % step, q % step
                n0 = dotGridGradient(gradient, width, p0, q0, p, q)
                n1 = dotGridGradient(gradient, width, p1, q0, p, q)
                ix0 = lerp(n0, n1, sp/step)
                #ix0 = lerp(n0, n1, sp)
                n0 = dotGridGradient(gradient, width, p0, q1, p, q)
                n1 = dotGridGradient(gradient, width, p1, q1, p, q)
                ix1 = lerp(n0, n1, sp/step)
                #ix1 = lerp(n0, n1, sp)
                value = lerp(ix0, ix1, sq/step)
                #value = lerp(ix0, ix1, sq)

                vertices[p*width + q][2] = value
    #print([z[2] for z in vertices])
    normals = get_normals(vertices, height, width)

    #print(len(used_grad))
    #print(len(defined_grad))
    #print("=================================")
    #for i in defined_grad:
    #    if i not in used_grad:
    #        print(i)

    return [vertices, normals], faces