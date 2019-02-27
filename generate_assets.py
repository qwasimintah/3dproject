from math import cos, sin, pi
from gen_texture import generate_texture
import pickle
import numpy as np
from generate_grid import generate_perlin_grid
from random import uniform

def geizer():
    """ Model of a geizer's basis"""
    w = 2*pi/6
    vertices = []
    faces = []
    for i in range(6):
        vertices.append(np.array([cos(i*w), 0, sin(i*w)]))
        faces.append(np.array([i, (i+1)%6, 6+i]))
        faces.append(np.array([(i+1)%6, 6+i, 6+((i+1)%6)]))
    for i in range(6):
        vertices.append(np.array([cos(i*w), 0.2 , sin(i*w)]))
        faces.append(np.array([6+i, (i+1)%6+6, 12+i]))
        faces.append(np.array([(i+1)%6+6, 12+i, 12+((i+1)%6)]))
    for i in range(6):
        vertices.append(np.array([0.7*cos(i*w), 0.6 , 0.7*sin(i*w)]))
        faces.append(np.array([12+i, (i+1)%6+12, 12+18]))
    vertices.append(np.array([0,0.4,0]))
    vertices = 0.25*np.array(vertices, dtype=np.float)
    normals = vertices.copy() # approximation
    return vertices, normals, faces

def sphere(n=20):
    """ Model of a sphere"""
    vertices = np.zeros(shape=(n*n+2,3), dtype=np.float32)
    faces =[]
    north = n*n
    south = n*n+1
    vertices[south] = np.array([0, -1, 0])
    for j in range(n):
        a = 2*j*pi/n
        r = 1*pi/n - pi/2
        vertices[j] = np.array([cos(r)*cos(a), sin(r), cos(r)*sin(a)])
        faces.append(np.array([south, j, j+1 % n]))
    for i in range(1,n):
        r = i*pi/n - pi/2
        for j in range(n):
            a = 2*j*pi/n
            vertices[i*n+j] = np.array([cos(r)*cos(a), sin(r), cos(r)*sin(a)])
            faces.append(np.array([(i-1)*n+j, i*n + j, i*n + (j+1 % n)]))
            faces.append(np.array([(i-1)*n+j, (i-1)*n + (j+1 % n), i*n + (j+1 % n)]))
    vertices[north] = np.array([0, 1, 0])
    for j in range(n):
        faces.append(np.array([north, n*(n-1)+j, n*(n-1)+((j + 1)%n)]))
    normals = vertices.copy()
    return vertices, normals, faces

def grid(height, width=None, step=25, scale=1, centered=False, new=False, textured=False):
    """ Open or generate an uneven ground, as well as a texure for it"""
    if not new:
        try:
            with open("ground", "rb") as fp:
                ground = pickle.load(fp)
        except:
            print("Cannot find an existing ground.")
            ground = generate_perlin_grid(height, width=width, step=step, centered=centered, scale=scale)
            try:
                with open('ground', 'wb') as fp:
                    pickle.dump(ground, fp)
            except:
                print("Could not save the generated ground.")
        tname = "GeneratedGroundTexture.png"
    else:
        ground = generate_perlin_grid(height, width=width, step=step, centered=centered, scale=scale)
        try:
            with open('ground', 'wb') as fp:
                pickle.dump(ground, fp)
        except:
            print("Could not save the generated ground.")

        if textured:
            try:
                generate_texture(ground[3], height, width)
                tname = "GeneratedGroundTexture.png"
            except:
                print("Cannot generate a texture for the ground, probably due to the absense of PyPNG.")
                print("Using plain green texture instead.")
                tname = "DefaultGroundTexture.png"
        else:
            tname = "DefaultGroundTexture.png"

    return ground[0], ground[1], ground[2], tname

def fountain(t, PARTICLES):
    translations = np.zeros(shape=(PARTICLES, 3))
    g = -9.82
    #spectrum = 70*uniform(0,1)*2*pi/PARTICLES
    spectrum = 30*2*pi/PARTICLES
    vy = 50
    d = 2
    tau = -vy/g
    tfast = t*0.5
    for i in range(PARTICLES):
        tfasti = (tfast + i*tau/PARTICLES) % tau
        #y = g[1]*tfasti*tfasti + (0.5*((i+15)/(15+PARTICLES)) *tfasti)
        translations[i][2] = sin(i*spectrum)*d*tfasti
        translations[i][1] = g*tfasti*tfasti + vy*tfasti
        translations[i][0] = cos(i*spectrum)*d*tfasti
    return translations

