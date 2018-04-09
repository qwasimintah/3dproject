import png
from transform import lerp
import pickle
import numpy as np


def lerp_smooth(a, b, fraction):
    f = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    return f(1-fraction)*a + f(fraction)*b

def generate_texture(normals_proj, height, step):
    with open("ground", "rb") as fp:
        [vertices, normals, faces, normals_proj] = pickle.load(fp)
    fp.close()

    f = open('ground_tex.png', 'wb')      # binary mode is important
    width = height
    res = 15
    #print(height, " ", width)
    w = png.Writer(height*res, width*res)
    tex = np.zeros(shape=(height*res,width*res,3))
    print(max(normals_proj))
    for i in range(height):
        for j in range(width):
            #print(i, j)
            x = normals_proj[i*width + j]

            # strict
            #if x > 0.85:
            #    for u in range(res):
            #        for v in range(res):
            #            tex[j*res+v][i*res+u] = np.array([0x26,0x60,0x0f])
            #else:
            #    for u in range(res):
            #        for v in range(res):
            #            tex[j*res+v][i*res+u] = np.array([128,128,128])

            # lerp
            for u in range(res):
                for v in range(res):
                    #tex[j*res+v][i*res+u] = lerp(np.array([0x4c,0x4f,0x37]), np.array([0x26,0x60,0x0f]), x)
                    tex[j*res+v][i*res+u] = lerp_smooth(np.array([0x4c,0x4f,0x47]), np.array([0x26,0x60,0x0f]), x)

    print("Texture generation complete")
    w.write(f, np.reshape(tex, (-1, width*res*3)))
    #w.write(f, tex)
    #w.write(f, [range(256) for i in range(256)])

    f.close()

if __name__ == "__main__":
    generate_texture([], 200, 25)
