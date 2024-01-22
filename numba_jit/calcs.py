from numba import njit, prange
import numpy as np
from math import sqrt

@njit(cache=True, fastmath=True)
def length(x,y,z):
    return sqrt(x * x + y * y + z * z)

@njit(cache=True, fastmath=True)
def dist(x1,y1,z1, x2,y2,z2):
    dx = x1-x2
    dy = y1-y2
    dz = z1-z2
    return length(dx,dy,dz)

@njit(cache=True, fastmath=True)
def norm(x,y,z):
    l = length(x,y,z)
    return x/l,y/l, z/l

@njit(cache=True)
def sd_sphere(x,y,z):
    return dist(x,y,z, 1,1,4.5) -1


@njit(cache=True)
def map(x,y,z):
    return sd_sphere(x,y,z)


"""@njit(cache=True)
def trace(x0,y0,z0, x,y,z):
    dist = 0.0
    for i in range(200):
        px = x0+ x*dist
        py = y0 + y*dist
        pz = z0 + z*dist
        d = map(px,py,pz)
        if d<0.01:
            break
        dist+=d
    return (dist)"""

@njit(cache=True)
def trace(x0,y0,z0, x,y,z):
    dist = 0.0
    for i in range(50):
        d = map(x0,y0,z0)*1.2
        x0 += x*d
        y0 += y*d
        z0 += z*d
        dist += d
        if d<0.01:
            break
    return dist



@njit(cache=True )
def shader_kernel(img, size):
    """
    """
    for row in prange(size[0]):
        for col in prange(size[1]):
            idx = 3*(row * size[0] + col)
            y = - (row - size[0]/2)/size[0]/2
            x = (col -size[1]/2)/size[1]/2
            #direction
            x,y,z = norm(x,y,1.0)
            #origin
            x0,y0,z0 = 1.0,1.0,-3.0
            ray = trace(x0,y0,z0,x,y,z)
            img[idx] = ray*10
            img[idx+1] = ray*10
            img[idx+2] = ray*10


def calc_shader_on(imgarray, size) -> None:
    """

    """
    shader_kernel(imgarray, size)
