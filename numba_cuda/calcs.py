from numba import cuda
import math


@cuda.jit(device=True)
def length(x,y,z):
    return math.sqrt(x * x + y * y + z * z)

@cuda.jit(device=True)
def dist(x1,y1,z1, x2,y2,z2):
    dx = x1-x2
    dy = y1-y2
    dz = z1-z2
    return length(dx,dy,dz)

@cuda.jit(device=True)
def norm(x,y,z):
    l = length(x,y,z)
    return x/l,y/l, z/l

@cuda.jit(device=True)
def sd_sphere(x,y,z):
    return dist(x,y,z, 1,1,2) -1


@cuda.jit(device=True)
def map(x,y,z):
    return sd_sphere(x,y,z)


@cuda.jit(device=True)
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
    return dist



@cuda.jit()
def shader_kernel(img, size):
    """
    """
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    idx = 3*(row * size[0] + col)


    y = - (row - size[0]/2)/size[0]/2
    x = (col -size[1]/2)/size[1]/2
    #direction
    x,y,z = norm(x,y,1.0)
    #origin
    x0,y0,z0 = 1.0,1.0,-3.0
    ray = trace(x0,y0,z0,x,y,z)
    img[idx] = ray*100
    img[idx+1] = ray*100
    img[idx+2] = ray*100


def calc_shader_on(imgarray, size) -> None:
    """

    """
    blockdim = (16, 16)
    griddim = (size[0]//16, size[1]//16)
    stream = cuda.stream()
    with stream.auto_synchronize():
        # transfer our numpy arrays to the GPU memory, the stream is not necessary
        device_imgarray = cuda.to_device(imgarray, stream=stream)
        device_size = cuda.to_device(size, stream=stream)
        # finally running the kernel to start the calculation with all necessary data
        shader_kernel[griddim, blockdim, stream](device_imgarray, size)
        # in the end, copy the result from the GPU memory back to our numpy array
        device_imgarray.copy_to_host(imgarray, stream=stream)

