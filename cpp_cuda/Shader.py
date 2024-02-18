import pygame
import hashlib
import numpy as np
from PIL import Image
from os import system,mkdir,listdir
from os.path import exists,abspath,sep,dirname, join
from ctypes import *
from myGUI.GUI import myGUI, Rectangular_object, Button

def isLoaded(lib):
   #this one shows an error on my system recently, but still works
   ret = system("lsof | grep " +lib + ">/dev/null" )
   return (ret == 0)
lib = CDLL("libdl.so")

def load_cuda(shader):
   '''very important function, compiling the shader files with the Nvidia compiler
   and loading into ctypes
   loaded every time a shader is changed or the running shader file is edited'''
   global mandel
   system(f"nvcc {shader} -arch sm_61 -Xcompiler -fPIC -shared -o cpp_cuda/frac.so")
   LibName = 'frac.so'
   AbsLibPath = dirname(abspath(__file__)) + sep + LibName
   while isLoaded("frac.so"): #this closes the old ctypes function if loaded
      mandel.exit_cuda()
      lib.dlclose(mandel._handle)
   mandel = CDLL(AbsLibPath,mode=RTLD_LOCAL)

def hash(filename):
   h = hashlib.sha256()
   b  = bytearray(128*1024)
   mv = memoryview(b)
   with open(filename, 'rb', buffering=0) as f:
      for n in iter(lambda : f.readinto(mv), 0):
          h.update(mv[:n])
   return h.hexdigest()


class Shader(Rectangular_object):
    _image = None

    def __init__(self, parent, shaderfile, **kwargs):
        super().__init__(parent, **kwargs)
        self.shaderfile = shaderfile
        load_cuda(shaderfile)
        self.f_size = hash(self.shaderfile)
        self.fun = getattr(mandel, "Mandel")

        objlength = self.size[0] * self.size[1] * 3
        self.result = (c_ubyte * objlength)()
        ### set the window size and stuff
        mandel.set_vec2(0,c_float(self.size[0]),c_float(self.size[1]))
        mandel.init_cuda(c_int(self.size[0]),c_int(self.size[1]))
        self.frame = 0
        mandel.set_int(0,0)

    def draw(self):
        self.screen.blit(self.image, self.pos)

    def update(self):
        self._image = None
        #self.frame += 1
        mandel.set_int(0,c_int(self.frame))
        mandel.set_float(0, c_float(self.parent.min_dist))
        mandel.set_float(1, c_float(self.parent.eps))

    @property
    def image(self):
        if self._image is not None:
            return self._image
        self.fun(self.result)
        self._image = pygame.image.frombuffer(self.result, (self.size[0], self.size[1]), "RGB")
        self.parent.toggle_update = True
        return self._image

    def removefromGUI(self):
        super().removefromGUI()
        mandel.exit_cuda()


