import pygame
import hashlib
import numpy as np
from PIL import Image
from os import system,mkdir,listdir
from os.path import exists,abspath,sep,dirname, join
from myGUI.GUI import myGUI, Rectangular_object, Button
from numba_cuda.calcs import calc_shader_on
from time import time

class Shader(Rectangular_object):
    _image = None

    def __init__(self, parent, shadername=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.fun = calc_shader_on
        self.size = np.array([480,480])

        objlength = self.size[0] * self.size[1] * 3
        self.result = np.zeros(objlength, dtype=np.uint8)
        ### set the window size and stuff
        self.frame = 0

    def draw(self):
        t = time()
        self.screen.blit(self.image, self.pos)
        print(f"{(time()-t)*1000:.2f}")
    def update(self):
        self._image = None
        self.frame += 1

    @property
    def image(self):
        if self._image is not None:
            return self._image
        self.fun(self.result, self.size)
        #result = self.result.reshape((self.size[0],self.size[1],3))
        self._image = pygame.image.frombuffer(self.result, (self.size[0], self.size[1]), "RGB")
        self.parent.toggle_update = True
        return self._image

    def removefromGUI(self):
        super().removefromGUI()


