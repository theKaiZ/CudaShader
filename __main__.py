from myGUI.GUI import myGUI, Button, ScrollTextfeld
from os import listdir
from os.path import join
import numpy as np
from sys import platform
from numba import cuda
if cuda.is_available():
    if platform == "linux":
        from cpp_cuda.Shader import Shader
    else:
        from numba_cuda.Shader import Shader
else:
    from numba_jit.Shader import Shader

def var_setter(parent, pos, key, change_value, operator, limits):
    pos = np.array(pos)
    Button(parent, pos, (50,20), key)
    ScrollTextfeld(parent, pos +(50,0), (100,20), key=key, change_value = change_value, operator=operator,limits= limits)

class ShaderGUI(myGUI):
    pos = np.array([0,0])
    size = np.array([1200,900])
    FPS = 25
    shader = None
    min_dist = 0.015
    eps = 0.01
    def exit_game(self):
        if self.shader is not None:
            self.shader.removefromGUI()

    def setup_buttons(self):
        dir = "cpp_cuda/shaders"
        for i, filename in enumerate(listdir(dir)):
            Button(self, (self.size[0]-100,i*20),(100,20), filename,command=lambda f=filename:self.set_shader(join(dir,f)))

        var_setter(self, (0,self.size[1]-100),  "min_dist",1.1, operator="*", limits=[0.00001, 0.5])
        var_setter(self, (0,self.size[1]-80),  "eps",1.1, operator="*", limits=[0.00001, 0.5])

    def set_shader(self, shaderfile):
        if self.shader is not None:
            print("delete old shader")
            self.shader.removefromGUI()
            self.shader = None
        self.shader = Shader(self,size=(720,720),shaderfile=shaderfile)


if __name__ == '__main__':
    ShaderGUI().run()
