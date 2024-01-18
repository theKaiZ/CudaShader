from myGUI.GUI import myGUI, Button
from os import listdir
from os.path import join
import numpy as np
from sys import platform
from numba import cuda
if cuda.is_available():
    if platform == "linax":
        from cpp_cuda.Shader import Shader
    else:
        from numba_cuda.Shader import Shader

#from cpp_cuda.Shader import Shader

class ShaderGUI(myGUI):
    pos = np.array([0,0])
    size = (640,640)
    FPS = 25
    shader = None


    def exit_game(self):
        self.shader.removefromGUI()

    def setup_buttons(self):
        dir = "cpp_cuda/shaders"
        for i, filename in enumerate(listdir(dir)):
            Button(self, (self.size[0],i*20),(100,20), filename,command=lambda f=filename:self.set_shader(join(dir,f)))

    def set_shader(self, shaderfile):
        if self.shader is not None:
            print("delete old shader")
            self.shader.removefromGUI()
            self.shader = None
        self.shader = Shader(self, shaderfile)


if __name__ == '__main__':
    ShaderGUI().run()