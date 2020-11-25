from shader import *

A = Animation(size=(1280,1280),save=True,shaderfile="shaders/test.cu")
for i in range(360):
  A.jump(1)
A.mk_video()
