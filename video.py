from shader import *
from SpeedTest import *




@time_runtime
def main():
  A = Animation(size=(1280,1280),save=True,shaderfile="shaders/test.cu")
  #for i in range(360):
  #  A.jump(1)
  A.mk_video()


if __name__ == '__main__':
  main()