from shader import *

def main():
  A = Animation(size=(640,640),save=True,shaderfile="shaders/test.cu")
  for i in range(360):
    A.jump(1)
  A.mk_video()

if __name__ == '__main__':
  main()
