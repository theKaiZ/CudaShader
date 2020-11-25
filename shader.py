import pygame
import hashlib
from PIL import Image
import os
from ctypes import *
from time import time,sleep
from sys import platform

def isLoaded(lib):
   #this one shows an error on my system recently, but still works
   ret = os.system("lsof | grep " +lib + ">/dev/null" )
   return (ret == 0)
lib = CDLL("libdl.so")

def load_cuda(shader):
  #very important function, compiling the shader files with the Nvidia compiler
  #and loading into ctypes
  #loaded every time a shader is changed or the running shader file is edited
  global mandel
  os.system("nvcc "+shader+" -arch sm_61 -Xcompiler -fPIC -shared -o frac.so")
  LibName = 'frac.so'
  AbsLibPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + LibName
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


class Button():
  active = False
  def __init__(self,parent,pos,size,text):
    self.parent = parent
    self.size = size
    self.pos = pos
    self.text = text
    self.text_surface = parent.myfont.render(text[2:],False,(0,0,0))

  def draw(self):
    pygame.draw.rect(self.parent.screen,(150,150+self.active*50,150),(self.pos[0],self.pos[1],self.size[0],self.size[1]),0)
    pygame.draw.rect(self.parent.screen,(200,200,200),(self.pos[0],self.pos[1],self.size[0],self.size[1]),1)
    x = int(self.pos[0]+int(self.size[0]/2)-len(self.text[2:])*4.5)
    y = int(self.pos[1]+int(self.size[1]/2)-12)
    self.parent.screen.blit(self.text_surface,(x,y))
  def click(self,pos):
    if pos[0] > self.pos[0] and pos[0] < self.pos[0]+self.size[0] and pos[1]>self.pos[1] and pos[1] < self.pos[1]+self.size[1]:
      self.action()
      return True
  def action(self):
    pass

class AdvButton(Button):
  reset = False
  jump = False
  active = False
  aa = False
  group = False
  toggle = False
  def __init__(self,parent,pos,size,text,command,**kwargs):
    self.parent = parent
    self.size = size
    self.pos = pos
    self.text = text
    self.text_surface = parent.myfont.render(text,False,(0,0,0))
    self.command = command
    for param in kwargs:
      setattr(self,param,kwargs[param])
  def action(self):
    self.command()
    if self.aa:
      if self.group:
        for button in self.parent.buttons:
          if button.group == self.group:
            button.active = False
      if self.toggle:
        self.active= not self.active
      else:
        self.active = True
    
    if self.reset:
      os.system("rm -r ./pics/*.png")
      self.parent.reset()
      self.parent.jump(0)
    if self.jump:
      self.parent.jump(0)


class Textfeld():
  def __init__(self,parent,pos,size,value):
    self.parent = parent
    self.size = size
    self.pos = pos
    self.value = value
    self.update()
  def draw(self):
    pygame.draw.rect(self.parent.screen,(150,150,150),(self.pos[0],self.pos[1],self.size[0],self.size[1]),0)
    pygame.draw.rect(self.parent.screen,(200,200,200),(self.pos[0],self.pos[1],self.size[0],self.size[1]),1)
    x = int(self.pos[0]+self.size[0]/2-len(self.text)*4.5)
    y = int(self.pos[1]+self.size[1]/2-12)
    self.parent.screen.blit(self.text_surface,(x,y))
  def update(self):
    wert = getattr(self.parent,self.value)
    if isinstance(wert,float):
      wert = "{0:.3f}".format(wert)
    self.text = str(wert)
    self.text_surface = self.parent.myfont.render(self.text,False,(0,0,0))



class Animation():
    func = "Mandel"
    shaderfile = "shaders/frac.cu"
    loadfunc = True
    size = (640,640)
    screen = False
    buttons = []
    textfelder = []
    folder = ""
    save = False
    fun = None
    f_size = 0
    timestamp= 1
    autozoom = False
    def __init__(self,**kwargs):
        if self.loadfunc:
          self.reset()
        for param in kwargs:
          setattr(self,param,kwargs[param])
        self.init_end()

    def init_end(self):
        if self.loadfunc:
          self.set_cuda_function(self.shaderfile)
        objlength = self.size[0]*self.size[1]*3
        self.result = (c_ubyte*objlength)()

    def set_cuda_function(self,shader):
        print(shader)
        self.shaderfile = shader
        load_cuda(self.shaderfile)
        self.f_size = hash(self.shaderfile)
        self.fun = getattr(mandel,"Mandel")
        mandel.set_window(self.size[0],self.size[1])
        mandel.init_cuda(self.size[0],self.size[1])
        self.frame = 0
        mandel.set_frame(0)
	
    def make_picture(self):
        self.fun(self.size[0],self.size[1],self.result)
        
    def reset(self):
        self.frame = 0
        #os.system("rm log.txt")
        if self.screen:
          self.update()

    def jump(self, steps=1):
        if steps > 0:
           self.log()
        self.frame += abs(steps)
        stamp = time()
        self.background()
        if self.save and steps > 0:
          self.save_pic()
        #print(str("{0:.4f}".format(time() - stamp)) + " Sekunden")
        self.update()
    def run(self):
        s = time()
        self.background()
        self.save_pic()
        print(time()-s)
        
    def background(self):
        self.make_picture()
        if self.screen:
          self.img  = pygame.image.frombuffer(self.result,(self.size[0],self.size[1]),"RGB")
          self.screen.blit(self.img,(0,0))

    def save_pic(self, ff = '.bmp'):
        self.filename = self.folder + "pics/" + (5 - len(str(self.frame))) * "0" + str(self.frame) + ff
        im = Image.frombuffer("RGB",(self.size[0],self.size[1]),self.result,"raw","RGB",0,1)
        im.save(self.filename)
        print(self.frame)

    def update(self):
        #check hash of frac.cu
        if (self.frame % 25 == 0):
          if self.autozoom:
            print(25/(time()-self.timestamp),"FPS")
            self.timestamp = time()
          f = hash(self.shaderfile)
          if f != self.f_size:
            self.set_cuda_function(self.shaderfile)
        #if hash is different, recompile stuff
        #wait
        #load function
        mandel.set_frame(self.frame)
        if self.screen:
          pos = pygame.mouse.get_pos()
          #optimiere das mouse ding, da ist noch "verzug"
          mandel.set_mouse(c_float((pos[0]-self.size[0]/2)/self.size[0]),c_float((pos[1]-self.size[1]/2)/self.size[1]))
    
        for textfeld in self.textfelder:
            textfeld.update()
        for button in self.buttons:
            button.draw()
        for textfeld in self.textfelder:
            textfeld.draw()
    def toggle(self, value):
        setattr(self, value, not getattr(self, value))

    def add_value(self, attr, value):
        setattr(self, attr, getattr(self, attr) + value)

    def mult_value(self, attr, value):
        setattr(self, attr, getattr(self, attr) * value)

    def mk_video(self,ff = '.bmp'):
        os.system("ffmpeg -f image2 -i ./pics/%05d"+ff+" -pix_fmt yuv420p -y out.mp4")

    def log(self):
        return
        with open("log.txt","a") as f:
          pass

    def make_buttons(self):
        self.buttons = []
        y = 0
        for f in os.listdir("shaders"):
          self.buttons.append(AdvButton(self,(self.size[0],y*20),(200,20), f,(lambda x =f:self.set_cuda_function("shaders/"+x)),reset=True,aa=True,group="Shaders"))
          y+=1

        self.buttons.append(AdvButton(self, (self.size[0], self.size[1]), (50, 30), "Video", lambda: self.mk_video()))
        self.buttons.append(
            AdvButton(self, (self.size[0], self.size[1] + 30), (50, 30), "Auto", lambda: self.toggle("autozoom"),
                      aa=True, toggle=True))
        self.textfelder.append(Textfeld(self, (0, self.size[1] + 30), (70, 30), "frame"))


    def window(self):
        '''Initialisiere Alle Einstellungen und Buttons für Pygame'''
        running = True
        left_click = right_click = False
        pygame.init()
        self.myfont = pygame.font.SysFont("Arial", 15 if 'win' in platform else 22)
        self.screen = pygame.display.set_mode((self.size[0] + 200, self.size[1] + 60))
        self.make_buttons()
        for button in self.buttons:
            button.draw()
        for textfeld in self.textfelder:
            textfeld.draw()
        self.jump(0)
        tick = time()
        while running:
            if left_click:
              pass
            elif right_click:
              pass
            elif self.autozoom:
              while (time()-tick) < 1/40:
                sleep(0.001)
              self.jump(1)
              tick = time()
            elif not self.autozoom:
              sleep(0.1)
              self.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_F2:
                        self.frame = 0
                    elif event.key == pygame.K_F3:
                        self.make_buttons()
                        self.update()
                    elif event.key == pygame.K_F5:
                        self.save_spot()
                    elif event.key == pygame.K_F9:
                        self.load_spot()
                    elif event.key == pygame.K_MINUS:
                        self.jump(-self.steps)
                        for textfeld in self.textfelder:
                            textfeld.draw()
                    elif event.key == pygame.K_PLUS:
                        self.jump(self.steps)
                        for textfeld in self.textfelder:
                            textfeld.draw()
                    elif event.key == pygame.K_s:
                        ##das kann man vlt noch Threaden!
                        pass
                elif event.type == pygame.MOUSEBUTTONDOWN:
                  if event.button == 1:
                    left_click = True
                  if event.button==3:
                    right_click = True 
                    #self.jump(-self.steps)
                  pos = pygame.mouse.get_pos()
                  for button in self.buttons:
                            button.click(pos)
                  self.update()
                  for button in self.buttons:
                        button.draw()
                  for textfeld in self.textfelder:
                        textfeld.draw()
                elif event.type == pygame.MOUSEBUTTONUP:
                 if event.button == 1:
                   left_click = False
                 elif event.button == 3:
                   right_click = False
            pygame.display.flip()
        mandel.exit_cuda() #important to free the memory in cuda
        pygame.quit()

if __name__ == '__main__':
    Animation(size=(720,720),autozoom=0).window()
