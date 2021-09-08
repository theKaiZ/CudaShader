#include <math.h>
#include <stdio.h>
#include "../shader.cu"

#define EPS 0.0125
#define MIN_DIST 0.0001

__device__ float mandelbulb(vec3 pos){
  float dr = 1;
  float r = 1;
  vec3 z = pos;
  float power =5 + cosfr;
  for(int i = 0; i < 15; i++){
    r = length(z);
    if(r>2.5)
      break;
    float theta = acos(z.z /r)  * power;
    float phi = atan2(z.y,z.x) * power;
    float zr = pow(r,power);
    dr = pow(r,power-1)*power*dr+1;
    
    z = vec3(sin(theta)*cos(phi),sin(phi)*sin(theta),cos(theta))*zr;
    z = z+ pos;
  
  }
  return 0.5 * log(r) * r /dr;
}




__device__ vec2 map(vec3 p){  
  vec2 plane = vec2(p.y +1.5-sinfr*0.5,4); 
  vec2 plane2 = vec2(abs(-p.z+1.5+sinfr*0.5 + 0.15*sin(frame/15+length(vec2(p.x,p.y))*5)),4); 
  vec2 plane3 = vec2(-p.x+2.0 + sinfr*0.5, 4);
  vec2 plane4 = vec2(abs(p.x+4),4);
  vec2 plane5 = vec2(abs(p.y - 3),4);
  //vec2 plane = vec2(-(p.x+p.z)*0.025+ p.y+0.25*sin(frame/15 + length(pxz)*5)+0.5;
  //vec2 cube = vec2(length(max(abs(p-vec3(0,1+sinfr,0))-vec3(0.5),0.0+sinfr*0.275))-0.5,1);
  //cube = abs(cube)-0.1;
  //vec2 ball = vec2(dist(p+vec3(0.7-3,-0.8,0.2),vec3(0.5))-1,2);
  vec2 mandel(mandelbulb(p),5);
  return min(min(min(min(min(mandel,plane),plane2),plane3),plane4),plane5);
}

__device__ vec3 normal(vec3 p){
  vec3 q = vec3(map(vec3(p.x + EPS, p.y, p.z)).x - map(vec3(p.x - EPS, p.y, p.z)).x,
            map(vec3(p.x, p.y + EPS, p.z )).x - map(vec3(p.x, p.y - EPS, p.z)).x,
            map(vec3(p.x, p.y, p.z + EPS)).x - map(vec3(p.x, p.y, p.z - EPS)).x);
  return normalize(q); 
} 

__device__ vec3 normal(vec3 p,float E){
  vec3 q = vec3(map(vec3(p.x + E, p.y, p.z)).x - map(vec3(p.x - E, p.y, p.z)).x,
            map(vec3(p.x, p.y + E, p.z )).x - map(vec3(p.x, p.y - E, p.z)).x,
            map(vec3(p.x, p.y, p.z + E)).x - map(vec3(p.x, p.y, p.z - E)).x);
  return normalize(q); 
} 
 
__device__ vec3 trace(vec3 org, vec3 dir){
   vec3 cl,p;
   vec3 light = vec3(sinfr*2,5.0+3*sinfr,-2.0);
   vec2 d;
   float dist = 0.0;   
   for(int i = 0; i < 256; i++)
   {
     d.y = 0;
     p = org+dir*dist;
     d = map(p);
     if( d.x <= 0.000125){
	   if (d.y==4){
            //p.x = p.x;
            //p.z = p.z;
            dir = reflect(dir,normal(p,0.00000355));
            dist = 0.001;
            org = p;         
          }
          else
            break;  
     }
     dist += d.x;
   }
   //if(d.y == 1)
   //  p = trace(p+vec3(cosfr,sinfr,0),dir); 
   vec3 norm = normal(p);
   vec3 reflection =  dir - norm* 2 * dot(dir, norm);
   vec3 c3po = vec3(0.8,1.0,0.8);
   c3po = c3po * dot(norm, normalize(light-p));
   vec3 ambient = vec3(0.3,0.4,0.65);
   c3po = c3po + ambient + vec3(1,1,1);
   float spec = pow(max(0.0,dot(reflection,normalize(light-p))),10);
   cl.x = dist*15*norm.x;
   cl.y = dist*5*norm.y;
   cl.z = dist*15*norm.z;
   if(d.y==0 || d.y ==3  || d.y == 4)//sky
     cl= vec3(10,25,50);//(c3po+  vec3(1)*spec+ambient)*40;
   if(d.y== 5 || d.y == 1) //ball
   {
       cl = cl + (c3po+  vec3(1)*spec+ambient)*50;
   }
   if(d.y == 1) //cube
     cl.z = cl.y/2;
     
   return cl;
}

__global__ void Mandel_calc(unsigned char* image_buffer){
  unsigned short int row = (blockIdx.y * blockDim.y + threadIdx.y);  // WIDTH
  unsigned short int col = (blockIdx.x * blockDim.x + threadIdx.x);  // HEIGHT
  unsigned int idx = 3*(row * window.x + col);

  float y0 = - (float) (row -window.x/2)/(window.x/2);
  float x0 = (float) (col -window.y/2)/(window.y/2);
   
  vec3 direction = normalize(vec3(x0-sinfr,y0-1.1, 1.0));
  vec3 origin =  vec3(-0.6+3*sinfr,1.5,-1-3.0*0.5*(cosfr+1));
  vec3 cl = trace(origin,direction);
  color(cl,&image_buffer[idx]);
  }



#include "../main.cu"