#include <math.h>
#include <stdio.h>
#include "../shader.cu"

#define EPS 0.0001

__device__ float map(vec3 p){ 
  vec2 pxz = {(p.x-15)+p.z,p.z-7};    
  float plane = -(p.x+p.z)*0.025+ p.y+0.25*sin(frame/15 + length(pxz)*5)+0.5;
  p.x = fract(p.x/4)*4-2; 
  p.z = fract(p.z/4)*4-2;
  float cube = length(max(abs(vec3(p.x, p.y + sinfr,p.z))-vec3(0.5),0.0))-0.5;
  cube = abs(cube)-0.1;
  float ball = dist(p+vec3(0.7+0.3*cosfr,-0.8+sinfr-cosfr*0.2,0.2*sinfr+0.7),vec3(0.5))-1;
  return smin(max(cube,-ball),plane,0.3 );
}

__device__ vec3 normal(vec3 p){
  vec3 q = vec3(map(vec3(p.x + EPS, p.y, p.z)) - map(vec3(p.x - EPS, p.y, p.z)),
            map(vec3(p.x, p.y + EPS, p.z )) - map(vec3(p.x, p.y - EPS, p.z)),
            map(vec3(p.x, p.y, p.z + EPS)) - map(vec3(p.x, p.y, p.z - EPS)));
  return normalize(q); 
} 
 
__device__ float trace(vec3 org, vec3 dir){
   float dist = 0.0;   
   for(int i = 0; i < 40; i++)
   {
     vec3 p = org+dir*dist;
     float d = map(p);
     if( d <= 0.000001){
        break;  
     }
     dist += d;
   }
   return dist;
}

__global__ void Mandel_calc(unsigned char* image_buffer){
  unsigned short int row = (blockIdx.y * blockDim.y + threadIdx.y);  // WIDTH
  unsigned short int col = (blockIdx.x * blockDim.x + threadIdx.x);  // HEIGHT
  unsigned int idx = 3*(row * window.x + col);

  float y0 = - (float) (row -window.x/2)/(window.x/2);
  float x0 = (float) (col -window.y/2)/(window.y/2);
  float r,g,b;  
  
  
  vec3 direction = normalize(vec3(x0+0.8,y0-1.5, 1.0));
  vec3 light = vec3(sinfr*2,5.0+3*sinfr,-2.0);
  vec3 origin =  vec3(1.0+3*sinfr,13.0,-5.0 );

  float dist = trace(origin,direction);
  vec3 p = origin + direction*dist;
  vec3 norm = normal(p);
  //double f = dot(direction, norm);
  vec3 reflection =  direction - norm* 2 * dot(direction, norm);
  vec3 c3po = vec3(0.8,1.0,0.8);
  c3po = c3po * dot(norm, normalize(light-p));
  float spec = pow(max(0.0,dot(reflection,normalize(light-p))),5);
  vec3 ambient = vec3(0.3,0.4,0.65);
  c3po = c3po + ambient + vec3(1,1,1);

  r = clamp(c3po.x,0,4)*51 *x0/2;
  g = clamp(c3po.y,0,4)*51*y0*2;
  b = clamp(c3po.z,0,5)*75+35*sin(x0*y0);
  color(r,g,b,&image_buffer[idx]);
  }



#include "../main.cu"