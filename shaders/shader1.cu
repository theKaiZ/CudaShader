#include <math.h>
#include <stdio.h>
#include "../shader.cu"
#define EPS 0.001

__device__ float map(vec3 p){ 
  vec2 pxz = vec2(p.x,p.z);    
  float plane = p.y+0.15*sin(frame/15 + length(pxz)*5)*1/p.x*1/p.z+0.5;
  //float cube = length(max(abs(vec3(p.x,p.y+ sinfr,p.z))-vec3(0.5),0.0))-0.3;
  float ball = dist(p,vec3(0)+vec3(0,1+sinfr*1.5 ,0))-1;
  return smin(ball,plane,0.9 );
}

__device__ vec3 normal(vec3 p){
  vec3 q = vec3(map(vec3(p.x + EPS, p.y, p.z)) - map(vec3(p.x - EPS, p.y, p.z)),
            map(vec3(p.x, p.y + EPS, p.z )) - map(vec3(p.x, p.y - EPS, p.z)),
            map(vec3(p.x, p.y, p.z + EPS)) - map(vec3(p.x, p.y, p.z - EPS)));
  return normalize(q); 
} 
 
__device__ float trace(vec3 org, vec3 dir){
   float dist = 0.0;   
   float d;
   for(int i = 0; i < 240; i++)
   {
     vec3 p = org+dir*dist;
     d = map(p);
     if( d <= 0.01){
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
  
  
  vec3 direction = normalize(vec3(x0,y0, 1));
  vec3 light = vec3(sinfr*2,5.0+3*sinfr,-2.0);
  vec3 origin =  vec3(1,1,-3);

  float dist = trace(origin,direction);
  vec3 p = origin + direction*dist;
  vec3 norm = normal(p);
  //double f = dot(direction, norm);
  vec3 reflection =  direction - norm* 2 * dot(direction, norm);
  vec3 c3po = vec3(0.8,1.0,0.8);
  c3po = c3po * dot(norm, normalize(light-p));
  float spec = pow(max(0.0,dot(reflection,normalize(light-p))),10);
  vec3 ambient = vec3(0.3,0.4,0.75);
  c3po = c3po + ambient + vec3(1,1,1);

  r = c3po.x*100;
  g = c3po.y*100;
  b = c3po.z*100;
  color(r,g,b,&image_buffer[idx]);
  }


#include "../main.cu"
