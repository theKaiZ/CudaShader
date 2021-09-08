#include <math.h>
#include <stdio.h>
#include "../shader.cu"

#define EPS 0.00001


 

__device__ vec2 mandelbulb(vec3 pos){
  float dr = 1;
  float r = 1;
  vec3 z = pos;
  float power =5 + 2*cosfr;
  for(int i = 0; i < 10; i++){
    r = length(z);
    if(r>2.5)
      break;
    float theta = acos(z.z /r)  * power;
    float phi = atan2(z.y,z.x) * power;
    float zr = pow(r,power);
    dr = pow(r,power)*power*dr+1;
    
    z = vec3(sin(theta)*cos(phi),sin(phi)*sin(theta),cos(theta))*zr;
    z = z+ pos;
  
  }
  return vec2(0.5 * log(r) * r /dr,1);
}


__device__ vec2 Sky(vec3 p){
  return vec2(abs(dist(p,vec3(0))-15)-0.2,5);
}

__device__ vec2 map(vec3 p){  
  // mat 4 is for mirroring surfaces
  vec2 plane = vec2(abs(p.y +1 + 0.015*sin(frame/15+length(vec2(p.x,p.z))*3)),4); 
  vec2 ball =  mandelbulb(p);
  //vec2(dist(p+vec3(0.7-3,-3.8-sinfr,-3.2+cosfr),vec3(0.5))-3,4);
  p.x = fract(p.x/4)*4-2;
  //mat 1 is a greybluish surface
  vec2 cube = vec2(dist(p-vec3(0,1,0),vec3(0.5))-1,1);//vec2(length(max(abs(p-vec3(0,1+sinfr,0))-vec3(0.5),0.0+sinfr*0.275))-0.5,1);
  return min(min(min(cube,ball),plane),Sky(p));
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
   for(int i = 0; i < 512; i++)
   {
     //d.y = 0;
     p = org+dir*dist;
     d = map(p);
     if( d.x <= 0.001){
          if (d.y==4){
            dir = reflect(dir,normal(p));
            dist = 0.011;
            org = p;         
          }
          else if ( d.y == 6){
            dir = reflect(dir,normal(p,0.125));
            dist = 0.011;
            org = p;
          }
          else
            break;  
     }
     else
       d.y = 0;
     dist += d.x;
   }
    
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
   if(d.y==5)//sky
     cl = (c3po+ vec3(1)*spec+ambient)*35;//(c3po+  vec3(1)*spec+ambient)*40;
   else if(d.y== 2 || d.y == 1) //ball
   {
       cl = (c3po+  vec3(1)*spec+ambient)*50;
   }
   else if(d.y == 0)
     cl = vec3(0);
   else if(d.y == 1) //cube
     {cl.z = cl.y/5;
      //cl.x *= cl.z;
     }
   if(d.y == 10){
     cl = vec3(50,50*abs(p.x*p.y),255);
     }
   if (d.y == 5)
   {
     cl.x /= 5;
   
   }
   return cl;
}

__global__ void Mandel_calc(unsigned char* image_buffer){
  unsigned short int row = (blockIdx.y * blockDim.y + threadIdx.y);  // WIDTH
  unsigned short int col = (blockIdx.x * blockDim.x + threadIdx.x);  // HEIGHT
  unsigned int idx = 3*(row * window.x + col);

  float y0 = - (float) (row -window.x/2)/(window.x/2)/2;
  float x0 = (float) (col -window.y/2)/(window.y/2)/2;
   
  vec3 direction = normalize(vec3(x0+0.5/*+sinfr*/,y0-0.6, 1.0));
  vec3 origin =  vec3(1.0-3,2,-4.0 );
  vec3 cl = trace(origin,direction);
  color(cl,&image_buffer[idx]);
  }



#include "../main.cu"