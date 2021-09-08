#include <math.h>
#include <stdio.h>
#include "../shader.cu"

#define EPS 0.0005

__device__ vec2 Sky(vec3 p){
  return vec2(abs(dist(p,vec3(0))-50)-0.2,5);
}

__device__ float mandelbulb2(vec3 pos){
  float dr = 1;
  float r = 1;
  vec3 z = pos;
  float power =4 + sinfr;
  for(int i = 0; i < 15; i++){
    r = length(z);
    if(r>2.5)
      break;
    float theta = acos(z.z /r)  * power;
    float phi = atan2(z.y,z.x) * power;
    float zr = pow(r,power);
    dr = pow(r,power-1)*power*dr+1;
    
    z = vec3(sin(theta)*cos(phi),sin(phi)*cos(theta),cos(theta))*zr;
    z = z+ pos;
  
  }
  return 0.5 * log(r) * r /dr;
}

__device__ float sdf_cube(vec3 p, vec3 pos,vec3 size){
  return length(max(abs(p-pos)-size,-0.1+sinfr*0.1))-0.1+sinfr*0.1;
}

__device__ vec2 map(vec3 p){  
  // mat 4 is for mirroring surfaces
  vec2 plane = vec2(abs(p.y +3 + 0.015*sin(frame/15+length(vec2(p.x,p.z))*3)),4); 
  vec2 mirrorball = vec2(dist(p+vec3(0.7-12-3*sinfr,-3.8-sinfr,3.2+cosfr),vec3(0.5))-3,4);
  //vec2 mirrorcube = vec2(length(max(abs(p-vec3(12,3+sinfr,12))-vec3(2.5),0.0/*+sinfr*0.1275*/))-0.225,4);
  //p.x = fract(p.x/4)*4-2;
  //mat 1 is a greybluish surface
  vec2 sky = Sky(p);
  //vec2 bulb = vec2(mandelbulb2(p),7);
  vec2 ball = vec2(dist(p-vec3(0,5,0),vec3(0))-7+2*sinfr,1);
  //vec2 cube = vec2(sdf_cube(p,vec3(1,1+1+sinfr,1),vec3(5,2+sinfr,1)),2);
  p.y = fract(p.y/1.5)*1.5+0.2;
  p.z = fract(p.z/1.5)*1.5+0.2;
  p.x = fract(p.x/1.5)*1.5+0.2;
  vec2 cut = vec2(abs(p.y -1)-0.15+0.175*sinfr,1);
  cut = min(cut, vec2(abs(p.z-1)-0.15+0.175*sinfr,4));
  cut = min(cut, vec2(abs(p.x-1)-0.15+0.175*sinfr,1));
  cut.x = -cut.x;
  return min(min(min(plane,mirrorball),max(ball,cut)),sky);//min(min(min(min(min(mirrorball,ball),plane),Sky(p)),cube),mirrorcube);
}

__device__ vec3 normal(vec3 p){
  vec3 q = vec3(map(vec3(p.x + EPS, p.y, p.z)).x - map(vec3(p.x - EPS, p.y, p.z)).x,
            map(vec3(p.x, p.y + EPS, p.z )).x - map(vec3(p.x, p.y - EPS, p.z)).x,
            map(vec3(p.x, p.y, p.z + EPS)).x - map(vec3(p.x, p.y, p.z - EPS)).x);
  return normalize(q); 
} 
 
__device__ vec3 trace(vec3 org, vec3 dir){
   vec3 cl,p;
   vec3 light = vec3(sinfr*2,12.0+3*sinfr,-2.0);
   vec2 d;
   float dist = 0.0;
   float off_dist = 0;   
   for(int i = 0; i < 1024; i++)
   {
     //d.y = 0;
     p = org+dir*dist;
     d = map(p);
     if( d.x <= 0.0001){
          if (d.y==4){
            off_dist += dist;
            dir = reflect(dir,normal(p));
            dist = 0.0011;
            org = p;         
          }
          else
            break;  
     }
     else
       d.y = 0;
     dist += d.x;
   }
   dist-=5*off_dist;
    
   vec3 norm = normal(p);
   vec3 reflection =  dir - norm* 2 * dot(dir, norm);
   vec3 c3po = vec3(0.8,1.0,0.8);
   c3po = c3po * dot(norm, normalize(light-p));
   vec3 ambient = vec3(0.3,0.4,0.65);
   c3po = c3po + ambient + vec3(1,1,1);
   float spec = pow(max(0.0,dot(reflection,normalize(light-p))),10);
   cl.x = dist*15*norm.x;
   cl.y = dist*15*norm.y;
   cl.z = dist*15*norm.z;
   if(d.y==5)//sky
     cl = (c3po+ vec3(1)*spec+ambient)*35;//(c3po+  vec3(1)*spec+ambient)*40;
   else if(d.y== 2 || d.y == 1) //ball
   {
       cl = (c3po+  vec3(1)*spec+ambient)*50;
   }
   else if(d.y == 0)
     cl = vec3(0);
   if(d.y == 2) //cube
     {cl.z = cl.y/5;
      //cl.x *= cl.z;
     }
   if (d.y == 1){
     cl.z *= cl.y/4;}
   if(d.y == 10){
     cl = vec3(50,50*abs(p.x*p.y),255);
     cl.y -= off_dist;
     }
   if (d.y == 5)
   {
     cl.x /= 5;
     if (off_dist > 0)
       cl.y-= off_dist/2; 
   }
   return cl;
}

__global__ void Mandel_calc(unsigned char* image_buffer){
  unsigned short int row = (blockIdx.y * blockDim.y + threadIdx.y);  // WIDTH
  unsigned short int col = (blockIdx.x * blockDim.x + threadIdx.x);  // HEIGHT
  unsigned int idx = 3*(row * window.x + col);

  float y0 = - (float) (row -window.x/2)/(window.x/2)*2;
  float x0 = (float) (col -window.y/2)/(window.y/2)*2;
   
  vec3 direction = normalize(vec3(x0+0.5+sinfr*0.3,y0-0.6, 1.0));
  vec3 origin =  vec3(1.0-3,7.0,-12.0 );
  vec3 cl = trace(origin,direction);
  color(cl,&image_buffer[idx]);
  }



#include "../main.cu"