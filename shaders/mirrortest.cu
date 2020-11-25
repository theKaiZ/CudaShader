#include <math.h>
#include <stdio.h>
#include "../shader.cu"

#define EPS 0.002
#define MIN_DIST 0.01
__device__ vec2 Sky(vec3 p){
  return vec2(abs(dist(p,vec3(0))-25)-0.2,4);
}

__device__ vec2 map(vec3 p){  
  // mat 4 is for mirroring surfaces
  vec2 plane = vec2(abs(p.y +1 + 0.015*sin(frame/15+length(vec2(p.x,p.z))*3)),5); 
  vec2 mirrorball = vec2(dist(p+vec3(0.7-3-3*sinfr,-3.8-sinfr,-3.2+cosfr),vec3(0.5))-3,4);
  vec2 mirrorcube = vec2(length(max(abs(p-vec3(12,3+sinfr,12))-vec3(2.5),0.0/*+sinfr*0.1275*/))-0.225,4);
  //p.x = fract(p.x/4)*4-2;
  //mat 1 is a greybluish surface
  vec2 ball = vec2(dist(p-vec3(0,1,0),vec3(0.5))-1,15);
  vec2 cube = vec2(length(max(abs(p-vec3(-2,1+sinfr,2))-vec3(0.5),0.0+sinfr*0.275))-0.5,15);
  return min(min(min(min(min(mirrorball,ball),plane),Sky(p)),cube),mirrorcube);
}

__device__ vec3 normal(vec3 p){
  vec3 q = vec3(map(vec3(p.x + EPS, p.y, p.z)).x - map(vec3(p.x - EPS, p.y, p.z)).x,
            map(vec3(p.x, p.y + EPS, p.z )).x - map(vec3(p.x, p.y - EPS, p.z)).x,
            map(vec3(p.x, p.y, p.z + EPS)).x - map(vec3(p.x, p.y, p.z - EPS)).x);
  return normalize(q); 
} 
 
__device__ vec3 trace(vec3 org, vec3 dir){
   vec3 cl,p;
   vec3 light = vec3(sinfr*2,5.0+3*sinfr,-2.0);
   vec2 d;
   float dist = 0.0;
   float off_dist = 0;   
   for(int i = 0; i < 1024; i++)
   {
     //d.y = 0;
     p = org+dir*dist;
     d = map(p);
     if( d.x <= MIN_DIST){
          if (d.y==4){
            off_dist += dist;
            dir = reflect(dir,normal(p));
            dist = 1.1*MIN_DIST;
            org = p;         
          }
          else if(d.y== 15){//glass
            off_dist += dist;
            org = p;
            dir = dir + normal(p)+vec3(sinfr);
            dist = 1.1*MIN_DIST;
/*            while(d.y == 15)
            {
              p = org+dir*dist;
              d = map(p);
              dist += d.x;
            }
*/            
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

  float y0 = - (float) (row -window.x/2)/(window.x/2)/2;
  float x0 = (float) (col -window.y/2)/(window.y/2)/2;
   
  vec3 direction = normalize(vec3(x0+0.5+sinfr*0.3,y0-0.6, 1.0));
  vec3 origin =  vec3(1.0-3,7.0,-12.0 );
  vec3 cl = trace(origin,direction);
  color(cl,&image_buffer[idx]);
  }



extern "C" {
 unsigned char*  d_image_buffer;
 unsigned int arr_size;

__host__ void init_cuda(const int width, const int height){
  arr_size = 3 * width * height;
  cudaMallocManaged(&d_image_buffer, arr_size*sizeof(unsigned char));
  printf("Cuda Memory allocated\n");
}

__host__ void Mandel(const int width, const int height,unsigned char* image_buffer){
  dim3 block_size(16, 16);
  dim3 grid_size(width / block_size.x, height / block_size.y);
  Mandel_calc<<<grid_size, block_size>>>(d_image_buffer);
  cudaPeekAtLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(image_buffer, d_image_buffer, arr_size, cudaMemcpyDeviceToHost);
  }
__host__ void exit_cuda(){
  cudaFree(d_image_buffer);
  printf("CudaMemory free\n");
  
}

}
