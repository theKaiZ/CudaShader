#include <math.h>
#include <stdio.h>
#include "../shader.cu"

#define EPS 0.0002

__device__ vec2 SkyBox(vec3 p){
  vec2 plane = vec2(abs(p.y),1);
  vec2 plane2 = vec2(abs(p.y+10),1);
  vec2 plane3 = vec2(abs(p.x-10),1);
  vec2 plane4 = vec2(abs(p.x+10),1);
  vec2 plane5 = vec2(abs(p.z+10),1);
  vec2 plane6 = vec2(abs(p.z-10),1);
  return min(min(min(min(min(plane,plane2),plane3),plane4),plane5),plane6); 
}


__device__ vec2 map(vec3 p){  
  vec2 sky = SkyBox(p);
  vec2 ball = vec2(dist(p+vec3(0.7-3,-1.8,0.2+cosfr),vec3(0.5))-3,4);
  p.x = fract(p.x/4)*4-2;
 // p.y = fract(p.y/4)*4-2;
  vec2 cube = vec2(length(max(abs(p-vec3(0,1+sinfr,0))-vec3(0.5),0.0+sinfr*0.275))-0.5,1);
  //cube = abs(cube)-0.1;
  
  return min(min(cube,ball),sky);
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
   for(int i = 0; i < 1024; i++)
   {
     //d.y = 0;
     p = org+dir*dist;
     d = map(p);
     if( d.x <= 0.001){
          if (d.y==4){
            dir = reflect(dir,normal(p));
            dist = 0.01;
            org = p;         
          }
          else
            break;  
     }
     else if (dist > 200)
     {
        d.y = 0;
        break;
     }
     else
       d.y = 0;
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
   if(d.y==5)//sky
     cl = (c3po+ vec3(1)*spec+ambient)*25;//(c3po+  vec3(1)*spec+ambient)*40;
   if(d.y== 2 || d.y == 1) //ball
   {
       cl = (c3po+  vec3(1)*spec+ambient)*50;
   }
   if(d.y == 0)
     cl = vec3(0);
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
   
  vec3 direction = normalize(vec3(x0+0.5+sinfr,y0-0.6, 1.0));
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
