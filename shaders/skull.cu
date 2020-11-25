#include <math.h>
#include <stdio.h>
#include "../shader.cu"
#define EPS 0.001

__device__ float trace(vec3 org, vec3 dir);

__device__ float MyBall(vec3 p,vec3 pos, float r){
  return dist(p,pos)-r;
}

__device__ float mandelbulb(vec3 pos){
  float dr = 1;
  float r = 1;
  vec3 z = pos;
  float power =3 + cosfr;
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

__device__ float Skull(vec3 p){
  float plane = 12.5-p.z+0.3*sin(frame/15 + length(vec2(p.x,p.y)));

  float ball = abs(MyBall(p,vec3(0,2 ,0),2))-0.025;
  float eye1 = abs(MyBall(p,vec3(-0.2,2.5,-1.7),0.5+sinfr*0.06))-0.05+sinfr*0.06;
  float eye2 = abs(MyBall(p,vec3(1.0,2.5,-1.7),0.5))-0.025+cosfr*0.03;
  
  float cube = abs(length(max(abs(p-vec3(0.1,0.4+0.2*sinfr,-0.2))-vec3(1,0.4,1),0))-0.2)-0.05;
  return min(smin(max(max(ball,-eye1),-eye2),cube,0.5),plane);
}

__device__ float map(vec3 p){ 
  //return mandelbulb(p);
  return min(Skull(p),mandelbulb(p));
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

  float y0 = - (float) (row -window.x/2)/(window.x/2)*1.5;
  float x0 = (float) (col -window.y/2)/(window.y/2)*1.5;
  float r,g,b;  
  
  
  vec3 direction = normalize(vec3(x0,y0, 1));
  vec3 light = vec3(sinfr*2,5.0+3*cosfr,-2.0);
  vec3 origin =  vec3(1,2,-4);

  float dist = trace(origin,direction);
  vec3 p = origin + direction*dist;
  vec3 norm = normal(p);
  //double f = dot(direction, norm);
  vec3 reflection =  direction - norm* 2 * dot(direction, norm);
  vec3 c3po = vec3(0.8,1.0,0.8);
  c3po = c3po * dot(norm, normalize(light-p));
  float spec = pow(max(0.0,dot(reflection,normalize(light-p))),15);
  vec3 ambient = vec3(0.1,0.1,0.75);
  c3po = c3po + ambient + vec3(1,1,1);

  r = c3po.x*100;
  g = c3po.y*100;
  b = c3po.z*100;
  color(r,g,b,&image_buffer[idx]);
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
