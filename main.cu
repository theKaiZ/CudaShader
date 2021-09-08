


extern "C" {
 unsigned char*  d_image_buffer;
 unsigned int arr_size;
 int WIDTH;
 int HEIGHT;

__host__ void init_cuda(const int width, const int height){
  arr_size = 3 * width * height;
  WIDTH = width;
  HEIGHT = height;
  cudaMallocManaged(&d_image_buffer, arr_size*sizeof(unsigned char));
  printf("Cuda Memory allocated\n");
}

__host__ void Mandel(unsigned char* image_buffer){
  dim3 block_size(16, 16);
  dim3 grid_size(WIDTH / block_size.x, HEIGHT / block_size.y);
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
