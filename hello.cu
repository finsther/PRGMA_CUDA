#include <stdio.h>

__global__ void print_hello_from_gpu (void) {
  printf("This hello message if from GPU, thread: %d!\n", threadIdx.x);
}

void main() {
  printf("This hello message if from CPU!\n");

  print_hello_from_gpu <<<1,10>>>();

  cudaDeviceSynchronize();

  return 0;
}
