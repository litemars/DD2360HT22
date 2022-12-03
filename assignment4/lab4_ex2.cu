
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define DataType double


__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(id < len) {
      out[id] = in1[id] + in2[id];
      }
}

//@@ Insert code to implement timer start
double startTimer() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
//@@ Insert code to implement timer stop
double stopTimer(double startime) {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return (((double)tp.tv_sec + (double)tp.tv_usec*1.e-6) - startime);
}

double RandomReal(double low, double high)
{
  double d;

  d = (double) rand() / ((double) RAND_MAX + 1);
  return (low + d * (high - low));
}

void printArray(double *in, int len){
  for(int i=0;i<len;i++){
    printf("%f ",in[i]);
  }
  printf("\n\n");
}

int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  if (argc<2){
    printf("input length invalid\n");
    return 0;
  }
  inputLength=atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType*)malloc(inputLength*sizeof(DataType));
  hostInput2 = (DataType*)malloc(inputLength*sizeof(DataType));
  hostOutput = (DataType*)malloc(inputLength*sizeof(DataType));
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU

  for(int i=0;i<inputLength;i++){
      hostInput1[i]=RandomReal(0,1);
      hostInput2[i]=RandomReal(0,1);
  }

  resultRef = (DataType*)malloc(inputLength*sizeof(DataType));
  double start_time=startTimer();
  for(int i=0;i<inputLength;i++){
    resultRef[i]=hostInput1[i]+hostInput2[i];
  }
  double stopCPU=stopTimer(start_time);
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength*sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength*sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength*sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  double start_mem=startTimer();
  cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double stop_mem=stopTimer(start_mem);
  //@@ Initialize the 1D grid and block dimensions here
  dim3 Dg(1024,1,1);
  dim3 Db(1024,1,1);

  //@@ Launch the GPU Kernel here
  double start_time_gpu=startTimer();
  vecAdd<<<Dg,Db>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);
  cudaDeviceSynchronize();
  double stopGPU=stopTimer(start_time_gpu);

  double start_mem_2=startTimer();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double stop_mem_2=stopTimer(start_mem_2);

  //@@ Insert code below to compare the output with the reference
  for(int i=0;i<inputLength;i++){
    if(resultRef[i] != hostOutput[i] && abs(resultRef[i]-hostOutput[i])>0.001 ){
        printf("Error counting numbers: %f",abs(resultRef[i]-hostOutput[i]) );
        return 0;
    }
  }
  printf("sum verified: Correct!\n");
  printf("Time Host->Device: %f - Time Device->Host: %f\n",stop_mem,stop_mem_2);
  printf("CPU time: %f - GPU time: %f\n",stopCPU,stopGPU);
  
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  //@@ Free the CPU memory here

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
