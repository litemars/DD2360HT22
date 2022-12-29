
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cuda.h>

#define DataType double
#define DEBUG 1

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < len)
      out[id] = in1[id] + in2[id];
      
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
  
  int inputLength, nStream;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  if (argc<3){
    printf("input length invalid - ./a.out inputLength nStreams\n");
    return 0;
  }
  inputLength=atoi(argv[1]);
  nStream=atoi(argv[2]);
  printf("The input length is %d and the streamSize: %d\n", inputLength,nStream);


  int StreamSize  = inputLength / nStream;
  
  int StreamByte = StreamSize*sizeof(DataType);

  #ifdef DEBUG
    printf("StreamSize: %d\n",StreamSize);
  #endif

  cudaStream_t stream[nStream];
  for(int i=0;i<nStream;++i)
    cudaStreamCreate(&stream[i]);
  
  //@@ Insert code below to allocate Host memory for input and output
  cudaMallocHost(&hostInput1,inputLength*sizeof(DataType));
  cudaMallocHost(&hostInput2,inputLength*sizeof(DataType));
  cudaMallocHost(&hostOutput,inputLength*sizeof(DataType));
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU

  for(int i=0;i<inputLength;i++){
      hostInput1[i]=RandomReal(0,1);
      hostInput2[i]=RandomReal(0,1);
  }

  resultRef = (DataType*)malloc(inputLength*sizeof(DataType));
  //double start_time=startTimer();
  for(int i=0;i<inputLength;i++){
    resultRef[i]=hostInput1[i]+hostInput2[i];
  }
  //double stopCPU=stopTimer(start_time);
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength*sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength*sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength*sizeof(DataType));

  dim3 Dg(1024/nStream,1,1);
  dim3 Db(1024,1,1);
  
  //@@ Insert code to below to Copy memory to the GPU here
  //double start_mem=startTimer();
  //printf("Host1\n");
  //printArray(hostInput1,inputLength);

  for(int i=0;i<nStream;++i){
    
    int offset=i*StreamSize;

    //printf("offeset: %d-val %f - streamByte %d\n ",offset,hostInput1[offset],StreamByte);
    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], StreamByte, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], StreamByte, cudaMemcpyHostToDevice,stream[i]);
    vecAdd<<<Dg,Db,0,stream[i]>>>(&deviceInput1[offset],&deviceInput2[offset],&deviceOutput[offset],StreamSize);
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], StreamByte, cudaMemcpyDeviceToHost,stream[i]);
  }
  //double stop_mem=stopTimer(start_mem);
  //@@ Initialize the 1D grid and block dimensions here
  for(int i=0; i<nStream; i++)
		cudaStreamSynchronize(stream[i]);

  //@@ Launch the GPU Kernel here
  //double start_time_gpu=startTimer();

  //double stopGPU=stopTimer(start_time_gpu);

  //double start_mem_2=startTimer();
  //@@ Copy the GPU memory back to the CPU here

  //double stop_mem_2=stopTimer(start_mem_2);
  //printf("Array from device:\n");
  //printArray(hostOutput,inputLength);
  //printf("Array from host:\n");
  //printArray(resultRef,inputLength);
  
  //@@ Insert code below to compare the output with the reference
  for(int i=0;i<inputLength;i++){
    if(resultRef[i] != hostOutput[i] && abs(resultRef[i]-hostOutput[i])>0.001 ){
        printf("Error counting numbers: %f",abs(resultRef[i]-hostOutput[i]) );
        return 0;
    }
  }
  printf("sum verified: Correct!\n");
  //printf("Time Host->Device: %f - Time Device->Host: %f\n",stop_mem,stop_mem_2);
  //printf("CPU time: %f - GPU time: %f\n",stopCPU,stopGPU);
  
  for(int i=0;i<nStream;++i)
    cudaStreamDestroy(stream[i]);

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  //@@ Free the CPU memory here

  cudaFree(hostInput1);
  cudaFree(hostInput2);
  cudaFree(hostOutput);

  return 0;
}
