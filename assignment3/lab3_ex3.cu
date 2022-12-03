
#include <stdio.h>
#include <sys/time.h>
#include <random>


#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
   
  __shared__ unsigned int shared_bins[NUM_BINS];
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  for (int j=threadIdx.x; j<num_bins; j+=blockDim.x) {
    if (j < num_bins) {
      shared_bins[j]=0;
    }
  } 
  __syncthreads();
  

  if (id < num_elements) {
        atomicAdd(&(shared_bins[input[id]]), 1);
  }
   __syncthreads(); 


  for (int j=threadIdx.x; j<num_bins; j+=blockDim.x) {
    if (j < num_bins) {
        atomicAdd(&(bins[j]), shared_bins[j]);
      }
  }
    
}


__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (id >= num_bins)
        return;

    if (bins[id] > 127)
        bins[id] = 127;

}


int main(int argc, char **argv) {
  
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
    if (argc<2){
        printf("input length invalid\n");
        return 0;
    }
    inputLength=atoi(argv[1]);
    printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
    hostInput = (unsigned int*)malloc(inputLength*sizeof(unsigned int));
    hostBins = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
    resultRef = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
   srand(clock());
    for (int i=0; i<inputLength; i++) {
        //hostInput[i]=5+i;
        hostInput[i] = rand()%NUM_BINS;
        //printf("%d ",hostInput[i]);
    }
    printf("\n\n");
    
  //@@ Insert code below to create reference result in CPU
    for (int i=0; i<NUM_BINS; i++) resultRef[i]=0;

    for (int i=0; i<inputLength; i++) {
        if(resultRef[hostInput[i]]<127){
            resultRef[hostInput[i]]++;
        }
    }
  
  //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
    dim3 Dg(1024,1,1);
    dim3 Db(1024,1,1);

  //@@ Launch the GPU Kernel here
    histogram_kernel<<<Dg, Db>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
    cudaDeviceSynchronize();
  //@@ Initialize the second grid and block dimensions here
    dim3 Dg1(1024,1,1);
    dim3 Db1(1024,1,1);

  //@@ Launch the second GPU Kernel here
    convert_kernel<<<Dg1, Db1>>>(deviceBins, NUM_BINS);
    cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Host\n");
    for(int i=0;i<20;i++)
        if(resultRef[i]!=0)
            printf("value %d in bin %d\n",resultRef[i],i);

    printf("Device\n");
    for(int i=0;i<20;i++)
        if(hostBins[i]!=0)
            printf("value %d in bin %d\n",hostBins[i],i);
  //@@ Insert code below to compare the output with the reference
    for(int i=0;i<NUM_BINS;i++)
        if(hostBins[i]!=resultRef[i]){
            printf("error\n data: %d:%d - %d:%d\n",i,hostBins[i],i,resultRef[i]);
            return 0;
        }
    printf("correct");

  //@@ Free the GPU memory here
    
    cudaFree(deviceInput);
    cudaFree(deviceBins);

  //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

  return 0;
}
