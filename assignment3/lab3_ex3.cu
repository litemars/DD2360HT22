
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics


}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127


}
void printH(unsigned int *bins, int len){
    printf("\n\n");
    printf("histogram\n");
    int safe_count;
    int max=127;
    for(int i=0;i<len;i++)
        printf("%d ",bins[i]);
    printf("\n");
    printf("a");    
    for(int j=0;j<max;j++){
        safe_count=0;
        for(int i=0;i<len;i++){
            if(bins[i]>=max-j){
                printf("x");
            }else{
                safe_count++;
                printf(" ");
            }
        }
        if(safe_count!=len)
            printf("\n");
    }
    printf("\n\n");
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
    hostInput = (unsigned int*)malloc(sizeof(NUM_BINS));
    hostBins = (unsigned int*)malloc(sizeof(NUM_BINS));
    resultRef = (unsigned int*)malloc(sizeof(NUM_BINS));
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
   srand(clock());
    for (int i=0; i<inputLength; i++) {
        //hostInput[i]=5+i;
        hostInput[i] = int((float)rand()*(NUM_BINS-1)/float(RAND_MAX));
        //printf("%d ",hostInput[i]);
    }
    
  //@@ Insert code below to create reference result in CPU
    for (int i=0; i<inputLength; i++) {
        if(hostInput[i]<127){
            resultRef[i]=hostInput[i];
        }else{
            resultRef[i]=127;
        }
        //printf("%d ",resultRef[i]);
    }
    printH(resultRef,inputLength);
  //@@ Insert code below to allocate GPU memory here


  //@@ Insert code to Copy memory to the GPU here


  //@@ Insert code to initialize GPU results


  //@@ Initialize the grid and block dimensions here


  //@@ Launch the GPU Kernel here


  //@@ Initialize the second grid and block dimensions here


  //@@ Launch the second GPU Kernel here


  //@@ Copy the GPU memory back to the CPU here


  //@@ Insert code below to compare the output with the reference


  //@@ Free the GPU memory here


  //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

  return 0;
}
