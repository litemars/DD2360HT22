
#include <stdio.h>
#include <sys/time.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if( id < numBColumns*numARows) 
    {
        C[id]=0;
        for(int k=0;k<numARows;k++){

            printf("id: %d : %d - %d\n",id,k+id-(id%numARows),id%numBColumns+k*(numBColumns));
            C[id]+=A[k+id-(id%numBColumns)]*B[id%numBColumns+k*(numBColumns)];
            }
          }
}

void printArray(double *in, int row, int col){
  printf("\ndims: %d-%d \n",row,col);
  for(int i=0;i<col*row;i=i+col){
    for(int j=0;j<col;j++){
      printf("%f ",in[i+j]);
    }
    printf("\n");
  }
  printf("\n\n");

}

double RandomReal(double low, double high)
{
  double d;

  d = (double) rand() / ((double) RAND_MAX + 1);
  return (low + d * (high - low));
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
   if (argc<4){
    printf("input length invalid\n");
    printf("./a.out numARows numAColumns numBColumns\n");
    return 0;
  }
  
  numARows=atoi(argv[1]);
  numAColumns=atoi(argv[2]);
  numBRows=atoi(argv[2]);
  numBColumns=atoi(argv[3]);
  numCRows=atoi(argv[1]);
  numCColumns=atoi(argv[3]);

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType*)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType*)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for(int i=0;i<numARows * numAColumns;i++){
      hostA[i]=RandomReal(0,1);
  }
  for(int i=0;i<numBRows * numBColumns;i++){
      hostB[i]=RandomReal(0,1);
  }
  printf("Matrix A:\n");
  printArray(hostA,numARows,numAColumns);

  printf("Matrix B:\n");
  printArray(hostB,numBRows,numBColumns);

  resultRef = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));

  //A0*B(0+0row)+A1*B(0+1row)+A2*B(0+2row)=C0
  //A0*B(1+0row)+A1*B(1+1row)+A2*B(1+2row)=C1
  //
  //A
  for(int i=0; i<numARows;i++){
    
    for(int j=0;j<numBColumns;j++){
      //printf("i: %d and j: %d\n",i,j);
      for(int k=0;k<numARows;k++){
        //printf("%d - %d\n",k+numBColumns*i,j+numBColumns*k);
        resultRef[j+numBColumns*i]+=hostA[k+numBColumns*i]*hostB[j+numBColumns*k];
      }
      
    }
  }
  

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns*sizeof(DataType), cudaMemcpyHostToDevice);


  //@@ Initialize the grid and block dimensions here
  dim3 Dg(1024,1,1);
  dim3 Db(1024,1,1);

  //@@ Launch the GPU Kernel here
  gemm<<<Dg,Db>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns);
  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC,  numCRows * numCColumns *sizeof(DataType), cudaMemcpyDeviceToHost);


  printf("Matrix C:\n");
  printArray(resultRef,numCRows,numCColumns);

  printf("Device C:\n");
  printArray(hostC,numCRows,numCColumns);
  //@@ Insert code below to compare the output with the reference
  for(int i=0;i<numCRows * numCColumns;i++){
    if(hostC[i]!=resultRef[i] && abs(resultRef[i]-hostC[i])>0.001 ){
      printf("error %f - %f\n",hostC[i],resultRef[i]);
      return 0;
    }
  }
  printf("Correct\n");
  //@@ Free the GP U memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
