
#include <stdio.h>
#include <sys/time.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    
    if( col < numBColumns && row < numARows) 
    {
      //printf("row %d and col %d",row,col);
        double sum=0.0f;
        for(int k=0;k<numAColumns;k++){
            sum+= A[row*numAColumns + k] * B[k*numBColumns + col];
            }
            C[row*numBColumns+col]=sum;
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

  numBRows=atoi(argv[2]); // numAColumns
  numBColumns=atoi(argv[3]);
  
  numCRows=atoi(argv[1]); 
  numCColumns=atoi(argv[3]);

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  cudaMallocManaged(&hostA,numARows * numAColumns * sizeof(DataType));
  cudaMallocManaged(&hostB,numBRows * numBColumns * sizeof(DataType));
  cudaMallocManaged(&hostC,numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for(int i=0;i<numARows * numAColumns;i++){
      hostA[i]=RandomReal(0,1);
  }
  for(int i=0;i<numBRows * numBColumns;i++){
      hostB[i]=RandomReal(0,1);
  }

  resultRef = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));

  for(int i=0; i<numARows;i++){
    for(int j=0;j<numBColumns;j++){
       resultRef[i*numBColumns+j]=0.0;
      for(int k=0;k<numBRows;k++){
        resultRef[i*numBColumns+j]+=hostA[k+numBRows*i]*hostB[j+numBColumns*k];
      }
      
    }
  }



  //@@ Initialize the grid and block dimensions here
  dim3 Dg(numCColumns,numCRows,1);
  dim3 Db(32,32,1);

  //@@ Launch the GPU Kernel here
  gemm<<<Dg,Db>>>(hostA,hostB,hostC,numARows,numAColumns,numBRows,numBColumns);
  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here

  //@@ Insert code below to compare the output with the reference
  for(int i=0;i<numCRows * numCColumns;i++){
    if(hostC[i]!=resultRef[i] && abs(resultRef[i]-hostC[i])>0.001 ){
      printf("error %f - %f\n",hostC[i],resultRef[i]);
      return -1;
    }
  }
  printf("Correct\n");
  //@@ Free the GP U memory here
  cudaFree(hostA);
  cudaFree(hostB);
  cudaFree(hostC);

  //@@ Free the CPU memory here
  free(resultRef);


  return 0;
}
