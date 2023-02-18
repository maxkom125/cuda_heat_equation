/*
CUDA Heat equation
See the description of the approach in the report file
Run:
g++ -c seqheateq.cpp
nvcc -c heat_equation.cu
nvcc heat_equation.o seqheateq.o -o output
./output
*/
#include <iostream>
#include <stdio.h> 
#include <cstring> //memset
#include <fstream> //file operation
#include <string>
#include <cmath>

#include <cuda_runtime.h>
#include "seqheateq.h"


void checkCudaSuccess(cudaError_t err, std::string funcName) {
    if (err != cudaSuccess) {
        if (funcName == std::string("cudaMemcpy")) {
            std::cout << "Error copying data in cudaMemcpy: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        else if (funcName == std::string("cudaMalloc")) {
            std::cout << "Error allocating memory on device: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

void printArr(float* arr, long len, std::string filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file.\n";
        return;
    }

    for(long i = 0; i < len; i++)
        outfile << arr[i] << "\n";
    outfile.close();
}

__device__ __forceinline__ float GetU(float* previousRow, long i) {
    float integrStep = 0.3; ///////////////HEAT FUNCTION HYPERPARAMETER//////////////
    //return previousRow[i] + previousRow[i + 1];
    return previousRow[i] + integrStep * (previousRow[i + 1] - 2 * previousRow[i] + previousRow[i - 1]);
}

__global__ void processTopTriangle(float* bound, float* innerBound, long coordSteps, long timeSteps) {
    /* Triangle under consideration:    
      /\
     /\/\
    /\/\/\
    ______
    Explonation:
    Use previous layer to calculate next.
    After calculating each layer, the total number of values that 
    we can calculate in the next layer is reduced by 2.
                   {U(1.a,1.b,1.c)}                <- 0 (last layer to proceed)
                  /                \
          {U(a,b,c)}  {U(b,c,d)}  {U(c,d,e)}       <- 1
            /         \       /          \   
        {a}    {b}       {c}       {d}     {e}
        /        \       / \       /         \  
      . . . . . . . . . . . . . . . . . . . . .  
      _________________________________________

    We need to store boundary and inner boundary for
    folowing processBotTriangle() function : 
      //\\   ib[n] b[n+1]
     //  \\       ib[n+1] b[n+2]
    //    \\             ib[n+2] b[n+3]
                        
    ib[n]  save to innerBound array
    b[n+1] save to bound array
    */
    if (timeSteps > blockDim.x / 2 - 1)
        timeSteps = blockDim.x / 2 - 1;
    // Calculate a part of bound array that coresponds to the triangle (of current block)
    long blockStartIdx = blockDim.x * blockIdx.x; //+bias // global pointer in bound array
    long blockEndIdx = blockDim.x * (blockIdx.x + 1); //+bias

    if (blockEndIdx - blockDim.x / 2 >= coordSteps) //TODO: +- 1?
        return;
    // Check right out of bound
    long rightBound, leftBound, rightStep, leftStep;
    extern __shared__ float currentLayer[]; //shared dynamic array
    float* previousLayer = currentLayer + blockDim.x; //second half of shared dynamic array
    float* innerLayer = previousLayer + blockDim.x; 
    //float* testLayer = innerLayer + blockDim.x; //test part of shared dynamic array

    long globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    long localIdx = threadIdx.x;
    float threadValue = bound[globalIdx];
    currentLayer[localIdx]  = threadValue;
    previousLayer[localIdx] = threadValue;
    innerLayer[localIdx]    = threadValue;
    __syncthreads();

    if (blockEndIdx >= coordSteps) {
        rightBound = blockDim.x - (blockEndIdx - coordSteps) - 2; //TODO: +1 ? check! 32 - 26 - 1 = 5  
        rightStep = 0;
    }
    else {
        rightBound = blockDim.x - 2;
        rightStep = -1;
    }
    if (blockStartIdx > 0) {
        leftBound = 1;
        leftStep = 1;
    }
    else {
        leftBound = 1;
        leftStep = 0;
    }
  
    for (long layer = 0; layer < timeSteps; layer++) {
        //check out of bounds
        if (leftBound <= localIdx && rightBound >= localIdx) {
            currentLayer[localIdx] = GetU(previousLayer, localIdx);
            if (leftBound == localIdx || rightBound == localIdx) // the last GetU calculation on this idx
                innerLayer[localIdx] = previousLayer[localIdx];
        } // ????? 
        __syncthreads();
        previousLayer[localIdx] = currentLayer[localIdx];
        __syncthreads();
        rightBound += rightStep;
        leftBound += leftStep;
    }
    bound[globalIdx] = currentLayer[localIdx];
    innerBound[globalIdx] = innerLayer[localIdx];
}

__global__ void processBotTriangle(float* bound, float* innerBound, long coordSteps, long timeSteps) {
    /* Triangle under consideration: ______
                                     \/\/\/
    multiple layers                   \/\/
                                       \/   
    each layer corresponds to one time step and part of the array of coordinate steps
    -> a[0] a[1] a[2] ...a[i].. - values
    to calculate next layer value we need previous layer a values:
    a_new[i] = U(a[i-1], a[i], a[i+1])
    boundaries:
    we need to know boundary(float* bound): \    / 
                                             \  / 
                                              \/ 
    and inner boundary(float* innerBound)  \\    //                 ib[n] b[n+1]
                                            \\  //          ib[n-1] b[n]
                                             \\//   ib[n-2] b[n-1]
               to calculate           a[0]
                                    /   |   \
                we need         ib[n] b[n+1] a[1]
                        
    ib[n] stored in innerBound array
    b[n+1] stored in bound array
    because to calculate a[1] in the previous step we need to know
    ib[n+1] from innerBound array and b[n+2] from bound array
    */
    //The number of layers is limited by the width of the triangle
    if (timeSteps > blockDim.x / 2 - 1)
        timeSteps = blockDim.x / 2 - 1;
    //Calculate a part of bound array that coresponds to the triangle (of current block)
    long bias = blockDim.x / 2;
    long blockStartIdx = blockDim.x * blockIdx.x + bias; //global pointer in the bound array
    long blockEndIdx = blockDim.x * (blockIdx.x + 1) + bias;
    
    if (blockStartIdx + blockDim.x / 2 >= coordSteps) //if true: this part is already calculated
        return;

    extern __shared__ float currentLayer[]; //shared dynamic array
    float* previousLayer = currentLayer + blockDim.x; //another part of the shared memory
    float* boundLayer = previousLayer + blockDim.x;   //another part of the shared memory

    //Determine the appropriate global array index for the thread
    long globalIdx = blockIdx.x * blockDim.x + threadIdx.x + bias;
    long localIdx = threadIdx.x;

    //Fill shared memory arrays
    float threadValue = bound[globalIdx];
    currentLayer[localIdx]  = threadValue;
    previousLayer[localIdx] = innerBound[globalIdx];
    boundLayer[localIdx]    = threadValue;
    __syncthreads();

    //Calculate block boundaries
    long rightBound, leftBound, rightStep, leftStep;
    if (blockEndIdx >= coordSteps) {
        rightBound = blockDim.x - (blockEndIdx - coordSteps) - 2; //TODO: +1 ? check! 32 - 26 - 1 = 5  
        rightStep = 0;
    }
    else {
        rightBound = blockDim.x / 2;
        rightStep = 1;
    }
    if (blockStartIdx > 0) {
        leftBound  = blockDim.x / 2 - 1;
        leftStep = -1;
    }
    else {
        leftBound = 1;
        leftStep = 0;
    }
  
    //Calculate layers inside a a triangle
    for (long layer = 0; layer < timeSteps; layer++) {
        //Check the boundaries and calculate value: 
        //U(a[i-1], a[i], a[i+1])
        //   /     |      \ 
        // a[i-1] a[i] a[i+1]
        if (leftBound <= localIdx && rightBound >= localIdx) 
            currentLayer[localIdx] = GetU(previousLayer, localIdx);
        __syncthreads(); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //assign calculated value a[i] = U(a[i-1], a[i], a[i+1])
        if (leftBound <= localIdx && rightBound >= localIdx)
            previousLayer[localIdx] = currentLayer[localIdx];
        else if (localIdx == leftBound + leftStep || localIdx == rightBound + rightStep) {//will be used in the next step
            previousLayer[localIdx] = boundLayer[localIdx]; //currentLayer[localIdx] = bound[globalIdx];
        }

        __syncthreads();
        //                                  \  /
        // next layer will be larger \/ ->   \/
        rightBound += rightStep;
        leftBound  += leftStep;
    }
    bound[globalIdx] = currentLayer[localIdx];
}

long getNumBlocks(long coordSteps, long blocksize) {
    return (coordSteps + blocksize - 1) / blocksize; //round up the division of two integers
}

void heatEquation(float* initialState, long blocksize, long timeSteps, long coordSteps) { //GPU heat equation
    //Define number of blocks
    long numBlocks = getNumBlocks(coordSteps, blocksize);
    dim3 threadsPerBlock = blocksize;

    float* bound; //device
    float* innerBound; //device

    //Allocate memory on device
    cudaError_t err;
    long boundMem = coordSteps * sizeof(float);

    err = cudaMalloc((void**)&bound, boundMem);
    checkCudaSuccess(err, "cudaMalloc");
    err = cudaMalloc((void**)&innerBound, boundMem);
    checkCudaSuccess(err, "cudaMalloc");

    //Fill device structures with initial values
    err = cudaMemcpy(bound, initialState, coordSteps * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaSuccess(err, "cudaMemcpy");
    err = cudaMemcpy(innerBound, initialState, coordSteps * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaSuccess(err, "cudaMemcpy");
    // Wait for cudaMemcpy completion
	cudaDeviceSynchronize();

    long time = 0;
    while(time < timeSteps) {
        /*
        1.1/\ 1.2/\ 1.3/\1.4    //dublicate to reach timeSteps
          /\/\  /\/\  /\/\               
         /0.1 \/0.2 \/0.3 \     //0.X TopTriangles; 1.X BotTrianges
        */
        err = cudaMemcpy(innerBound, initialState, coordSteps * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaSuccess(err, "cudaMemcpy");

        processTopTriangle<<<numBlocks, threadsPerBlock, 3*blocksize*sizeof(float)>>>(bound, innerBound, coordSteps, timeSteps - time);//0.X
        err = cudaMemcpy(initialState, bound, coordSteps * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaSuccess(err, "cudaMemcpy");

        // Wait for 0.X completion
        cudaDeviceSynchronize();

        processBotTriangle<<<numBlocks, threadsPerBlock, 3*blocksize*sizeof(float)>>>(bound, innerBound, coordSteps, timeSteps - time);//1.X

        err = cudaMemcpy(initialState, bound, coordSteps * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaSuccess(err, "cudaMemcpy");

        // Wait for layer completion
        cudaDeviceSynchronize();

        time += blocksize / 2 - 1;
    }
    // Free the GPU array
    cudaFree(bound);    
    cudaFree(innerBound);    
    }


int main(int argc, char* argv[]) {
    std::cout << "Start" << std::endl;

    //-----------------------HYPERPARAMETERS-----------------------
    float leftValue = 3;
    float rightValue = 1;

    //The maximum number of threads per block is limited, typically 1024 on current GPUs.
	const long blocksize = 64; // must be divisible by 2 !!!
    //-------------------------------------------------------------
    if (blocksize % 2 != 0) {
        std::cout << "Incorrect parameter: Blocksize must be divisible by 2 " << std::endl;
        return EXIT_FAILURE;
    }

    cudaEvent_t start, stop;
    float elapsedTime1, elapsedTime2;

    // -----------------------CALL MAIN FUNCTION with different inputs-----------------------
    for (long timeSteps = 150; timeSteps < 152; timeSteps++) {/////////////////////////////HYPERPARAMETERS////
        for (long coordSteps = 10000001; coordSteps < 100000001; coordSteps+=10000023) { //HYPERPARAMETERS////
            // Create host data struture
            float* initialState = new float[coordSteps](); //host (filled with zeros)

            //Functions to detect execution time
            cudaEventCreate(&start);
            cudaEventRecord(start, 0);

            //Heat on the borders
            initialState[0] = leftValue;
            initialState[coordSteps - 1] = rightValue;

            heatEquation(initialState, blocksize, timeSteps, coordSteps); //MAIN FUNCTION

            //Functions to detect execution time
            cudaEventCreate(&stop);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime1, start, stop);

            std::string outFile = "Heat_out.txt";
            printArr(initialState, coordSteps, outFile);

            //Functions to detect execution time
            cudaEventCreate(&start);
            cudaEventRecord(start, 0);

            //Heat Equation calculations and test on CPU 
            SequentialHeatEquation(coordSteps, timeSteps, leftValue, rightValue, outFile); 

            //Stop timer for function 2
            cudaEventCreate(&stop);
            cudaEventRecord(stop, 0);

            //Calculate elapsed time for function 2 and speedup
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime2, start, stop);
            std::cout << "Speedup: " << elapsedTime2 / elapsedTime1 << " GPU time(ms): " << elapsedTime1 << std::endl;

            //Free memory
            delete[] initialState;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	return EXIT_SUCCESS;
}
