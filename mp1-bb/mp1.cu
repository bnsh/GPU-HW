// MP 1
#include <stdio.h>
#include <assert.h>
#include "wb.h"
#include "cuPrintf.cu"

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    cuPrintf("\ti is:%d len=%d\n", i, len);
/* Binesh - 2014-08-05 - Alex, this should have been an _if_ statement
    while (i < len)
 */
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    //wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));

    printf("inputLength: %d\n", inputLength);

/* Binesh - 2014-08-05 - Alex, this was wrong... This should have been with cudaMalloc, remember?
    deviceInput1 = (float *) malloc(inputLength * sizeof(float));
    deviceInput2 = (float *) malloc(inputLength * sizeof(float));
    deviceOutput = (float *) malloc(inputLength * sizeof(float));
 */
    assert(cudaMalloc(&deviceInput1, inputLength*sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&deviceInput2, inputLength*sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&deviceOutput, inputLength*sizeof(float)) == cudaSuccess);


    //wbTime_stop(Generic, "Importing data and creating memory on host");

    //wbLog(TRACE, "The input length is ", inputLength);

    //wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here


    //wbTime_stop(GPU, "Allocating GPU memory.");

    //wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here


    //wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here

    for (int i = 0; i < inputLength; ++i) hostOutput[i] = i;

    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutput, hostOutput, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    //wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    cudaPrintfInit();
    vecAdd<<<16, 16>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaThreadSynchronize();
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < inputLength; i++) {
        printf("%.7f\n", hostOutput[i]);
    }
    //wbTime_stop(Compute, "Performing CUDA computation");
    
    //wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here

    //wbTime_stop(Copy, "Copying output memory to the CPU");

    //wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
/* Binesh - 2014-08-05 - Need to free GPU memory */
    assert(cudaFree(deviceInput1) == cudaSuccess);
    assert(cudaFree(deviceInput2) == cudaSuccess);
    assert(cudaFree(deviceOutput) == cudaSuccess);


    //wbTime_stop(GPU, "Freeing GPU Memory");

    //wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

