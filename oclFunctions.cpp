

#include "helperFunctions.cpp"
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>
#if defined(__APPLE__) || defined(APPLE)
    #include <OpenCL/OpenCL.h>
#else
    #include <CL/opencl.h>
#endif

#define BLOCK_SIZE 16

cl_context ctx;
cl_kernel kernel;
cl_command_queue queue;

const char* get_kernel_source();


char * load_program_source(const char *filename)
{

    struct stat statbuf;
    FILE *fh;
    char *source;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}


float* ocl_matrix_multiply(float A[], float B[], int ah, int ud, int bw)
{
    float* C = new float[ah*bw];
    cl_platform_id plat = NULL;
    cl_device_id *devices = NULL;
    cl_device_id device = NULL;
    cl_uint dev_count = 0;
    cl_int err_num = CL_SUCCESS;

    // Plat setup
    err_num = clGetPlatformIDs(1, &plat, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "Plat fail" << endl;
        exit(err_num);
    }

    // Dev setup
    err_num = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);
    devices = (cl_device_id *)malloc(dev_count * sizeof(cl_device_id));
    err_num = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, dev_count, devices, NULL);
    device = devices[0];
    if (err_num != CL_SUCCESS)
    {
        cout << "Dev fail" << endl;
        exit(err_num);
    }

    // Context setup
    // 1 == my device count (arbitrary)
    ctx = clCreateContext(0, 1, &device, NULL, NULL, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "Ctx fail" << endl;
        exit(err_num);
    }

    // queue setup
    queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "queue fail" << endl;
        exit(err_num);
    }

    // prog setup
    size_t program_length;
    const char *source = load_program_source("oclKernels.cl");
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&source, NULL, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "compile fail" << endl;
        exit(err_num);
    }

    // build program

    err_num = clBuildProgram(prog, 1, &device, "-cl-mad-enable", NULL, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "build fail" << endl;
        exit(err_num);
    }

    // kernel setup
    kernel = clCreateKernel(prog, "matMul", &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make kernel fail" << endl;
        exit(err_num);
    }

    // array rounding
    int pad_to = 64;

    int bw_round = round_up(bw, pad_to);
    int ud_round = round_up(ud, pad_to);
    int ah_round = round_up(ah, pad_to);

    float* round_A = pad(A, ah, ud, pad_to);
    float* round_B = pad(B, ud, bw, pad_to);
    float* round_C = pad(C, ah, bw, pad_to);

    // work dim setup
    size_t global_work_size[] = {ah_round, bw_round};
    size_t local_work_size[] = {BLOCK_SIZE, BLOCK_SIZE};

    // buffer setup
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, ah_round*ud_round*sizeof(float), round_A, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bw_round*ud_round*sizeof(float), round_B, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, ah_round*bw_round*sizeof(float), round_C, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }

    cl_int temp_ah = ah;
    cl_int temp_bw = bw;
    cl_int temp_bw_round = bw_round;
    cl_int temp_ud = ud;
    cl_int temp_ud_round = ud_round;

    // set kernel args
    err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &d_A);
    err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &d_B);
    err_num |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &d_C);
    err_num |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *) &temp_ud);
    err_num |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void *) &temp_ud_round);
    err_num |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void *) &temp_bw_round);
    err_num |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void *) &temp_bw);
    err_num |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void *) &temp_ah);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel arg set fail" << endl;
        exit(err_num);
    }

    // launch kernel
    timeval t1, t2;
    double elapsedTime;
    gettimeofday(&t1, NULL);

    err_num = clEnqueueNDRangeKernel(queue, kernel, 2, 0, global_work_size, local_work_size, 0, NULL, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel launch fail" << endl;
        exit(err_num);
    }
    clFinish(queue);
    gettimeofday(&t2, NULL);

    elapsedTime = (t2.tv_sec -t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    cout << "Multiplication: " << elapsedTime << " ms.\n";

    // get results
    err_num = clEnqueueReadBuffer(queue, d_C, CL_FALSE, 0, ah_round*bw_round*sizeof(float), round_C, 0, NULL, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "read fail" << endl;
        exit(err_num);
    }
    clFinish(queue);

    // cleanup
    err_num |= clReleaseMemObject(d_A);
    err_num |= clReleaseMemObject(d_B);
    err_num |= clReleaseMemObject(d_C);
    err_num |= clReleaseKernel(kernel);
    err_num |= clReleaseCommandQueue(queue);
    err_num |= clReleaseProgram(prog);
    err_num |= clReleaseContext(ctx);
    if (err_num != CL_SUCCESS)
    {
        cout << "free fail" << endl;
        exit(err_num);
    }

    // return results
    for (int i = 0; i < ah; i++)
    {
        for (int j = 0; j < bw; j++)
        {
            C[i*bw + j] = round_C[i*bw_round + j];
        }
    }
    /*
    */

    return C;
}

const char* get_kernel_source()
{
    return " "\
    " #define BLOCK_SIZE 16 " \
    "__kernel void matMul( " \
    "                __global float* A, " \
    "                __global float* B, " \
    "                __global float* C, " \
    "                int a_row_len, " \
    "                int a_round_row_len, " \
    "                int c_round_row_len, " \
    "                int row_bound, " \
    "                int col_bound) " \
    "{ " \
    "    int wA = a_round_row_len; " \
    "    int wB = c_round_row_len; " \
    " " \
    "    // Block index " \
    "    int bx = get_group_id(0); " \
    "    int by = get_group_id(1); " \
    " " \
    "    // Thread index " \
    "    int tx = get_local_id(0); " \
    "    int ty = get_local_id(1); " \
    " " \
    "    // Index of the first sub-matrix of A processed " \
    "    // by the block " \
    "    int aBegin = wA * BLOCK_SIZE * by; " \
    " " \
    "    // Index of the last sub-matrix of A processed " \
    "    // by the block " \
    "    int aEnd   = aBegin + wA - 1; " \
    " " \
    "    // Step size used to iterate through the " \
    "    // sub-matrices of A " \
    "    int aStep  = BLOCK_SIZE; " \
    " " \
    "    // Index of the first sub-matrix of B processed " \
    "    // by the block " \
    "    int bBegin = BLOCK_SIZE * bx; " \
    " " \
    "    // Step size used to iterate through the " \
    "    // sub-matrices of B " \
    "    int bStep  = BLOCK_SIZE * wB; " \
    " " \
    "    float Csub = 0.0f; " \
    " " \
    "    // Loop over all the sub-matrices of A and B " \
    "    // required to compute the block sub-matrix " \
    "    for (int a = aBegin, b = bBegin; " \
    "             a <= aEnd; " \
    "             a += aStep, b += bStep) " \
    "    { " \
    " " \
    "        // Declaration of the local memory array As " \
    "        // used to store the sub-matrix of A " \
    "        __local float As[BLOCK_SIZE][BLOCK_SIZE]; " \
    " " \
    "        // Declaration of the local memory array Bs " \
    "        // used to store the sub-matrix of B " \
    "        __local float Bs[BLOCK_SIZE][BLOCK_SIZE]; " \
    " " \
    "        // Load the matrices from global memory " \
    "        // to local memory; each thread loads " \
    "        // one element of each matrix " \
    "        As[ty][tx] = A[a + wA * ty + tx]; " \
    "        Bs[ty][tx] = B[b + wB * ty + tx]; " \
    " " \
    "        // Synchronize to make sure the matrices " \
    "        // are loaded " \
    "        barrier(CLK_LOCAL_MEM_FENCE); " \
    " " \
    "        // Multiply the two matrices together; " \
    "        // each thread computes one element " \
    "        // of the block sub-matrix " \
    "        for (int k = 0; k < BLOCK_SIZE; ++k) " \
    "            Csub += As[ty][k] * Bs[k][tx]; " \
    " " \
    "        // Synchronize to make sure that the preceding " \
    "        // computation is done before loading two new " \
    "        // sub-matrices of A and B in the next iteration " \
    "        barrier(CLK_LOCAL_MEM_FENCE); " \
    " " \
    "    } " \
    " " \
    "    // Write the block sub-matrix to device memory; " \
    "    // each thread writes one element " \
    "    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx; " \
    "    C[c + wB * ty + tx] = Csub; " \
    "} ";
 }
