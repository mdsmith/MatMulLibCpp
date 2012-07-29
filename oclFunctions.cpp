

#include "helperFunctions.cpp"
#if defined(__APPLE__) || defined(APPLE)
    #include <OpenCL/OpenCL.h>
#else
    #include <CL/opencl.h>
#endif

#define BLOCK_SIZE 16

cl_context ctx;
cl_kernel kernel;
cl_command_queue queue;

float* ocl_matrix_multiply(float A[], float B[], int ah, int ud, int bw)
{
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
    ctx = clCreateContext(0, 1, device, NULL, NULL, &err_num);
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
    char *source = oclLoadProgSource("oclKernels.cl", "", &program_length);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&source, &program_length, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "compile fail" << endl;
        exit(err_num);
    }
    free(source);

    // kernel setup
    kernel = clCreateKernel(prog, "matMul", &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make kernel fail" << endl;
        exit(err_num);
    }

    // array rounding
    float* C = new float[ah*bw];
    int pad_to = 64;

    int bw_round = round_up(bw, pad_to);
    int ud_round = round_up(ud, pad_to);
    int ah_round = round_up(ah, pad_to);

    float* round_A = pad(A, ah, ud, pad_to);
    float* round_B = pad(B, ud, bw, pad_to);
    float* round_C = pad(C, ah, bw, pad_to);

    // work dim setup
    int global_work_size = [ah_round, bw_round];
    int local_work_size = [BLOCK_SIZE, BLOCK_SIZE];

    // buffer setup
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, ah_round*ud_round*sizeof(float), round_A, err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bw_round*ud_round*sizeof(float), round_B, err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, ah_round*bw_round*sizeof(float), round_C, err_num);
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
    err_num = clEnqueueNDRangeKernel(queue, kernel, 2, 0, global_wwork_size, local_work_size, 0, NULL, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel launch fail" << endl;
        exit(err_num);
    }

    // get results
    err_num = clEnqueueReadBuffer(queue, d_C, CL_FALSE, 0, ah_round*bw_round*sizeof(float), round_C, 0, NULL, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "read fail" << endl;
        exit(err_num);
    }
    clFinish(queue);

    // cleanup
    err_num |= clReaseMemObject(d_A);
    err_num |= clReaseMemObject(d_B);
    err_num |= clReaseMemObject(d_C);
    err_num |= clReaseKernel(kernel);
    err_num |= clReaseCommandQueue(queue);
    err_num |= clReaseProgram(prog);
    err_num |= clReaseContext(ctx);
    if (err_num != CL_SUCCESS)
    {
        cout << "free fail" << endl;
        exit(err_num);
    }
    free(d_A);
    free(d_B);
    free(d_C);

    // return results
    for (int i = 0; i < ah; i++)
    {
        for (int j = 0; j < bw; j++)
        {
            C[i*bw + j] = round_C[i*bw_round + j];
        }
    }

    return C;
}
