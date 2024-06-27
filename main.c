#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>

#define PROGRAM_FILE "filter.cl"
#define KERNEL_FUNC "filter"

#define MAX_SOURCE_SIZE (0x100000)

const unsigned int width = 1920;
const unsigned int height = 1024;

// iagostorch
void probe_error(cl_int error, const char* message){
    if (error != CL_SUCCESS ) {
        printf("Code %d, %s\n", error, message);
    }
}

int main() {
    // Read source program
    char *kernelSource;
    size_t source_size;

    double nanoSeconds = 0;
    // These variabels are used to profile the time spend writing to memory objects "clEnqueueWriteBuffer"
    cl_ulong time_start;
    cl_ulong time_end;
    cl_event event;

 
    FILE* fp = fopen(PROGRAM_FILE, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(kernelSource, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Length of vectors
    unsigned int dimension = width * height;

    // Host input vectors
    int *h_input_image;

    // Host output vector
    int *h_output_image;
 
    // Device input buffers
    cl_mem d_input_image;

    // Device output buffer
    cl_mem d_output_image;
 
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
 
    // Size, in bytes, of each vector
    size_t bytes = dimension * sizeof(int);

    // Allocate memory for each matrix on host
    h_input_image = (int*)malloc(bytes);
    h_output_image = (int*)malloc(bytes);

    // Initialize matrices
    FILE* file = fopen("original_0.csv", "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to load file.\n");
        exit(1);
    }
    
    char line[10240];   
    unsigned int row = 0;

    while (fgets(line, sizeof(line), file) && row < height) {
        char *token;
        unsigned int col = 0;

        token = strtok(line, ",");
        while (token != NULL && col < width) {
            h_input_image[row * width + col] = atoi(token);  // Use atoi to convert string to int
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    fclose(file);

    for (unsigned int i = 0; i < width; i++) {
        for (unsigned int j = 0; j < height; j++) {
            h_output_image[i * width + j] = 0;
        }
    }

    size_t globalSize, localSize;
    cl_int err;
    unsigned int nwg;
 
    // Number of work items in each local work group
    localSize = 64;
 
    // Number of total work items - localSize must be divisor
    nwg = ceil((float)dimension /(localSize*localSize) );
    globalSize = nwg*localSize;
 
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
 
    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue 
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
 
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, &source_size, &err);
    // iagostorch
    probe_error(err, "ERROR - Creating program from source");
 
    // Build the program executable 
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    // iagostorch
    probe_error(err, "ERROR - Building program\n");

    // iagostorch
    // Show debugging information when the build is not successful
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        free(log);
    }
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
 
    // Create the input and output arrays in device memory for our calculation
    d_input_image = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_output_image = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_input_image, CL_TRUE, 0, bytes, h_input_image, 0, NULL, NULL);
    probe_error(err, "ERROR - Writing to input buffer");
    err |= clEnqueueWriteBuffer(queue, d_output_image, CL_TRUE, 0, bytes, h_output_image, 0, NULL, NULL);
    probe_error(err, "ERROR - Writing to output buffer");
 
    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input_image);
    probe_error(err, "ERROR - Setting kernel arg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output_image);
    probe_error(err, "ERROR - Setting kernel arg 1");
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &height); 
    probe_error(err, "ERROR - Setting kernel arg 2");
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &width);
    probe_error(err, "ERROR - Setting kernel arg 3");

    // Execute the kernel over the entire range of the data set  
    printf("%ld,%ld, %d\n", globalSize, localSize, nwg);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &event);
    probe_error(err, "ERROR - Enqueueing kernel");

    err = clWaitForEvents(1, &event);
    probe_error(err, (char*)"Error waiting for events\n");

 
    // Wait for the command queue to get serviced before reading back results
    err =    clFinish(queue);
   // clFinish(queue);
    probe_error(err, "ERROR - Error infinite");

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    nanoSeconds = time_end-time_start;

    printf("Tempo de execucao:%f \n", nanoSeconds);
 
    // Read the results from the device
    err = clEnqueueReadBuffer(queue, d_output_image, CL_TRUE, 0, bytes, h_output_image, 0, NULL, NULL);
    probe_error(err, "ERROR - Reading from output buffer");
 
    // Print results (optional)
    // printf("\nResults:\n");
    // for (unsigned int i = 0; i < width; i++) {
    //     for (unsigned int j = 0; j < height; j++) {
    //         printf("%d ", h_output_image[i * width + j]);
    //     }
    //     printf("\n");
    // }

    // Save results to a CSV file
    FILE* outFile = fopen("resultado.csv", "w");
    if (outFile == NULL) {
        fprintf(stderr, "Failed to open result file.\n");
        exit(1);
    }

    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            fprintf(outFile, "%d", h_output_image[i * width + j]);
            if (j < width - 1) {
                fprintf(outFile, ",");
            }
        }
        fprintf(outFile, "\n");
    }

    fclose(outFile);

    // Release OpenCL resources
    clReleaseMemObject(d_input_image);
    clReleaseMemObject(d_output_image);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Release host memory
    //free(h_input_image);
    //free(h_output_image);
    //free(kernelSource);
    

    return 0;
}
