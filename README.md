# OpenCl_SharedMemory_Exemple

This program demonstrates the use of OpenCL to apply a 3x3 filter to an image using shared memory.


*Using shared memory in OpenCL can significantly improve performance by reducing the number of global memory accesses, which are slower compared to shared memory accesses. This is especially beneficial when applying filters to images, as it allows for more efficient data handling within workgroups.*



The main components include:


Host Code (main.c):

- Reads an image from a CSV file (original_0.csv).
- Initializes the OpenCL environment: platform, device, context, command queue.
- Reads and builds the OpenCL kernel (filter.cl).
- Sets up memory buffers and transfers data between host and device.
- Executes the kernel to apply the filter to the image.
- Measures and prints the execution time.
- Reads the processed image from the device and writes the result to a CSV file (resultado.csv).

Kernel (filter.cl):

- Loads the input image into shared memory for efficient processing.
- Applies a 3x3 filter to the input image, using shared memory to optimize memory access.
- Handles edge cases for pixels at the borders and corners of the image.
- The filter weights and dimensions are defined within the kernel.



Compilation:

	(Geral) 	g++ -Wall main.c -o main -l OpenCL
	(AMD) 		g++ -Wall main.c -o main -I/opt/rocm-4.2.0/opencl/include/ -L/opt/rocm-4.2.0/lib/ -l OpenCL
	(NVIDIA) 	g++ -Wall main.c -o main -I /usr/local/cuda/include/ -l OpenCL

Execution:

./main
