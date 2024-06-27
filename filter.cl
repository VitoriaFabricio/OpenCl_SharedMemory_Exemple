#define filterWidth 3
#define filterHeight 3

__kernel void filter(__global unsigned int* input_image,
                    __global unsigned int* output_image,
                    unsigned int height, unsigned int width) {

    // Indices           
    unsigned int gid = get_global_id(0);
    unsigned int wg = get_group_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int wgSize = get_local_size(0);
    
    // Work group position
    unsigned int y_work_group = (wg / (1920 / 64)) * 64;
    unsigned int x_work_group = (wg % (1920 / 64)) * 64;
    // Local position within the work group
    unsigned int y_local = lid / 64;
    unsigned int x_local = lid % 64;

    // Dimensions of the block processed by the work group, with margin for the filter
    __local unsigned int local_block[(64 + 2) * (64 + 2)]; 

    // Load the center values
    for (int y = 0; y < 64; y++) {
        int global_x = x_work_group + lid;
        int global_y = y_work_group + y;

        if (global_y < height) {
            local_block[(y + 1) * 66 + (lid + 1)] = input_image[global_y * 1920 + global_x];     
        }
    }

    // Corners

    // Top-left corner
    if (x_local == 0 && y_local == 0) { 
        if (x_work_group > 0 && y_work_group > 0) { 
            local_block[0] = input_image[(y_work_group - 1) * 1920 + (x_work_group - 1)]; 
        } 
    }  

    // Top-right corner
    if (x_local == 63 && y_local == 0) {
        if (x_work_group + 64 < width && y_work_group > 0) {
            local_block[65] = input_image[(y_work_group - 1) * 1920 + (x_work_group + 64)];
        }
    }

    // Bottom-left corner
    if (x_work_group > 0 && y_work_group + 64 < height) {
        local_block[(64 + 1) * 66] = input_image[(y_work_group + 64) * 1920 + (x_work_group - 1)];
    }

    // Bottom-right corner
    if (x_work_group + 64 < width && y_work_group + 64 < height) {
        local_block[(64 + 1) * 66 + (64 + 1)] = input_image[(y_work_group + 64) * 1920 + (x_work_group + 64)];
    }

    // Edges

    // Top edge 
    if (y_work_group > 0) {  
        local_block[x_local + 1] = input_image[(y_work_group - 1) * 1920 + x_work_group + x_local];
    }

    // Bottom edge
    if (y_work_group + 64 < height) {
        local_block[(64 + 1) * 66 + (lid + 1)] = input_image[(y_work_group + 64) * 1920 + (x_work_group + lid)];
    }

    // Left edge
    if (x_work_group > 0) {
        local_block[(lid + 1) * 66] = input_image[(y_work_group + lid) * 1920 + (x_work_group - 1)];
    }

    // Right edge
    if (x_work_group + 64 < width) {
        local_block[(lid + 1) * 66 + (64 + 1)] = input_image[(y_work_group + lid) * 1920 + (x_work_group + 64)];
    }

    // Ensure all work items have loaded local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Filter
    int filter[filterWidth][filterHeight] = {
        {1, 1, 1},
        {1, 3, 1},
        {1, 1, 1}
    };

    int filter_sum = 0;
    for (int i = 0; i < filterWidth; i++) {
        for (int j = 0; j < filterHeight; j++) {
            filter_sum += filter[i][j];
        }
    }

    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            int sum = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    sum += local_block[(y + 1 + i) * 66 + (x + 1 + j)] * filter[i + 1][j + 1];
                }
            }
            output_image[(y_work_group + y) * 1920 + (x_work_group + x)] = sum / filter_sum;
        }
    }
}
