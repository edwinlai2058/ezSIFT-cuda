#include "sift_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define CHECKn(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " << \
        cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

namespace ezsift {

// Constant memory for filter coefficients
#define MAX_KERNEL_RADIUS 32
__constant__ float d_coef1d[MAX_KERNEL_RADIUS * 2 + 1];

__constant__ int d_gR;

// Simple kernel: One thread per pixel
// row_filter_transpose: 
// Src: w (width), h (height)
// Dst: h (width), w (height)
// Operation: Convolve row 'r' of src, write to column 'r' of dst (at row c)
// dst[c * h + r] = result
__global__ void row_filter_transpose_kernel(float* src, float* dst, int w, int h) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= h || c >= w) return;

    // Convolution
    float sum = 0.0f;
    int gR = d_gR;
    
    // Center pixel: (r, c)
    // We iterate kernel from -gR to gR
    // Neighbors are (r, c+k)
    
    // Wait, row filter means filtering along the row (varying column index).
    // So for pixel (r, c), we look at (r, c-gR) ... (r, c+gR)
    
    for (int k = -gR; k <= gR; k++) {
        int col_idx = c + k;
        // Replication PADDING
        if (col_idx < 0) col_idx = 0;
        if (col_idx >= w) col_idx = w - 1;
        
        float val = src[r * w + col_idx];
        // Coef index: k + gR
        sum += val * d_coef1d[k + gR];
    }
    
    // Write to transpose location
    // Dst dimensions: width=h, height=w
    // Destination coord: (c, r) -> index = c * h + r
    dst[c * h + r] = sum;
}

int allocate_device_image(float** d_mem, int w, int h) {
    size_t size = w * h * sizeof(float);
    CHECKn(cudaMalloc(d_mem, size));
    return 0;
}

int upload_image(float* d_mem, float* h_mem, int w, int h) {
    auto ret = cudaMemcpy(d_mem, h_mem, w * h * sizeof(float), cudaMemcpyHostToDevice);
    return (int)ret;
}

int download_image(float* d_mem, Image<float>& h_img) {
    size_t size = h_img.w * h_img.h * sizeof(float);
    CHECKn(cudaMemcpy(h_img.data, d_mem, size, cudaMemcpyDeviceToHost));
    return 0;
}

int free_device_mem(float* d_mem) {
    if (d_mem) CHECKn(cudaFree(d_mem));
    return 0;
}

int row_filter_transpose_cuda(float* d_src, float* d_dst, int w, int h) {
    dim3 block(32, 16); // 32x16 = 512 threads
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    
    row_filter_transpose_kernel<<<grid, block>>>(d_src, d_dst, w, h);
    CHECKn(cudaGetLastError());
    
    return 0;
}

int gaussian_blur_cuda(const Image<float>& in_image, Image<float>& out_image, std::vector<float> coef1d) {
    int w = in_image.w;
    int h = in_image.h;
    
    // 1. Copy coefficients to constant memory
    int gR = (int)coef1d.size() / 2;
    if (gR > MAX_KERNEL_RADIUS) {
        std::cerr << "Error: Kernel radius too large for constant memory!" << std::endl;
        return -1;
    }
    
    CHECKn(cudaMemcpyToSymbol(d_coef1d, coef1d.data(), coef1d.size() * sizeof(float)));
    CHECKn(cudaMemcpyToSymbol(d_gR, &gR, sizeof(int)));
    
    // 2. Allocate device memory
    // Depending on optimization strategy, we might want to keep data on GPU
    // But for this function signature (cpu image input/output), we copy here.
    // Ideally we should refactor to keep images on GPU.
    // For now, let's implement the copy-based version to verify correctness/speedup of kernel itself
    // vs transfer overhead.
    
    float *d_src, *d_tmp;
    size_t size = w * h * sizeof(float);
    
    CHECKn(cudaMalloc(&d_src, size));
    CHECKn(cudaMalloc(&d_tmp, size)); // Transposed buffer (h * w)

    
    // Reuse d_src for final output? 
    // Step 1: src -> tmp (filter rows, transpose)
    // Step 2: tmp -> src (filter rows of tmp, transpose back to src dims)
    // So yes, we can reuse d_src for final result.
    
    // Copy input
    CHECKn(cudaMemcpy(d_src, in_image.data, size, cudaMemcpyHostToDevice));
    
    // Pass 1: Src(w,h) -> Tmp(h,w)
    row_filter_transpose_cuda(d_src, d_tmp, w, h);
    
    // Pass 2: Tmp(h,w) -> Src(w,h)
    // Note dimensions swapped: w'=h, h'=w
    row_filter_transpose_cuda(d_tmp, d_src, h, w);
    
    // Copy back
    CHECKn(cudaMemcpy(out_image.data, d_src, size, cudaMemcpyDeviceToHost));
    
    CHECKn(cudaFree(d_src));
    CHECKn(cudaFree(d_tmp));
    
    return 0;
}

// Subtract kernel: out = im2 - im1
__global__ void subtract_kernel(float* im1, float* im2, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    out[idx] = im2[idx] - im1[idx];
}

int subtract_cuda(const Image<float>& im1, const Image<float>& im2, Image<float>& out_image) {
    int w = im1.w;
    int h = im1.h;
    size_t size = w * h * sizeof(float);
    
    float *d_im1, *d_im2, *d_out;
    CHECKn(cudaMalloc(&d_im1, size));
    CHECKn(cudaMalloc(&d_im2, size));
    CHECKn(cudaMalloc(&d_out, size));
    
    CHECKn(cudaMemcpy(d_im1, im1.data, size, cudaMemcpyHostToDevice));
    CHECKn(cudaMemcpy(d_im2, im2.data, size, cudaMemcpyHostToDevice));
    
    dim3 block(32, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    
    subtract_kernel<<<grid, block>>>(d_im1, d_im2, d_out, w, h);
    CHECKn(cudaGetLastError());
    
    CHECKn(cudaMemcpy(out_image.data, d_out, size, cudaMemcpyDeviceToHost));
    
    CHECKn(cudaFree(d_im1));
    CHECKn(cudaFree(d_im2));
    CHECKn(cudaFree(d_out));
    
    return 0;
}

// Downsample 2x kernel
// src: w x h
// dst: w/2 x h/2
__global__ void downsample_kernel(float* src, float* dst, int src_w, int src_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // dst x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // dst y
    
    int dst_w = src_w / 2;
    int dst_h = src_h / 2;
    
    if (x >= dst_w || y >= dst_h) return;
    
    // Nearest neighbor: src(2x, 2y)
    // To match CPU implementation exactly:
    // ezsift uses: dst[r*w_dst + c] = src[(r*2)*w_src + (c*2)];
    
    int src_idx = (y * 2) * src_w + (x * 2);
    int dst_idx = y * dst_w + x;
    
    dst[dst_idx] = src[src_idx];
}

int downsample_cuda(const Image<float>& in_image, Image<float>& out_image) {
    int w = in_image.w;
    int h = in_image.h;
    size_t src_size = w * h * sizeof(float);
    size_t dst_size = out_image.w * out_image.h * sizeof(float);
    
    float *d_src, *d_dst;
    CHECKn(cudaMalloc(&d_src, src_size));
    CHECKn(cudaMalloc(&d_dst, dst_size));
    
    CHECKn(cudaMemcpy(d_src, in_image.data, src_size, cudaMemcpyHostToDevice));
    
    dim3 block(32, 16);
    dim3 grid((out_image.w + block.x - 1) / block.x, (out_image.h + block.y - 1) / block.y);
    
    downsample_kernel<<<grid, block>>>(d_src, d_dst, w, h);
    CHECKn(cudaGetLastError());
    
    CHECKn(cudaMemcpy(out_image.data, d_dst, dst_size, cudaMemcpyDeviceToHost));
    
    CHECKn(cudaFree(d_src));
    CHECKn(cudaFree(d_dst));
    
    return 0;
}

    


// ==========================================
// Device versions (No alloc/copy)
// ==========================================

int gaussian_blur_cuda_device(float* d_src, float* d_dst, int w, int h, std::vector<float> coef1d) {
    // Copy coefficients to constant memory (still needed, or done once globally?)
    // For safety, do it here. Cost is low.
    int gR = (int)coef1d.size() / 2;
    if (gR > MAX_KERNEL_RADIUS) {
        std::cerr << "Error: Radius " << gR << " > " << MAX_KERNEL_RADIUS << std::endl;
        return -1;
    }

    
    // Note: This is an async call usually, but host memory copy is sync unless pinned.
    // coef1d is small.
    CHECKn(cudaMemcpyToSymbol(d_coef1d, coef1d.data(), coef1d.size() * sizeof(float)));
    CHECKn(cudaMemcpyToSymbol(d_gR, &gR, sizeof(int)));
    CHECKn(cudaDeviceSynchronize());

    
    // We need a temporary buffer for the transpose step.
    // The calling function should ideally provide this workspace to avoid malloc here.
    // BUT, for now, we allocate TMP here. The big win is avoiding the Image transfer.
    // Optimization: Pre-allocate workspace in ezsift.cpp loop.
    
    float *d_tmp;
    CHECKn(cudaMalloc(&d_tmp, w * h * sizeof(float)));
    
    // Pass 1: Src(w,h) -> Tmp(h,w)
    row_filter_transpose_cuda(d_src, d_tmp, w, h);
    
    // Pass 2: Tmp(h,w) -> Src(w,h) (writing to dst)
    // Note dimensions swapped: w'=h, h'=w
    row_filter_transpose_cuda(d_tmp, d_dst, h, w);
    
    CHECKn(cudaDeviceSynchronize()); // Prevent race on d_coef1d constant memory
    CHECKn(cudaFree(d_tmp));
    return 0;

}

int subtract_cuda_device(float* d_im1, float* d_im2, float* d_out, int w, int h) {
    dim3 block(32, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    subtract_kernel<<<grid, block>>>(d_im1, d_im2, d_out, w, h);
    CHECKn(cudaGetLastError());
    return 0;
}

int downsample_cuda_device(float* d_src, float* d_dst, int src_w, int src_h) {
    int dst_w = src_w / 2;
    int dst_h = src_h / 2;
    dim3 block(32, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    downsample_kernel<<<grid, block>>>(d_src, d_dst, src_w, src_h);
    CHECKn(cudaGetLastError());
    return 0;
}



// Full Pyramid Build on GPU (Device Resident)
int build_gaussian_pyramid_cuda_device(
    float* d_base, int w, int h,
    std::vector<float*>& d_gpyr, 
    int nOctaves, int nGpyrLayers, 
    const std::vector<std::vector<float>>& gaussian_coefs)
{
    d_gpyr.resize(nOctaves * nGpyrLayers);
    
    for (int i = 0; i < nOctaves; i++) {
        for (int j = 0; j < nGpyrLayers; j++) {
            int idx = i * nGpyrLayers + j;
            int curr_w, curr_h;
            
            if (i == 0 && j == 0) {
                // Base image
                curr_w = w;
                curr_h = h;
                
                // Allocate and Copy from d_base (Device->Device)
                CHECKn(cudaMalloc(&d_gpyr[idx], curr_w * curr_h * sizeof(float)));
                CHECKn(cudaMemcpy(d_gpyr[idx], d_base, curr_w * curr_h * sizeof(float), cudaMemcpyDeviceToDevice));
                
                gaussian_blur_cuda_device(d_base, d_gpyr[idx], curr_w, curr_h, gaussian_coefs[j]);

            } else if (i > 0 && j == 0) {
                // Downsample from previous octave's last required scale
                int nScales = nGpyrLayers - 3;
                int src_idx = (i - 1) * nGpyrLayers + nScales;
                
                // We need dims of src
                int prev_w = w >> (i - 1);
                int prev_h = h >> (i - 1);
                
                curr_w = w >> i;
                curr_h = h >> i;
                
                CHECKn(cudaMalloc(&d_gpyr[idx], curr_w * curr_h * sizeof(float)));
                downsample_cuda_device(d_gpyr[src_idx], d_gpyr[idx], prev_w, prev_h);
                
            } else {
                // Gaussian Blur
                int prev_idx = idx - 1;
                curr_w = w >> i;
                curr_h = h >> i;
                
                CHECKn(cudaMalloc(&d_gpyr[idx], curr_w * curr_h * sizeof(float)));
                gaussian_blur_cuda_device(d_gpyr[prev_idx], d_gpyr[idx], curr_w, curr_h, gaussian_coefs[j]);
            }
        }
    }
    return 0;
}

// Task 14: Keypoint Detection Kernel
__global__ void detect_extrema_kernel(float* low, float* curr, float* high,
                                      int w, int h, float threshold,
                                      int octave, int layer,
                                      SiftKeypointCoords* keypoints, int* counter, int max_kpts)
{
    // Use offset 5 for border
    int c = blockIdx.x * blockDim.x + threadIdx.x + 5; 
    int r = blockIdx.y * blockDim.y + threadIdx.y + 5;

    // Check boundary (h - 5)
    if (c >= w - 5 || r >= h - 5) return;

    int idx = r * w + c;
    float val = curr[idx];

    // Threshold check
    if (fabsf(val) < threshold) return;

    // Check neighbors
    // We check 26 neighbors: 8 in curr, 9 in low, 9 in high.
    // Optimization: Check 8 in curr first to fail fast.
    
    // Offsets for 3x3 neighborhood centered at idx
    // Rows: r-1, r, r+1. Cols: c-1, c, c+1
    // row step: w
    
    // Pre-calculate offsets
    // -w-1, -w, -w+1
    // -1,       +1
    // +w-1, +w, +w+1
    
    // Check current layer first (8 neighbors)
    if (val > 0) {
        if (val <= curr[idx - w - 1] || val <= curr[idx - w] || val <= curr[idx - w + 1] ||
            val <= curr[idx - 1]     ||                         val <= curr[idx + 1]     ||
            val <= curr[idx + w - 1] || val <= curr[idx + w] || val <= curr[idx + w + 1])
            return;
            
        // Check low layer (9 neighbors)
        int low_idx = idx; // pointers are already offset to start of image
        if (val <= low[low_idx - w - 1] || val <= low[low_idx - w] || val <= low[low_idx - w + 1] ||
            val <= low[low_idx - 1]     || val <= low[low_idx]     || val <= low[low_idx + 1]     ||
            val <= low[low_idx + w - 1] || val <= low[low_idx + w] || val <= low[low_idx + w + 1])
            return;

        // Check high layer (9 neighbors)
        int high_idx = idx;
        if (val <= high[high_idx - w - 1] || val <= high[high_idx - w] || val <= high[high_idx - w + 1] ||
            val <= high[high_idx - 1]     || val <= high[high_idx]     || val <= high[high_idx + 1]     ||
            val <= high[high_idx + w - 1] || val <= high[high_idx + w] || val <= high[high_idx + w + 1])
            return;
    } else {
        // val < 0
        if (val >= curr[idx - w - 1] || val >= curr[idx - w] || val >= curr[idx - w + 1] ||
            val >= curr[idx - 1]     ||                         val >= curr[idx + 1]     ||
            val >= curr[idx + w - 1] || val >= curr[idx + w] || val >= curr[idx + w + 1])
            return;

        int low_idx = idx;
        if (val >= low[low_idx - w - 1] || val >= low[low_idx - w] || val >= low[low_idx - w + 1] ||
            val >= low[low_idx - 1]     || val >= low[low_idx]     || val >= low[low_idx + 1]     ||
            val >= low[low_idx + w - 1] || val >= low[low_idx + w] || val >= low[low_idx + w + 1])
            return;
            
        int high_idx = idx;
        if (val >= high[high_idx - w - 1] || val >= high[high_idx - w] || val >= high[high_idx - w + 1] ||
            val >= high[high_idx - 1]     || val >= high[high_idx]     || val >= high[high_idx + 1]     ||
            val >= high[high_idx + w - 1] || val >= high[high_idx + w] || val >= high[high_idx + w + 1])
            return;
    }

    // Found extrema
    int old = atomicAdd(counter, 1);
    if (old < max_kpts) {
        keypoints[old].octave = octave;
        keypoints[old].layer = layer;
        keypoints[old].r = r;
        keypoints[old].c = c;
    }
}

std::vector<SiftKeypointCoords> detect_keypoints_cuda(
    const std::vector<Image<float>>& dogPyr, 
    int nOctaves, 
    int nDogLayers,
    float contrast_threshold)
{
    std::vector<SiftKeypointCoords> all_kpts;
    int max_kpts = 100000;
    
    SiftKeypointCoords* d_kpts;
    int* d_counter;
    
    // Allocate result buffers
    CHECKn(cudaMalloc(&d_kpts, max_kpts * sizeof(SiftKeypointCoords)));
    CHECKn(cudaMalloc(&d_counter, sizeof(int)));
    
    // Allocate image buffers (reusable)
    // Find max size
    int max_w = dogPyr[0].w;
    int max_h = dogPyr[0].h;
    size_t max_size = max_w * max_h * sizeof(float);
    
    float *d_low, *d_curr, *d_high;
    CHECKn(cudaMalloc(&d_low, max_size));
    CHECKn(cudaMalloc(&d_curr, max_size));
    CHECKn(cudaMalloc(&d_high, max_size));
    
    for (int i = 0; i < nOctaves; i++) {
        int w = dogPyr[i * nDogLayers].w;
        int h = dogPyr[i * nDogLayers].h;
        size_t size = w * h * sizeof(float);
        
        for (int j = 1; j < nDogLayers - 1; j++) {
            int layer_index = i * nDogLayers + j;
            
            // Pointers to host data
            float* h_low = dogPyr[layer_index - 1].data;
            float* h_curr = dogPyr[layer_index].data;
            float* h_high = dogPyr[layer_index + 1].data;
            
            // Copy to device
            // Optimization: Could cache previous layers, but simplistic copy is robust
            CHECKn(cudaMemcpy(d_low, h_low, size, cudaMemcpyHostToDevice));
            CHECKn(cudaMemcpy(d_curr, h_curr, size, cudaMemcpyHostToDevice));
            CHECKn(cudaMemcpy(d_high, h_high, size, cudaMemcpyHostToDevice));
            
        }
    }
    
    // Initialize counter to 0 globally
    CHECKn(cudaMemset(d_counter, 0, sizeof(int)));
    
    for (int i = 0; i < nOctaves; i++) {
        int w = dogPyr[i * nDogLayers].w;
        int h = dogPyr[i * nDogLayers].h;
        size_t size = w * h * sizeof(float);
        
        for (int j = 1; j < nDogLayers - 1; j++) {
            int layer_index = i * nDogLayers + j;
            
            CHECKn(cudaMemcpy(d_low, dogPyr[layer_index - 1].data, size, cudaMemcpyHostToDevice));
            CHECKn(cudaMemcpy(d_curr, dogPyr[layer_index].data, size, cudaMemcpyHostToDevice));
            CHECKn(cudaMemcpy(d_high, dogPyr[layer_index + 1].data, size, cudaMemcpyHostToDevice));
            
            dim3 block(32, 16);
            // Grid covers w-10, h-10 effectively
            dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
            
            detect_extrema_kernel<<<grid, block>>>(d_low, d_curr, d_high, w, h, contrast_threshold, i, j, d_kpts, d_counter, max_kpts);
            CHECKn(cudaGetLastError());
        }
    }
    
    // Copy back
    int count;
    CHECKn(cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (count > max_kpts) count = max_kpts;
    
    std::vector<SiftKeypointCoords> results(count);
    if (count > 0) {
        CHECKn(cudaMemcpy(results.data(), d_kpts, count * sizeof(SiftKeypointCoords), cudaMemcpyDeviceToHost));
    }
    
    CHECKn(cudaFree(d_kpts));
    CHECKn(cudaFree(d_counter));
    CHECKn(cudaFree(d_low));
    CHECKn(cudaFree(d_curr));
    CHECKn(cudaFree(d_high));
    
    return results;
}

int build_dog_pyr_cuda_device(
    const std::vector<float*>& d_gpyr,
    std::vector<float*>& d_dogPyr,
    int nOctaves, int nDogLayers,
    int base_w, int base_h)
{
    int nGpyrLayers = nDogLayers + 1;
    d_dogPyr.resize(nOctaves * nDogLayers);

    for (int i = 0; i < nOctaves; i++) {
        // Calculate dims for this octave
        int w = base_w >> i;
        int h = base_h >> i;
        if (i == 0 && (nGpyrLayers > 0) && d_gpyr[0] == nullptr) {
            // Should not happen if gpyr is built correctly
             return -1;
        }

        for (int j = 0; j < nDogLayers; j++) {
            int dog_idx = i * nDogLayers + j;
            int gpyr_idx1 = i * nGpyrLayers + j;
            int gpyr_idx2 = i * nGpyrLayers + j + 1;
            
            // Allocate DoG layer
            CHECKn(cudaMalloc(&d_dogPyr[dog_idx], w * h * sizeof(float)));
            
            // Subtraction: out = im2 - im1
            // We can reuse subtract_cuda_device logic
            subtract_cuda_device(d_gpyr[gpyr_idx1], d_gpyr[gpyr_idx2], d_dogPyr[dog_idx], w, h);
        }
    }
    return 0;
}


std::vector<SiftKeypointCoords> detect_keypoints_cuda_device(
    const std::vector<float*>& d_dogPyr, 
    int nOctaves, 
    int nDogLayers,
    int base_w, int base_h,
    float contrast_threshold)
{
    std::vector<SiftKeypointCoords> all_kpts;
    int max_kpts = 100000; // Cap
    
    SiftKeypointCoords* d_kpts;
    int* d_counter;
    
    // Allocate result buffers
    CHECKn(cudaMalloc(&d_kpts, max_kpts * sizeof(SiftKeypointCoords)));
    CHECKn(cudaMalloc(&d_counter, sizeof(int)));
    CHECKn(cudaMemset(d_counter, 0, sizeof(int)));
    
    for (int i = 0; i < nOctaves; i++) {
        int w = base_w >> i;
        int h = base_h >> i;
        
        for (int j = 1; j < nDogLayers - 1; j++) {
            int layer_index = i * nDogLayers + j;
            
            float* d_low = d_dogPyr[layer_index - 1];
            float* d_curr = d_dogPyr[layer_index];
            float* d_high = d_dogPyr[layer_index + 1];
            
            dim3 block(32, 16);
            dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
            
            detect_extrema_kernel<<<grid, block>>>(d_low, d_curr, d_high, w, h, contrast_threshold, i, j, d_kpts, d_counter, max_kpts);
        }
    }
    
    CHECKn(cudaGetLastError()); // Check for kernel errors
    
    // Copy back result count
    int count;
    CHECKn(cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (count > max_kpts) count = max_kpts;
    
    std::vector<SiftKeypointCoords> results(count);
    if (count > 0) {
        CHECKn(cudaMemcpy(results.data(), d_kpts, count * sizeof(SiftKeypointCoords), cudaMemcpyDeviceToHost));
    }
    
    CHECKn(cudaFree(d_kpts));
    CHECKn(cudaFree(d_counter));
    
    return results;
}

void free_pyramid_device(std::vector<float*>& pyr) {
    for(size_t i=0; i<pyr.size(); i++) {
        if(pyr[i]) {
            cudaFree(pyr[i]);
            pyr[i] = nullptr;
        }
    }
    pyr.clear();
}

} // namespace ezsift

