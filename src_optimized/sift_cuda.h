
#ifndef SIFT_CUDA_H
#define SIFT_CUDA_H

#include "image.h"
#include <vector>

namespace ezsift {
    int gaussian_blur_cuda(const Image<float>& in_image, Image<float>& out_image, std::vector<float> coef1d);
    int subtract_cuda(const Image<float>& im1, const Image<float>& im2, Image<float>& out_image);
    int subtract_cuda(const Image<float>& im1, const Image<float>& im2, Image<float>& out_image);
    int downsample_cuda(const Image<float>& in_image, Image<float>& out_image);

    // Device-resident versions
    struct SiftKeypointCoords {
        int octave;
        int layer;
        int r;
        int c;
    };

    // Helper functions for memory management to avoid cuda headers in cpp
    int allocate_device_image(float** d_mem, int w, int h);
    int upload_image(float* d_mem, float* h_mem, int w, int h);
    int download_image(float* d_mem, Image<float>& h_img);
    int free_device_mem(float* d_mem);

    // Device-resident versions (pointers/vectors of pointers must be on GPU or managed)
    
    // Pyramid Construction (returns vector of device pointers)
    // IMPORTANT: Caller is responsible for allocating and freeing the result pointers!
    // We use std::vector<float*> to store device pointers on Host.
    int build_gaussian_pyramid_cuda_device(
        float* d_base, int w, int h,
        std::vector<float*>& d_gpyr, 
        int nOctaves, int nGpyrLayers, 
        const std::vector<std::vector<float>>& gaussian_coefs);

    int build_dog_pyr_cuda_device(
        const std::vector<float*>& d_gpyr,
        std::vector<float*>& d_dogPyr,
        int nOctaves, int nDogLayers,
        int base_w, int base_h);

    // Keypoint Detection using device pointers
    std::vector<SiftKeypointCoords> detect_keypoints_cuda_device(
        const std::vector<float*>& d_dogPyr, 
        int nOctaves, 
        int nDogLayers,
        int base_w, int base_h,
        float contrast_threshold);

    // Helper to free device pointer vector
    void free_pyramid_device(std::vector<float*>& pyr);
}




#endif
