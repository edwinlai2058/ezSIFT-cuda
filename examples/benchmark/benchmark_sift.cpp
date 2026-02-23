#include "ezsift.h"
#include <iostream>
#include <chrono>
#include <omp.h>


int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Please input an image filename.\n");
        printf("usage: benchmark_sift img1\n");
        return -1;
    }

    char *file1 = argv[1];

    ezsift::Image<unsigned char> image;
    if (image.read_pgm(file1) != 0) {
        std::cerr << "Failed to open input image!" << std::endl;
        return -1;
    }

    std::cout << "Image loaded, size: " << image.w << " x " << image.h << std::endl;

    // Double the original image as the first octive.
    ezsift::double_original_image(true);

    int nOctaves = 0; // Will be calculated by init_sift
    
    #pragma omp parallel
    {
        #pragma omp single
        printf("Running with %d OpenMP threads\n", omp_get_num_threads());
    }

    // Timer
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Build Gaussian Pyramid
    // 2. Build DoG Pyramid
    // 3. Detect Keypoints
    std::list<ezsift::SiftKeypoint> kpt_list;
    ezsift::sift_cpu(image, kpt_list, true);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Total SIFT time: " << duration.count() << " ms" << std::endl;
    std::cout << "Keypoints detected: " << kpt_list.size() << std::endl;

    return 0;
}
