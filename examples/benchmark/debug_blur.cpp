#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "ezsift.h"
#include "image.h"
#include "common.h"
#include "sift_cuda.h"

// Copy of CPU implementation to avoid link collision
int ref_row_filter_transpose(float* src, float* dst, int w, int h, float* coef1d, int gR) {
    float* row_buf = new float[w + gR * 2];
    float* srcData = src;
    
    for (int r = 0; r < h; r++) {
        float* src_row_start = src + r * w;
        memcpy(row_buf + gR, src_row_start, sizeof(float) * w);
        float firstData = *src_row_start;
        float lastData = *(src_row_start + w - 1);
        for (int i = 0; i < gR; i++) {
            row_buf[i] = firstData;
            row_buf[i + w + gR] = lastData;
        }
        
        float* prow = row_buf;
        for (int c = 0; c < w; c++) {
            float partialSum = 0.0f;
            float* coef = coef1d;
            for (int k = -gR; k <= gR; k++) {
                partialSum += (*coef++) * (*prow++);
            }
            prow -= 2 * gR;
            dst[c * h + r] = partialSum;
        }
    }
    delete[] row_buf;
    return 0;
}

int ref_gaussian_blur(const ezsift::Image<float>& in_image, ezsift::Image<float>& out_image, std::vector<float> coef1d) {
    int w = in_image.w;
    int h = in_image.h;
    ezsift::Image<float> tmp_image(h, w);
    
    int gR = coef1d.size() / 2;
    ref_row_filter_transpose(in_image.data, tmp_image.data, w, h, coef1d.data(), gR);
    ref_row_filter_transpose(tmp_image.data, out_image.data, h, w, coef1d.data(), gR);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }

    ezsift::Image<unsigned char> img_uchar;
    if (img_uchar.read_pgm(argv[1]) != 0) {
        printf("Failed to read image.\n");
        return -1;
    }

    ezsift::Image<float> base_cpu = img_uchar.to_float();
    ezsift::Image<float> base_gpu = img_uchar.to_float(); // Float 0-255

    for (int r = 1; r <= 16; r++) { // Test up to 16
         int gR = r;
         int w = 2*gR + 1;
         std::vector<float> kernel(w);
         float sigma = r / 3.0f; // Approx
         if (sigma < 0.5f) sigma = 0.5f;
         
         float accu = 0.0f;
         for(int j=0; j<w; j++) {
             float v = (j - gR) / sigma;
             // Match ezsift hack
             kernel[j] = expf(v*v*-0.5f) * (1.0f + j / 1000.0f);
             accu += kernel[j];
         }

         for(int j=0; j<w; j++) kernel[j] /= accu;
         
         printf("Testing Radius %d (Sigma %.2f)... ", gR, sigma);
         
         ezsift::Image<float> out_cpu(base_cpu.w, base_cpu.h);
         ezsift::Image<float> out_gpu(base_gpu.w, base_gpu.h);
         
         ref_gaussian_blur(base_cpu, out_cpu, kernel);
         ezsift::gaussian_blur_cuda(base_gpu, out_gpu, kernel);
         
         float max_diff = 0.0f;
         int diff_count = 0;
         for (int i = 0; i < out_cpu.w * out_cpu.h; i++) {
            float diff = std::fabs(out_cpu.data[i] - out_gpu.data[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-3) diff_count++;
         }
         
         if (diff_count == 0) {
             printf("PASS (Max Diff: %f)\n", max_diff);
         } else {
             printf("FAIL (Max Diff: %f, Count: %d)\n", max_diff, diff_count);
         }
    }

    return 0;
}
