/* No-OpenCV Implementation for Image Stitching
   Dependencies: standard C++ library, ezsift
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "ezsift.h"

using namespace std;

// ==========================================
// Part 0: 基本結構與 I/O (取代 cv::Point2f, cv::Mat)
// ==========================================

struct Point2f {
    float x, y;
};

// 簡單的影像結構，支援灰階
struct SimpleImage {
    int w, h;
    vector<unsigned char> data;

    SimpleImage(int width, int height) : w(width), h(height) { data.resize(w * h, 0); }

    unsigned char get(int x, int y) const {
        if (x < 0 || x >= w || y < 0 || y >= h)
            return 0;
        return data[y * w + x];
    }

    void set(int x, int y, unsigned char val) {
        if (x >= 0 && x < w && y >= 0 && y < h) {
            data[y * w + x] = val;
        }
    }
};

// 寫入 PGM 檔案 (可用一般看圖軟體開啟)
void writePGM(const string& filename, const SimpleImage& img) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Cannot open file for writing: " << filename << endl;
        return;
    }
    file << "P5\n" << img.w << " " << img.h << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    file.close();
}

// ==========================================
// Part 1: 手刻數學運算 (取代 OpenCV 的矩陣運算)
// ==========================================

// 高斯消去法解線性方程組 Ax = B
// A 是 8x8 矩陣, B 是 8x1 向量, result 會存 8 個未知數
bool solveGaussian(vector<vector<double>>& A, vector<double>& B, vector<double>& result) {
    int n = 8;
    for (int i = 0; i < n; i++) {
        // Pivot
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (abs(A[k][i]) > abs(A[maxRow][i]))
                maxRow = k;
        }
        swap(A[i], A[maxRow]);
        swap(B[i], B[maxRow]);

        if (abs(A[i][i]) < 1e-9)
            return false;  // Singular matrix

        // Eliminate
        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; j++) {
                A[k][j] -= factor * A[i][j];
            }
            B[k] -= factor * B[i];
        }
    }

    // Back substitution
    result.resize(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * result[j];
        }
        result[i] = (B[i] - sum) / A[i][i];
    }
    return true;
}

// 計算 Homography (取代 getPerspectiveTransform)
// 輸入 4 對點，輸出 3x3 矩陣 (最後一個元素固定為 1)
vector<double> computeHomography4Points(const vector<Point2f>& src, const vector<Point2f>& dst) {
    // h00*x + h01*y + h02 - h20*u*x - h21*u*y = u
    // h10*x + h11*y + h12 - h20*v*x - h21*v*y = v
    vector<vector<double>> A(8, vector<double>(8, 0.0));
    vector<double> B(8, 0.0);

    for (int i = 0; i < 4; i++) {
        double x = src[i].x;
        double y = src[i].y;
        double u = dst[i].x;
        double v = dst[i].y;

        // Equation 1 for x'
        A[2 * i][0] = x;
        A[2 * i][1] = y;
        A[2 * i][2] = 1;
        A[2 * i][6] = -u * x;
        A[2 * i][7] = -u * y;
        B[2 * i] = u;

        // Equation 2 for y'
        A[2 * i + 1][3] = x;
        A[2 * i + 1][4] = y;
        A[2 * i + 1][5] = 1;
        A[2 * i + 1][6] = -v * x;
        A[2 * i + 1][7] = -v * y;
        B[2 * i + 1] = v;
    }

    vector<double> res;
    if (!solveGaussian(A, B, res))
        return {};

    // 補上最後一個元素 h22 = 1
    res.push_back(1.0);
    return res;
}

// 3x3 矩陣反矩陣 (取代 H.inv())
// M 是 row-major 的 9 個元素
bool invert3x3(const vector<double>& M, vector<double>& Inv) {
    double det =
        M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);

    if (abs(det) < 1e-9)
        return false;
    double invDet = 1.0 / det;

    Inv.resize(9);
    Inv[0] = (M[4] * M[8] - M[5] * M[7]) * invDet;
    Inv[1] = (M[2] * M[7] - M[1] * M[8]) * invDet;
    Inv[2] = (M[1] * M[5] - M[2] * M[4]) * invDet;
    Inv[3] = (M[5] * M[6] - M[3] * M[8]) * invDet;
    Inv[4] = (M[0] * M[8] - M[2] * M[6]) * invDet;
    Inv[5] = (M[2] * M[3] - M[0] * M[5]) * invDet;
    Inv[6] = (M[3] * M[7] - M[4] * M[6]) * invDet;
    Inv[7] = (M[1] * M[6] - M[0] * M[7]) * invDet;
    Inv[8] = (M[0] * M[4] - M[1] * M[3]) * invDet;
    return true;
}

// ==========================================
// Part 2: RANSAC (現在只用 vector 和手刻數學)
// ==========================================

vector<double> computeHomographyRANSAC(const vector<Point2f>& srcPoints, const vector<Point2f>& dstPoints,
                                       int numIterations, float threshold) {
    vector<double> bestH;
    int maxInliers = -1;

    if (srcPoints.size() < 4)
        return {};

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, srcPoints.size() - 1);

    for (int i = 0; i < numIterations; i++) {
        vector<Point2f> srcSubset, dstSubset;

        // 1. 隨機選點 (這裡簡單實作，沒檢查重複，實際專案建議檢查)
        for (int k = 0; k < 4; k++) {
            int idx = distrib(gen);
            srcSubset.push_back(srcPoints[idx]);
            dstSubset.push_back(dstPoints[idx]);
        }

        // 2. 解方程式求 H
        vector<double> H = computeHomography4Points(srcSubset, dstSubset);
        if (H.empty())
            continue;

        // 3. 計算 Inliers
        int currentInliers = 0;
        for (size_t j = 0; j < srcPoints.size(); j++) {
            double x = srcPoints[j].x;
            double y = srcPoints[j].y;

            // Project x, y using H
            double w = H[6] * x + H[7] * y + H[8];
            if (abs(w) < 1e-9)
                continue;

            double Px = (H[0] * x + H[1] * y + H[2]) / w;
            double Py = (H[3] * x + H[4] * y + H[5]) / w;

            double dist = sqrt(pow(Px - dstPoints[j].x, 2) + pow(Py - dstPoints[j].y, 2));

            if (dist < threshold) {
                currentInliers++;
            }
        }

        if (currentInliers > maxInliers) {
            maxInliers = currentInliers;
            bestH = H;
        }
    }
    return bestH;
}

// ==========================================
// Part 3: Image Mapping (手刻 Perspective Transform)
// ==========================================

SimpleImage warpPerspectiveSimple(const SimpleImage& srcImg, const vector<double>& H, int outW, int outH) {
    SimpleImage dstImg(outW, outH);

    // 計算 Inverse H
    vector<double> H_inv;
    if (!invert3x3(H, H_inv)) {
        cerr << "Matrix inversion failed!" << endl;
        return dstImg;
    }

    // Sequential Loop - 這就是你要用 CUDA / OpenMP 加速的部分
    for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++) {
            // [u, v, w] = H_inv * [x, y, 1]
            // H_inv 是 row-major 的 1D array
            double w = H_inv[6] * x + H_inv[7] * y + H_inv[8];
            double u = H_inv[0] * x + H_inv[1] * y + H_inv[2];
            double v = H_inv[3] * x + H_inv[4] * y + H_inv[5];

            if (w != 0) {
                double src_x = u / w;
                double src_y = v / w;

                // Nearest Neighbor Interpolation
                int pixel_x = round(src_x);
                int pixel_y = round(src_y);

                if (pixel_x >= 0 && pixel_x < srcImg.w && pixel_y >= 0 && pixel_y < srcImg.h) {
                    dstImg.set(x, y, srcImg.get(pixel_x, pixel_y));
                }
            }
        }
    }
    return dstImg;
}

// ==========================================
// Main Function
// ==========================================

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: ./plane_projection img1.pgm img2.pgm\n");
        return -1;
    }

    // 1. ezSIFT 讀檔
    ezsift::Image<unsigned char> ezImg1, ezImg2;
    if (ezImg1.read_pgm(argv[1]) != 0 || ezImg2.read_pgm(argv[2]) != 0) {
        cerr << "Failed to read PGM images." << endl;
        return -1;
    }

    // 轉換成我們的 SimpleImage 結構 (方便後續 Mapping 操作)
    SimpleImage img1(ezImg1.w, ezImg1.h);
    memcpy(img1.data.data(), ezImg1.data, ezImg1.w * ezImg1.h);

    SimpleImage img2(ezImg2.w, ezImg2.h);
    memcpy(img2.data.data(), ezImg2.data, ezImg2.w * ezImg2.h);

    // 2. SIFT 特徵擷取與配對
    ezsift::double_original_image(true);
    list<ezsift::SiftKeypoint> kpt_list1, kpt_list2;

    cout << "Running SIFT..." << endl;
    auto start1 = std::chrono::high_resolution_clock::now();
    ezsift::sift_cpu(ezImg1, kpt_list1, true);
    auto end1 = std::chrono::high_resolution_clock::now();

    auto start2 = std::chrono::high_resolution_clock::now();
    ezsift::sift_cpu(ezImg2, kpt_list2, true);
    auto end2 = std::chrono::high_resolution_clock::now();

    list<ezsift::MatchPair> match_list;
    auto start3 = std::chrono::high_resolution_clock::now();
    ezsift::match_keypoints(kpt_list1, kpt_list2, match_list);
    auto end3 = std::chrono::high_resolution_clock::now();
    cout << "Matches found: " << match_list.size() << endl;

    // 3. 資料轉換 ezSIFT -> Point2f
    vector<Point2f> pts1, pts2;
    for (const auto& m : match_list) {
        pts1.push_back({m.c1, m.r1});  // Img1 (Source)
        pts2.push_back({m.c2, m.r2});  // Img2 (Dest/Ref)
    }

    // 4. RANSAC 計算 Homography (Img1 -> Img2)
    std::chrono::duration<double, std::milli> duration1 = end1 - start1;
    std::cout << "Time cost of SIFT detection for image 1: " << duration1.count() << " ms" << std::endl;
    std::chrono::duration<double, std::milli> duration2 = end2 - start2;
    std::cout << "Time cost of SIFT detection for image 2: " << duration2.count() << " ms" << std::endl;
    std::chrono::duration<double, std::milli> duration3 = end3 - start3;
    std::cout << "Time cost of matching keypoints: " << duration3.count() << " ms" << std::endl;

    cout << "Running RANSAC..." << endl;
    auto start = chrono::high_resolution_clock::now();
    vector<double> H = computeHomographyRANSAC(pts1, pts2, 1000, 1.0f);
    auto end = chrono::high_resolution_clock::now();
    cout << "RANSAC Time: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

    if (H.empty()) {
        cerr << "RANSAC failed!" << endl;
        return -1;
    }

    // 5. Image Mapping (Stitching)
    cout << "Stitching images..." << endl;
    int canvasW = img1.w + img2.w;
    int canvasH = max(img1.h, img2.h) * 1.5;

    // Step A: Warp Image 1
    SimpleImage panorama = warpPerspectiveSimple(img1, H, canvasW, canvasH);

    // Step B: Overlay Image 2 (Reference)
    // 簡單覆蓋: 如果 Img2 有像素值，就蓋過去
    for (int y = 0; y < img2.h; y++) {
        for (int x = 0; x < img2.w; x++) {
            unsigned char val = img2.get(x, y);
            if (val > 0) {  // 假設背景是全黑 0
                panorama.set(x, y, val);
            }
        }
    }

    // 6. 存檔
    writePGM("result_no_opencv.pgm", panorama);
    cout << "Done! Result saved to result_no_opencv.pgm" << endl;

    return 0;
}