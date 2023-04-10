#pragma once
#include <opencv2/opencv.hpp>
#include "cuda.h"
#include "eigen3/Eigen/Core"
#include "cuVector.cuh"

struct upindex
{
    upindex()
    {
        cudaMalloc(&num, sizeof(uint32_t));
    }
    float max_height_;
    float height_thresholds;
    float cell_size_;
    float2 min_p;
    uint32_t *num;
    void read()
    {
        cudaMemcpy(&h_num, num, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    uint32_t h_num;
    CUVector<int> index;
    CUVector<int> outindex;

    Patch<float> cuA;
    inline __device__ void operator()(CUVector<cv::Point3f> &ps)
    {
        int p_idx = threadIdx.x + blockIdx.x * 1024;
        if (p_idx >= index.len)
            return;
        // printf("%d\n", p_idx);
        const auto &p = ps[index[p_idx]]; // cloud->points[p_idx]; //(*input_)[ground[p_idx]];// (*cloud)[p_idx];
        if (p.z > max_height_)
            return;
        int ecol = __float2int_rd((p.x - min_p.x) * cell_size_);
        int erow = __float2int_rd((p.y - min_p.y) * cell_size_);

        float diff = p.z - cuA(erow, ecol);
        if (diff < height_thresholds)
        {
            unsigned int val = atomicInc(num, 0xffffff);
            outindex[val] = index[p_idx];
        }
    }
};

class cudaha
{
public:
    Patch<float> cuA;
    Patch<float> cuZ;
    Patch<float> cuZf;
struct upindex up;
    CUVector<int> cuHalf;
    CUVector<cv::Point3f> pPoints;

    cv::Mat A, Z, Zf;
    int rows, cols;
    float max_height_;
    float min_p[2];
    float cell_size_, height_thresholds;
    cudaha(int rows, int cols);
    void updateA(cv::Mat &A, std::vector<cv::Point3f> &input_, Eigen::Vector4f global_min, float cell_size_);
    void GpuFindMaxMin(int rows, int val, int cols, const cv::Mat &A, cv::Mat &Zf, cv::Mat &Z);
    void CpuFindMaxMin(int half_sizesval);
    cv::Mat memo;
    void update_idnex(std::vector<int> &ground, int half_sizes, float height_thresholds, std::vector<cv::Point3f> &___input_);
    void end(std::vector<int> &ground);
    ~cudaha();
};
