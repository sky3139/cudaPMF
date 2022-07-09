#pragma once
#include <opencv2/opencv.hpp>
#include "cuda.h"
#include "eigen3/Eigen/Core"
#include "cuVector.cuh"

class cudaha
{
public:
    Patch<float> cuA;
    Patch<float> cuZ;
    Patch<float> cuZf;

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
