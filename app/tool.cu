#include "tool.h"
#include "cuda.h"
#include "Timer.hpp"
#include "cuVector.cuh"
#include <iostream>
#include <math_constants.h>

#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;
typedef float Etype;

cudaha::cudaha(int rows, int cols) : rows(rows), cols(cols)
{

    A.create(rows, cols, CV_32FC1);
    A.setTo(std::numeric_limits<float>::quiet_NaN());
    Zf.create(rows, cols, CV_32FC1);
    Z.create(rows, cols, CV_32FC1);
    Zf.setTo(std::numeric_limits<float>::quiet_NaN());
    Z.setTo(std::numeric_limits<float>::quiet_NaN());

    cuA.create(rows, cols);
    cuZf.create(rows, cols);
    cuZ.create(rows, cols);

    // std::cout << rows << " "     << cols << std::endl;
}
cudaha::~cudaha()
{
    A.release();
    Z.release();
    Zf.release();

    cuZf.release();
    cuZ.release();
    cuA.release();

    pPoints.release();
    up.outindex.release();
    up.index.release();
};

void cudaha::updateA(cv::Mat &A, std::vector<cv::Point3f> &input_, Eigen::Vector4f global_min, float cell_size_)
{
    for (int i = 0; i < (int)input_.size(); ++i)
    {
        // ...then test for lower points within the cell
        auto &p = input_[i];
        // if (!pcl::isFinite(p))
        //     continue;
        if (p.z > 3)
            continue;
        int row = std::floor((p.y - global_min.y()) * cell_size_);
        int col = std::floor((p.x - global_min.x()) * cell_size_);
        //
        if (p.z < A.at<float>(row, col) || std::isnan(A.at<float>(row, col)))
        {
            A.at<float>(row, col) = p.z;
        }
    }
    cuA.upload(A.ptr<float>(), A.step);
}

__global__ void kernerl(int rows, int half_sizesval, int cols, const Patch<float> Zin, Patch<float> Zout)
{
    int row = blockIdx.x * 32 + threadIdx.x;
    int col = blockIdx.y * 32 + threadIdx.y;
    if (row >= rows || col >= cols)
        return;
    int rs, re;
    rs = ((row - half_sizesval) < 0) ? 0 : row - half_sizesval;
    re = ((row + half_sizesval) > (rows - 1)) ? (rows - 1) : row + half_sizesval;
    int cs, ce;
    cs = ((col - half_sizesval) < 0) ? 0 : col - half_sizesval;
    ce = ((col + half_sizesval) > (cols - 1)) ? (cols - 1) : col + half_sizesval;

    float min_coeff = CUDART_INF_F;
    for (int j = rs; j < (re + 1); ++j)
    {
        for (int k = cs; k < (ce + 1); ++k)
        {
            if (Zin(j, k) < min_coeff)
                min_coeff = Zin(j, k);
        }
    }
    if (min_coeff != CUDART_INF_F)
        Zout(row, col) = min_coeff;
}

__global__ void kernerl3(int rows, int half_sizesval, int cols, const Patch<float> Zin, Patch<float> Zout)
{
    int row = blockIdx.x * 32 + threadIdx.x;
    int col = blockIdx.y * 32 + threadIdx.y;
    if (row >= rows || col >= cols)
        return;
    int rs, re;
    rs = ((row - half_sizesval) < 0) ? 0 : row - half_sizesval;
    re = ((row + half_sizesval) > (rows - 1)) ? (rows - 1) : row + half_sizesval;

    int cs, ce;
    cs = ((col - half_sizesval) < 0) ? 0 : col - half_sizesval;
    ce = ((col + half_sizesval) > (cols - 1)) ? (cols - 1) : col + half_sizesval;

    float max_coeff = -CUDART_INF_F;
    for (int j = rs; j < (re + 1); ++j)
    {
        for (int k = cs; k < (ce + 1); ++k)
        {
            auto p = Zin(j, k);
            if (p > max_coeff) //找该范围最大值
                max_coeff = p;
        }
    }
    if (max_coeff != -CUDART_INF_F)
        Zout(row, col) = max_coeff;
}

__global__ void upindex(struct upindex up, CUVector<cv::Point3f> p)
{
    up(p);
}

void cudaha::update_idnex(std::vector<int> &ground, int half_sizes, float height_thresholds, std::vector<cv::Point3f> &___input_)
{
    if (pPoints.len == 0)
    {
        pPoints.create(___input_.size());
        pPoints.upload(___input_.data(), ___input_.size());
        up.outindex.create(ground.size());
        up.index.create(ground.size());
        up.index.upload(ground.data(), ground.size());
    }
    cudaMemset(up.num, 0, sizeof(uint32_t));

    up.max_height_ = max_height_;
    up.height_thresholds = height_thresholds;
    up.cell_size_ = cell_size_;
    up.min_p = make_float2(min_p[0], min_p[1]);

    up.cuA = cuA;
    upindex<<<(up.index.len / 1024 + 1), 1024>>>(up, pPoints);
    ck(cudaDeviceSynchronize());
    up.read();
    int cnt = 0;
    cudaMemcpy(up.index.devPtr, up.outindex.devPtr, up.h_num * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    up.index.len = up.h_num;
}
void cudaha::end(std::vector<int> &ground)
{
    ground.resize(up.h_num);
    up.outindex.download(ground.data(), ground.size());
}
void cudaha::GpuFindMaxMin(int rows, int half_sizesval, int cols, const cv::Mat &A, cv::Mat &Zf, cv::Mat &Z)
{
}
void cudaha::CpuFindMaxMin(int half_sizesval)
{

    dim3 grid((rows / 32 + 1), cols / 32 + 1, 1), block(32, 32, 1);
    {
        // Timer t("K1:");

        cuZ.upload(Z.ptr<float>(), Z.step);
        kernerl<<<grid, block>>>(rows, half_sizesval, cols, cuA, cuZ);
        ck(cudaDeviceSynchronize());
    }
    // std::cout << half_sizesval << std::endl;
    {
        // Timer t("K2:");
        kernerl3<<<grid, block>>>(rows, half_sizesval, cols, cuZ, cuZf);
        ck(cudaDeviceSynchronize());
    }

    cuZf.copyTo(cuA);
}
// for (int row = 0; row < rows; ++row)
// {
//     int rs, re;
//     rs = ((row - half_sizes[i]) < 0) ? 0 : row - half_sizes[i];
//     re = ((row + half_sizes[i]) > (rows - 1)) ? (rows - 1) : row + half_sizes[i];

//     for (int col = 0; col < cols; ++col)
//     {
//         int cs, ce;
//         cs = ((col - half_sizes[i]) < 0) ? 0 : col - half_sizes[i];
//         ce = ((col + half_sizes[i]) > (cols - 1)) ? (cols - 1) : col + half_sizes[i];

//         float max_coeff = -std::numeric_limits<float>::max();

//         for (int j = rs; j < (re + 1); ++j)
//         {
//             for (int k = cs; k < (ce + 1); ++k)
//             {
//                 assert(j < rows && k < cols);
//                 if (Z.at<float>(j, k) != std::numeric_limits<float>::quiet_NaN())
//                 {
//                     if (Z.at<float>(j, k) > max_coeff)
//                         max_coeff = Z.at<float>(j, k);
//                 }
//             }
//         }
//         if (max_coeff != -std::numeric_limits<float>::max())
//             Zf.at<float>(row, col) = max_coeff;
//     }
// }