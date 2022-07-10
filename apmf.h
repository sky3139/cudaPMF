

#pragma once

#include <pcl/pcl_base.h>
#include <pcl/search/search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/filters/morphological_filter.h>
#include <pcl/filters/extract_indices.h>
#include <opencv2/opencv.hpp>
#include "tool.h"
#include "Timer.hpp"

#define DEBUG
namespace pcl
{
  template <typename PointT>
  class Apmf : public pcl::PCLBase<PointT>
  {
  public:
    ~Apmf(){};

    /** \brief Get the maximum window size to be used in filtering ground returns. */
    inline int
    getMaxWindowSize() const { return (max_window_size_); }
    /** \brief Set the maximum window size to be used in filtering ground returns. */
    inline void
    setMaxWindowSize(int max_window_size) { max_window_size_ = max_window_size; }

    /** \brief Get the slope value to be used in computing the height threshold. */
    inline float
    getSlope() const { return (slope_); }

    /** \brief Set the slope value to be used in computing the height threshold. */
    inline void
    setSlope(float slope) { slope_ = slope; }

    /** \brief Get the maximum height above the parameterized ground surface to be considered a ground return. */
    inline float
    getMaxDistance() const { return (max_distance_); }

    /** \brief Set the maximum height above the parameterized ground surface to be considered a ground return. */
    inline void
    setMaxDistance(float max_distance) { max_distance_ = max_distance; }

    /** \brief Get the initial height above the parameterized ground surface to be considered a ground return. */
    inline float
    getInitialDistance() const { return (initial_distance_); }

    /** \brief Set the initial height above the parameterized ground surface to be considered a ground return. */
    inline void
    setInitialDistance(float initial_distance) { initial_distance_ = initial_distance; }

    /** \brief Get the cell size. */
    inline float
    getCellSize() const { return (1.0f / cell_size_); }

    /** \brief Set the cell size. */
    inline void
    setCellSize(float cell_size) { cell_size_ = 1.0f / cell_size; }

    /** \brief Get the base to be used in computing progressive window sizes. */
    inline float
    getBase() const { return (base_); }

    /** \brief Set the base to be used in computing progressive window sizes. */
    inline void
    setBase(float base) { base_ = base; }

    /** \brief Get flag indicating whether or not to exponentially grow window sizes? */
    inline bool
    getExponential() const { return (exponential_); }

    /** \brief Set flag indicating whether or not to exponentially grow window sizes? */
    inline void
    setExponential(bool exponential) { exponential_ = exponential; }

    /** \brief Initialize the scheduler and set the number of threads to use.
     * \param nr_threads the number of hardware threads to use (0 sets the value back to automatic)
     */
    inline void
    setNumberOfThreads(unsigned int nr_threads = 0) { threads_ = nr_threads; }

    /** \brief This method launches the segmentation algorithm and returns indices of
     * points determined to be ground returns.
     * \param[out] ground indices of points determined to be ground returns.
     */
    virtual void
    extract(Indices &ground, std::vector<PointT> &a_input)
    {

      // Compute the series of window sizes and height thresholds
      std::vector<float> height_thresholds;
      std::vector<float> window_sizes;
      std::vector<int> half_sizes;
      int iteration = 0;
      float window_size = 0.0f;
      while (window_size < max_window_size_)
      {
        // Determine the initial window size.
        int half_size = (exponential_) ? (static_cast<int>(std::pow(static_cast<float>(base_), iteration))) : ((iteration + 1) * base_);
        window_size = 2 * half_size + 1;
        // Calculate the height threshold to be used in the next iteration.
        float height_threshold = (iteration == 0) ? (initial_distance_) : (slope_ * (window_size - window_sizes[iteration - 1]) * cell_size_ + initial_distance_);
        // Enforce max distance on height threshold
        if (height_threshold > max_distance_)
          height_threshold = max_distance_;
        half_sizes.push_back(half_size);
        window_sizes.push_back(window_size);
        height_thresholds.push_back(height_threshold);
        iteration++;
      }
      // setup grid based on scale and extents
      std::vector<cv::Point3f> ___input_;
      Eigen::Array4f min_p, max_p;
      min_p.setConstant(FLT_MAX);
      max_p.setConstant(-FLT_MAX);
      // for (auto &p :input_->points)
      // std::cout << a_input.size() << std::endl;
      for (auto &p : a_input)
      {
        if (!std::isfinite(p.x) ||
            !std::isfinite(p.y) ||
            !std::isfinite(p.z))
        {
          ___input_.push_back(cv::Point3f(12.103020, -0.738641, 100));
          continue;
        }
        // std:: cout << p <<  std::endl;
        const auto pt = p.getArray4fMap();
        min_p = min_p.min(pt);
        max_p = max_p.max(pt);
        ___input_.push_back(cv::Point3f(p.x, p.y, p.z));
      }
      float xextent = max_p.x() - min_p.x();
      float yextent = max_p.y() - min_p.y();
      int rows = static_cast<int>(std::floor(yextent * cell_size_) + 1);
      int cols = static_cast<int>(std::floor(xextent * cell_size_) + 1);
      cudaha pc(rows, cols);
      pc.updateA(pc.A, ___input_, min_p, cell_size_);
      ground.resize(___input_.size()); // *indices_;
      std::iota(ground.begin(), ground.end(), 0);
      pc.max_height_ = max_height_;
      pc.cell_size_ = cell_size_;
      pc.min_p[0] = min_p.x();
      pc.min_p[1] = min_p.y();
      // t.PrintSeconds("window_size");
      {

        for (std::size_t i = 0; i < window_sizes.size(); ++i)
        {

          pc.height_thresholds = height_thresholds[i];
          pc.CpuFindMaxMin(half_sizes[i]);
          pc.update_idnex(ground, half_sizes[i], height_thresholds[i], ___input_);
        }
      }
      pc.end(ground);
      return;
    };

  public:
    /** \brief Maximum window size to be used in filtering ground returns. */
    int max_window_size_;

    /** \brief Slope value to be used in computing the height threshold. */
    float slope_;

    /** \brief Maximum height above the parameterized ground surface to be considered a ground return. */
    float max_distance_;

    /** \brief Initial height above the parameterized ground surface to be considered a ground return. */
    float initial_distance_;

    /** \brief Cell size. */
    float cell_size_;

    /** \brief Base to be used in computing progressive window sizes. */
    float base_;

    /** \brief Exponentially grow window sizes? */
    bool exponential_;

    /** \brief Number of threads to be used. */
    unsigned int threads_;
    float max_height_;

  public:
    /** \brief Constructor that sets default values for member variables. */
    Apmf() : max_window_size_(33),
             slope_(0.7f),
             max_distance_(10.0f),
             initial_distance_(0.15f),
             cell_size_(1.0f / 1.0f),
             base_(2.0f),
             exponential_(true),
             threads_(0),
             max_height_(2.0f) {}
  };
}

#ifdef PCL_NO_PRECOMPILE
#endif
