/**
 * Vibe-Check Core Motion Engine
 * 
 * High-performance optical flow heatmap generation using Farneback algorithm.
 */

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace py = pybind11;

class MotionEngine {
public:
    /**
     * Construct a MotionEngine with specified grid dimensions.
     * 
     * @param grid_h Number of vertical grid cells for heatmap
     * @param grid_w Number of horizontal grid cells for heatmap
     * @param use_gpu Whether to attempt GPU acceleration (if available)
     */
    MotionEngine(int grid_h = 8, int grid_w = 8, bool use_gpu = false)
        : grid_h_(grid_h), grid_w_(grid_w), use_gpu_(use_gpu),
          frame_count_(0), total_time_ms_(0.0) {
        
        if (grid_h_ <= 0 || grid_w_ <= 0) {
            throw std::invalid_argument("grid_h and grid_w must be positive.");
        }
        
        // Pre-allocate heatmap storage
        heatmap_data_.resize(grid_h_ * grid_w_, 0.0f);
    }

    /**
     * Process a BGR frame and return motion heatmap.
     * 
     * @param frame_bgr HxWx3 uint8 numpy array in BGR order
     * @return tuple of (heatmap ndarray, mean_energy float)
     */
    py::tuple process(py::array frame_bgr) {
        auto start = std::chrono::high_resolution_clock::now();
        
        py::buffer_info info = frame_bgr.request();

        // Validate input
        if (info.ndim != 3) {
            throw std::invalid_argument("frame must be a HxWx3 numpy array.");
        }
        if (info.shape[2] != 3) {
            throw std::invalid_argument("frame must have 3 channels (BGR).");
        }
        if (info.format != py::format_descriptor<uint8_t>::format()) {
            throw std::invalid_argument("frame must be dtype=uint8.");
        }

        const int h = static_cast<int>(info.shape[0]);
        const int w = static_cast<int>(info.shape[1]);
        const int stride0 = static_cast<int>(info.strides[0]);
        const int stride1 = static_cast<int>(info.strides[1]);
        const int stride2 = static_cast<int>(info.strides[2]);

        if (stride2 != 1) {
            throw std::invalid_argument("Unexpected memory layout for frame.");
        }

        // Wrap numpy buffer into cv::Mat without copy
        cv::Mat bgr(h, w, CV_8UC3, info.ptr, static_cast<size_t>(stride0));

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

        // If first frame, store and return zeros
        if (prev_gray_.empty()) {
            prev_gray_ = gray.clone();
            return make_zero_result();
        }

        // Compute optical flow using Farneback algorithm
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(
            prev_gray_, gray, flow,
            0.5,// pyr_scale - scale between pyramid levels
            3,// levels - number of pyramid levels
            15,// winsize - averaging window size
            3,// iterations - per pyramid level
            5,// poly_n - size of pixel neighborhood
            1.2,// poly_sigma - Gaussian std for derivatives
            0// flags
        );

        // Store current frame for next iteration
        prev_gray_ = gray.clone();

        // Compute heatmap by averaging magnitude within grid cells
        py::array_t<float> heatmap({grid_h_, grid_w_});
        auto hm = heatmap.mutable_unchecked<2>();

        const int cell_h = std::max(1, h / grid_h_);
        const int cell_w = std::max(1, w / grid_w_);

        double total_energy = 0.0;
        int total_cells = 0;

        for (int gy = 0; gy < grid_h_; gy++) {
            const int y0 = gy * cell_h;
            const int y1 = (gy == grid_h_ - 1) ? h : (gy + 1) * cell_h;

            for (int gx = 0; gx < grid_w_; gx++) {
                const int x0 = gx * cell_w;
                const int x1 = (gx == grid_w_ - 1) ? w : (gx + 1) * cell_w;

                double sum_mag = 0.0;
                int count = 0;

                // Sum magnitudes in this cell
                for (int y = y0; y < y1; y++) {
                    const cv::Point2f* row = flow.ptr<cv::Point2f>(y);
                    for (int x = x0; x < x1; x++) {
                        const float fx = row[x].x;
                        const float fy = row[x].y;
                        sum_mag += std::sqrt(fx * fx + fy * fy);
                        count++;
                    }
                }

                float avg_mag = (count > 0) ? static_cast<float>(sum_mag / count) : 0.0f;
                
                // Apply temporal smoothing
                const float alpha = 0.3f;  // Smoothing factor
                float prev_val = heatmap_data_[gy * grid_w_ + gx];
                float smoothed = alpha * avg_mag + (1.0f - alpha) * prev_val;
                
                hm(gy, gx) = smoothed;
                heatmap_data_[gy * grid_w_ + gx] = smoothed;

                total_energy += smoothed;
                total_cells++;
            }
        }

        float mean_energy = (total_cells > 0) 
            ? static_cast<float>(total_energy / total_cells) 
            : 0.0f;

        // Update stats
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        frame_count_++;
        total_time_ms_ += elapsed_ms;

        return py::make_tuple(heatmap, mean_energy);
    }

    /**
     * Get flow vectors for visualization (optional).
     * Returns Nx4 array of [x, y, dx, dy] for sampled flow vectors.
     */
    py::array_t<float> get_flow_vectors(int sample_step = 16) const {
        if (prev_gray_.empty()) {
            return py::array_t<float>({0, 4});
        }
        
        // This would require storing the last flow field
        // Simplified: return empty for now
        return py::array_t<float>({0, 4});
    }

    /**
     * Reset the engine state (clears previous frame).
     */
    void reset() {
        prev_gray_.release();
        std::fill(heatmap_data_.begin(), heatmap_data_.end(), 0.0f);
        frame_count_ = 0;
        total_time_ms_ = 0.0;
    }

    /**
     * Get processing statistics.
     */
    py::dict get_stats() const {
        py::dict stats;
        stats["frame_count"] = frame_count_;
        stats["total_time_ms"] = total_time_ms_;
        stats["avg_time_ms"] = (frame_count_ > 0) 
            ? total_time_ms_ / frame_count_ 
            : 0.0;
        stats["grid_size"] = py::make_tuple(grid_h_, grid_w_);
        return stats;
    }

private:
    int grid_h_;
    int grid_w_;
    bool use_gpu_;
    cv::Mat prev_gray_;
    std::vector<float> heatmap_data_;  // For temporal smoothing
    
    // Stats
    uint64_t frame_count_;
    double total_time_ms_;

    py::tuple make_zero_result() {
        py::array_t<float> heatmap({grid_h_, grid_w_});
        auto hm = heatmap.mutable_unchecked<2>();
        for (int i = 0; i < grid_h_; i++) {
            for (int j = 0; j < grid_w_; j++) {
                hm(i, j) = 0.0f;
            }
        }
        return py::make_tuple(heatmap, 0.0f);
    }
};

// Python module definition
PYBIND11_MODULE(vibe_core, m) {
    m.doc() = "Vibe-Check core motion engine (C++ optical flow heatmap generation)";
    
    py::class_<MotionEngine>(m, "MotionEngine")
        .def(py::init<int, int, bool>(), 
             py::arg("grid_h") = 8, 
             py::arg("grid_w") = 8,
             py::arg("use_gpu") = false,
             "Create a MotionEngine with specified grid dimensions")
        .def("process", &MotionEngine::process, 
             py::arg("frame_bgr"),
             "Process a BGR frame and return (heatmap, mean_energy)")
        .def("reset", &MotionEngine::reset,
             "Reset engine state (clears previous frame)")
        .def("get_stats", &MotionEngine::get_stats,
             "Get processing statistics");
    
    m.attr("__version__") = "1.0.0";
}