//
// SLucAM_initialization.h
//
// Description.
//


#ifndef SLUCAM_INITIALIZATION_H
#define SLUCAM_INITIALIZATION_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>
#include <SLucAM_measurement.h>
#include <SLucAM_state.h>



// -----------------------------------------------------------------------------
// Main initialization functions
// -----------------------------------------------------------------------------
namespace SLucAM {

    bool initialize(const Measurement& meas1, \
                    const Measurement& meas2, \
                    Matcher& matcher, \
                    const cv::Mat& K, cv::Mat& predicted_pose, \
                    std::vector<cv::DMatch>& matches, \
                    std::vector<unsigned int>& matches_filter, \
                    std::vector<cv::Point3f>& triangulated_points, \
                    const float& parallax_threshold=1.0, \
                    const bool verbose=false);

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to compute Essential and Homography
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    const float compute_essential(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const cv::Mat K, \
                                    const std::vector<cv::DMatch>& matches, \
                                    std::vector<unsigned int>& matches_filter, \
                                    cv::Mat& E, \
                                    const float& inliers_threshold=1);
    
    const float compute_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const cv::Mat K, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const float& inliers_threshold=3);

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions that evaluates the "goodness" of a matrix F or H
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    float evaluate_fundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const cv::Mat& F, \
                                const float& inliers_threshold);
    
    float evaluate_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const cv::Mat& H, \
                                const float& inliers_threshold);

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to compute the pose from the Essential Matrix
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool extract_X_from_E(const std::vector<cv::KeyPoint>& p_img1, \
                            const std::vector<cv::KeyPoint>& p_img2, \
                            const std::vector<cv::DMatch>& matches, \
                            std::vector<unsigned int>& matches_filter, \
                            const cv::Mat& F, const cv::Mat& K, \
                            cv::Mat& X);
    unsigned int compute_transformation_inliers(const std::vector<cv::KeyPoint>& p_img1, \
                                                const std::vector<cv::KeyPoint>& p_img2, \
                                                const std::vector<cv::DMatch>& matches, \
                                                const std::vector<unsigned int>& matches_filter, \
                                                std::vector<unsigned int>& matches_inliers, \
                                                const cv::Mat& R, const cv::Mat& t, \
                                                const cv::Mat& K);
} // namespace SLucAM


#endif // SLUCAM_INITIALIZATION_H