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
                    std::vector<cv::Point3f>& triangulated_points, \
                    std::vector<std::pair<unsigned int, unsigned int>>& \
                        meas1_points_associations, \
                    std::vector<std::pair<unsigned int, unsigned int>>& \
                        meas2_points_associations, \
                    const unsigned int& ransac_iter=200, \
                    const float& rotation_only_threshold_rate=5);

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions that uses RANSAC to compute F and H 
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    int ransac_foundamental(const std::vector<cv::KeyPoint>& p_img1_normalized, \
                            const std::vector<cv::KeyPoint>& p_img2_normalized, \
                            const std::vector<cv::KeyPoint>& p_img1, \
                            const std::vector<cv::KeyPoint>& p_img2, \
                            const std::vector<cv::DMatch>& matches, \
                            const cv::Mat& T1, const cv::Mat& T2, \
                            const std::vector<std::vector<unsigned int>>& rand_idxs, \
                            std::vector<bool>& inliers_mask, \
                            const unsigned int n_iter=200, \
                            const float inliers_threshold=3.84);
    
    int ransac_homography(const std::vector<cv::KeyPoint>& p_img1_normalized, \
                            const std::vector<cv::KeyPoint>& p_img2_normalized, \
                            const std::vector<cv::KeyPoint>& p_img1, \
                            const std::vector<cv::KeyPoint>& p_img2, \
                            const std::vector<cv::DMatch>& matches, \
                            const cv::Mat& T1, const cv::Mat& T2, \
                            const std::vector<std::vector<unsigned int>>& rand_idxs, \
                            std::vector<bool>& inliers_mask, \
                            const unsigned int n_iter=200, \
                            const float inliers_threshold=5.99);

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions that evaluates the "goodness" of a matrix F or H
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    unsigned int evaluate_foundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                        const std::vector<cv::KeyPoint>& p_img2, \
                                        const std::vector<cv::DMatch>& matches, \
                                        const cv::Mat& F, \
                                        const float& inliers_threshold);
    
    unsigned int evaluate_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                        const std::vector<cv::KeyPoint>& p_img2, \
                                        const std::vector<cv::DMatch>& matches, \
                                        const cv::Mat& H, \
                                        const float& inliers_threshold);

} // namespace SLucAM


#endif // SLUCAM_INITIALIZATION_H