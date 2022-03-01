//
// SLucAM_geometry.h
//
// In this file are defined all the functions to deal with geomtry for SLAM.
//


#ifndef SLUCAM_GEOMETRY_H
#define SLUCAM_GEOMETRY_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>
#include <SLucAM_state.h>



// -----------------------------------------------------------------------------
// Basic geometric functions
// -----------------------------------------------------------------------------
namespace SLucAM {
    void normalize_points(const std::vector<cv::KeyPoint>& points, \
                            std::vector<cv::KeyPoint>& normalized_points, \
                            cv::Mat& T);
    
    unsigned int triangulate_points(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const std::vector<unsigned int>& idxs, \
                                    const cv::Mat& X, const cv::Mat& K,
                                    std::vector<cv::Point3f>& triangulated_points);
    void apply_perturbation_Tmatrix(const cv::Mat& perturbation, \
                                    cv::Mat& T_matrix, const unsigned int& starting_idx);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Multi-view geometry functions
// -----------------------------------------------------------------------------
namespace SLucAM {
    void estimate_foundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const std::vector<unsigned int>& idxs, \
                                cv::Mat& F);
    void estimate_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const std::vector<unsigned int>& idxs, \
                                cv::Mat& H);
    void extract_X_from_F(const std::vector<cv::KeyPoint>& p_img1, \
                            const std::vector<cv::KeyPoint>& p_img2, \
                            const std::vector<cv::DMatch>& matches, \
                            const std::vector<unsigned int>& idxs, \
                            const cv::Mat& F, const cv::Mat& K, \
                            cv::Mat& X, \
                            std::vector<cv::Point3f>& triangulated_points);
} // namespace SLucAM



#endif // SLUCAM_GEOMETRY_H