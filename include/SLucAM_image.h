//
// SLucAM_image.h
//
// In this module are defined all the functions to deal with images, included
// functions to extract features from them and find associations between 
// keypoints.
//


#ifndef SLUCAM_IMAGE_H
#define SLUCAM_IMAGE_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/features2d.hpp>



//
// Loading and saving utilities
//
namespace SLucAM {
    bool load_image(const std::string& filename, cv::Mat& img);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Visualization utilities
// -----------------------------------------------------------------------------
namespace SLucAM {
    void visualize_matches(const cv::Mat& img1, const cv::Mat& img2, \
                        const std::vector<cv::KeyPoint>& keypoints1, \
                        const std::vector<cv::KeyPoint>& keypoints2, \
                        const std::vector<cv::DMatch>& matches, \
                        const std::string image_name = "Matches");
} // namespace SLucAM


#endif // SLUCAM_IMAGE_H