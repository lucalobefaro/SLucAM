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
#include <SLucAM_measurement.h>



//
// Loading and saving utilities
//
namespace SLucAM {
    bool load_image(const std::string& filename, cv::Mat& img, \
                    const cv::Mat& K=cv::Mat(), \
                    const cv::Mat& distorsion_coefficients=cv::Mat());
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Visualization utilities
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool visualize_matches(const Measurement& meas1, const Measurement& meas2, \
                        const std::vector<cv::DMatch>& matches, \
                        const std::vector<unsigned int>& matches_filter);
} // namespace SLucAM


#endif // SLUCAM_IMAGE_H