//
// SLucAM_dataset.h
//
// In this module we have all the function to deal with dataset of
// different format.
//


#ifndef SLUCAM_DATASET_H
#define SLUCAM_DATASET_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <string>
#include <opencv2/features2d.hpp>
#include <SLucAM_state.h>



// -----------------------------------------------------------------------------
// Functions to deal with my personal dataset format
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool load_my_dataset(const std::string& dataset_folder, State& state);
    bool load_camera_matrix(const std::string& filename, cv::Mat& K);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to deal with the Pering Laboratory Dataset
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool load_PRD_dataset(const std::string& dataset_folder, State& state);
    bool load_PRD_camera_matrix(const std::string& filename, cv::Mat& K);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to deal with my Synthetic Dataset
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool load_synthetic_dataset(const std::string& dataset_folder, State& state, \
                                std::vector<std::vector<unsigned int>>& associations);
    bool load_synthetic_camera_matrix(const std::string& filename, cv::Mat& K);
    bool load_3dpoints_ground_truth(const std::string& filename, \
                                    std::vector<cv::Point3f>& gt_points);
    float test_predicted_points(const std::string& dataset_folder, \
                                const std::vector<Keyframe>& keyframes, \
                                const std::vector<cv::Point3f>& predicted_points, \
                                const std::vector<std::vector<unsigned int>>& associations);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to save and load general infos on files
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool save_keypoints_on_file(const std::string& filename, \
                                const std::vector<cv::KeyPoint>& points, \
                                const cv::Mat& descriptors);
    bool load_keypoints_from_file(const std::string& filename, \
                                std::vector<cv::KeyPoint>& points, \
                                cv::Mat& descriptors);
} // namespace SLucAM



#endif // SLUCAM_STATE_H