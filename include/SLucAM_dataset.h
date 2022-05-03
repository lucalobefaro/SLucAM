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
#include <SLucAM_image.h>



// -----------------------------------------------------------------------------
// Functions to deal with my personal dataset format
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool load_my_dataset(const std::string& dataset_folder, State& state, \
                            const cv::Ptr<cv::Feature2D>& detector, \
                            const bool verbose=false);
    bool load_camera_matrix(const std::string& filename, cv::Mat& K, \
                            cv::Mat& distorsion_coefficients);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to deal with the TUM Dataset
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool load_TUM_dataset(const std::string& dataset_folder, State& state, \
                            FeatureExtractor& feature_extractor, \
                            const bool verbose=false);
    bool load_TUM_camera_matrix(const std::string& filename, cv::Mat& K, \
                                cv::Mat& distorsion_coefficients);
    bool save_TUM_results(const std::string& dataset_folder, const State& state);
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to deal with the Pering Laboratory Dataset
// -----------------------------------------------------------------------------
namespace SLucAM {
    bool load_PRD_dataset(const std::string& dataset_folder, State& state);
    bool load_PRD_camera_matrix(const std::string& filename, cv::Mat& K);
    bool save_keypoints_PRD(const std::string& filename, \
                            const std::vector<cv::KeyPoint>& points, \
                            const cv::Mat& descriptors);
    bool load_keypoints_PRD(const std::string& filename, \
                            std::vector<cv::KeyPoint>& points, \
                            cv::Mat& descriptors);
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
    bool save_current_state(const std::string& folder, \
                            const State& state);
    bool save_poses(const std::string& folder, \
                    const std::vector<cv::Mat>& poses);
    bool save_landmarks(const std::string& folder, \
                        const std::vector<cv::Point3f>& landmarks);
    bool save_edges(const std::string& folder, \
                    const Keyframe& keyframe);
    bool save_keypoints(const std::string& folder, \
                        const std::vector<cv::KeyPoint>& points);
} // namespace SLucAM



#endif // SLUCAM_STATE_H