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



#endif // SLUCAM_STATE_H