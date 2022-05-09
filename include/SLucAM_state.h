//
// SLucAM_state.h
//
// In this module we have all the function to deal with the state and the
// state class itself.
//


#ifndef SLUCAM_STATE_H
#define SLUCAM_STATE_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>
#include <SLucAM_keyframe.h>
#include <SLucAM_measurement.h>
#include <SLucAM_matcher.h>
#include <SLucAM_keypoint.h>
#include <map>
#include <iostream>



// -----------------------------------------------------------------------------
// State class
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This class represent the state of the system. It mantains all the poses
    * accumulated so far and all the triangulated "landmarks" positions.
    */
    class State {
    
    public:

        State();

        State(cv::Mat& K, std::vector<Measurement>& measurements, \
            const unsigned int expected_poses, \
            const unsigned int expected_landmarks);
        
        State(cv::Mat& K, cv::Mat& distorsion_coefficients, \
            std::vector<Measurement>& measurements, \
            const unsigned int expected_poses, \
            const unsigned int expected_landmarks);
        
        bool initializeState(Matcher& matcher, \
                            const unsigned int n_iters_ransac, \
                            const float& parallax_threshold=1.0, \
                            const bool verbose=false);
        
        bool integrateNewMeasurement(Matcher& matcher, \
                                        const bool& triangulate_new_points, \
                                        const unsigned int& local_map_size, \
                                        const float& kernel_threshold_POSIT, \
                                        const float& inliers_threshold_POSIT, \
                                        const float& parallax_threshold, \
                                        const float& new_landmark_threshold, \
                                        const bool verbose);
        
        void performTotalBA(const unsigned int& n_iters, const bool verbose=false);

        void performLocalBA(const unsigned int& n_iters, const bool verbose=false);

        const unsigned int reaminingMeasurements() const {
            return (this->_measurements.size() - this->_next_measurement_idx);
        };

        void addKeyFrame(const unsigned int& meas_idx, const unsigned int& pose_idx, \
                        std::vector<std::pair<unsigned int, unsigned int>>& _points_associations, \
                        const int& observer_keyframe_idx, const bool verbose=false);

        void getCommonKeypoints(const unsigned int& k1_idx, \
                                const unsigned int& k2_idx, \
                                std::vector<unsigned int>& common_keypoints_ids);

        void getLocalMap(const unsigned int& keyframe_idx, \
                            std::vector<unsigned int>& observed_keypoints, \
                            std::vector<unsigned int>& near_local_keyframes, \
                            std::vector<unsigned int>& far_local_keyframes);

        const cv::Mat& getCameraMatrix() const \
            {return this->_K;};
        
        const cv::Mat& getDistorsionCoefficients() const \
            {return this->_distorsion_coefficients;};

        const Measurement& getNextMeasurement() \
            {return this->_measurements[this->_next_measurement_idx++];};
        
        const std::vector<Measurement>& getMeasurements() const \
            {return this->_measurements;};

        const std::vector<cv::Mat>& getPoses() const \
            {return this->_poses;};

        const std::vector<Keypoint>& getKeypoints() const \
            {return this->_keypoints;};

        const std::vector<Keyframe>& getKeyframes() const \
            {return this->_keyframes;};
    
    private: 
        
        static bool predictPose(cv::Mat& guessed_pose, \
                                const Measurement& meas_to_predict, \
                                std::vector<std::pair<unsigned int, unsigned int>>& \
                                        points_associations, \
                                Matcher& matcher, \
                                const std::vector<Keyframe>& keyframes, \
                                const std::vector<Keypoint>& keypoints, \
                                const std::vector<Measurement>& measurements, \
                                const std::vector<cv::Mat>& poses, \
                                const cv::Mat& K, \
                                const float& kernel_threshold_POSIT, \
                                const float& inliers_threshold_POSIT, \
                                const bool verbose=false);

        static void triangulateNewPoints(std::vector<Keyframe>& keyframes, \
                                        std::vector<Keypoint>& keypoints, \
                                        const std::vector<Measurement>& measurements, \
                                        const std::vector<cv::Mat>& poses, \
                                        Matcher& matcher, \
                                        const cv::Mat& K, \
                                        const unsigned int& triangulation_window, \
                                        const float& new_landmark_threshold, \
                                        const float& parallax_threshold, \
                                        const bool verbose=false);
        
        static void addAssociationKeypoints(std::vector<Keypoint>& keypoints, \
                                            const std::vector<std::pair<unsigned int, unsigned int>>& \
                                                    points_associations, \
                                            const unsigned int keyframe_idx, \
                                            const std::vector<Keyframe>& keyframes, \
                                            const std::vector<Measurement>& measurements);

        static void associateNewKeypoints(const std::vector<cv::Point3f>& predicted_landmarks, \
                                        const std::vector<cv::DMatch>& matches, \
                                        const std::vector<unsigned int>& matches_filter, \
                                        std::vector<Keypoint>& keypoints, \
                                        std::vector<std::pair<unsigned int, \
                                                unsigned int>>& meas1_points_associations, \
                                        std::vector<std::pair<unsigned int, \
                                                unsigned int>>& meas2_points_associations, \
                                        const bool verbose=false);
        
        static bool containsLandmark(const std::vector<std::pair<unsigned int, \
                                        unsigned int>>& points_associations, \
                                        const unsigned int& landmark_idx);

        static void projectAssociations(const Measurement& meas, \
                                        const cv::Mat& T, const cv::Mat& K, \
                                        const std::vector<Keypoint>& keypoints, \
                                        const std::vector<Keyframe>& keyframes, \
                                        const std::vector<Measurement>& measurements, \
                                        std::vector<std::pair<unsigned int, unsigned int>>& \
                                                points_associations);

        // Camera matrix and distorsion coefficients
        cv::Mat _K;
        cv::Mat _distorsion_coefficients;

        // The vector containing all the measurements, ordered by time
        std::vector<Measurement> _measurements;
     
        // The vector containing all the poses, ordered by time (poses of
        // the world wrt cameras)
        std::vector<cv::Mat> _poses;

        // The vector containing all the triangulated points, ordered
        // by insertion
        std::vector<Keypoint> _keypoints;

        // This vector contains all the keyframe
        std::vector<Keyframe> _keyframes;

        // Reference to the next measurement to analyze
        unsigned int _next_measurement_idx;

    };

} // namespace SLucAM



#endif // SLUCAM_STATE_H