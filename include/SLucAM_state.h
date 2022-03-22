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
#include <SLucAM_measurement.h>
#include <SLucAM_matcher.h>
#include <map>



// -----------------------------------------------------------------------------
// Keyframe class
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    /*
    * This class take note of each KeyFrame in the state. Basically it contains
    * informations about:
    *   - which measurement in the state is the measurement from which
    *       this keyframe is taken
    *   - which pose in the state is the predicted pose for this keyframe
    *   - a vector of Map <point_idx, landmark_idx> that associates at each point
    *       in the measurement to which this keyframe refers, the 3D 
    *       predicted landmark in the state
    *   - a vector that contains all the keyframe(poses) that the current Keyframe 
    *       observes
    */
    class Keyframe {

    public:

        Keyframe(const unsigned int& meas_idx, \
                const unsigned int& pose_idx, \
                std::vector<std::pair<unsigned int, unsigned int>>& points_associations, \
                const unsigned int& n_points_meas) {
            this->_meas_idx = meas_idx;
            this->_pose_idx = pose_idx;
            this->_points_associations = points_associations;
            this->_points_associations.reserve(n_points_meas);
        }

        void addPointAssociation(const unsigned int& point_idx, \
                                const unsigned int& landmark_idx){
            this->_points_associations.emplace_back(point_idx, landmark_idx);
        }

        void addKeyframeAssociation(const unsigned int& pose_idx){
            this->_keyframes_associations.emplace_back(pose_idx);
        }

        const unsigned int& getPoseIdx() const {return this->_pose_idx;}

        const unsigned int& getMeasIdx() const {return this->_meas_idx;}

        const std::vector<std::pair<unsigned int, unsigned int>>& \
                    getPointsAssociations() const {
            return this->_points_associations;
        }

        const std::vector<unsigned int>& getKeyframesAssociations() const {
            return this->_keyframes_associations;
        }
        

    private:

        unsigned int _meas_idx;
        unsigned int _pose_idx;
        std::vector<std::pair<unsigned int, unsigned int>> _points_associations;
        std::vector<unsigned int> _keyframes_associations;

    };

} // namespace SLucAM



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

        State() {this->_next_measurement_idx=0;};

        State(cv::Mat& K, std::vector<Measurement>& measurements, \
            const unsigned int expected_poses, \
            const unsigned int expected_landmarks);
        
        bool initializeState(Matcher& matcher, \
                            const unsigned int& ransac_iter=200, \
                            const float& rotation_only_threshold_rate=5);
        
        bool updateState(Matcher& matcher, \
                        const bool& triangulate_new_points);
        
        void performBundleAdjustment(const float& n_iterations, \
                                        const float& damping_factor, \
                                        const float& kernel_threshold, \
                                        const float& threshold_to_ignore);

        const unsigned int reaminingMeasurements() const {
            return (this->_measurements.size() - this->_next_measurement_idx);
        };

        void addKeyFrame(const unsigned int& meas_idx, const unsigned int& pose_idx, \
                        std::vector<std::pair<unsigned int, unsigned int>>& _points_associations, \
                        const unsigned int& observer_keyframe_idx);

        const cv::Mat& getCameraMatrix() const \
            {return this->_K;};

        const Measurement& getNextMeasurement() \
            {return this->_measurements[this->_next_measurement_idx++];};
        
        const std::vector<Measurement>& getMeasurements() const \
            {return this->_measurements;};

        const std::vector<cv::Mat>& getPoses() const \
            {return this->_poses;};

        const std::vector<cv::Point3f>& getLandmarks() const \
            {return this->_landmarks;};

        const std::vector<Keyframe>& getKeyframes() const \
            {return this->_keyframes;};
    
    private:

        void boxPlus(cv::Mat& dx);
    
        static unsigned int buildLinearSystemProjections(\
                        const std::vector<cv::Mat>& poses, \
                        const std::vector<cv::Point3f>& landmarks, \
                        const std::vector<Measurement>& measurements, \
                        const std::vector<Keyframe>& keyframes, \
                        const cv::Mat& K, \
                        cv::Mat& H, cv::Mat& b, \
                        float& chi_tot, \
                        const float& kernel_threshold, \
                        const float& threshold_to_ignore, \
                        const float& img_rows, \
                        const float& img_cols);
        
        static unsigned int buildLinearSystemPoses(\
                        const std::vector<cv::Mat>& poses, \
                        const std::vector<Keyframe>& keyframes, \
                        const std::vector<cv::Mat>& poses_measurements, \
                        cv::Mat& H, cv::Mat& b, \
                        float& chi_tot, \
                        const float& kernel_threshold);

        static bool computeProjectionErrorAndJacobian(const cv::Mat& pose, \
                        const cv::Point3f& landmark_pose, \
                        const cv::KeyPoint& img_point, const cv::Mat& K, \
                        cv::Mat& J_pose, cv::Mat& J_landmark, cv::Mat& error, \
                        const float& img_rows, const float& img_cols);
      
        static void computePoseErrorAndJacobian(const cv::Mat& pose_1, \
                        const cv::Mat& pose_2, \
                        const cv::Mat& pose2_wrt_pose1, \
                        cv::Mat& J_2, cv::Mat& error);
            
        static inline unsigned int poseMatrixIdx(const unsigned int&  idx) {
            return idx*6;
        };

        static inline unsigned int landmarkMatrixIdx(const unsigned int&  idx, \
                                                    const unsigned int& n_poses) {
            return (n_poses*6) + (idx*3);
        };

        // Camera matrix
        cv::Mat _K;

        // The vector containing all the measurements, ordered by time
        std::vector<Measurement> _measurements;
     
        // The vector containing all the poses, ordered by time
        std::vector<cv::Mat> _poses;

        // The vector containing all the triangulated points, ordered
        // by insertion
        std::vector<cv::Point3f> _landmarks;

        // This vector contains all the keyframe
        std::vector<Keyframe> _keyframes;

        // Reference to the next measurement to analyze
        unsigned int _next_measurement_idx;

    };

} // namespace SLucAM



#endif // SLUCAM_STATE_H