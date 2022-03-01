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



// -----------------------------------------------------------------------------
// Association struct
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    /*
    * This struct is useful to keep note of each observation, by memorizing:
    *   pose_idx: the observer pose
    *   landmark_idx: the believed position of the landmark observed
    *   measurement_idx: the measure where we take this observation
    *   point_idx: the point which this observation refers to, in the given
    *               measurement
    */
    struct Association {
        const unsigned int pose_idx;
        const unsigned int landmark_idx;
        const unsigned int measurement_idx;
        const unsigned int point_idx;

        Association(const unsigned int p_idx, \
                    const unsigned int l_idx, \
                    const unsigned int m_idx, \
                    const unsigned int pnt_idx)
            : pose_idx(p_idx)
            , landmark_idx(l_idx)
            , measurement_idx(m_idx)
            , point_idx(pnt_idx)
        {}
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

        State(cv::Mat& K, \
            const unsigned int expected_poses, \
            const unsigned int expected_landmarks);

        void initializeObservations(cv::Mat& new_pose, \
                                    std::vector<cv::Point3f>& new_landmarks, \
                                    const std::vector<cv::KeyPoint>& points1, \
                                    const std::vector<cv::KeyPoint>& points2, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const std::vector<unsigned int>& idxs, \
                                    const unsigned int& measure1_idx, \
                                    const unsigned int& measure2_idx);

        const std::vector<cv::Mat>& getPoses() const {return this->_poses;};

        const std::vector<cv::Point3f>& getLandmarks() const {return this->_landmarks;};

        const cv::Mat& getCameraMatrix() const {return this->_K;};

        float performBundleAdjustment(const float& n_iterations, \
                                        const float& damping_factor, \
                                        const float& kernel_threshold);

    private: 

        void boxPlus(cv::Mat& dx);
    
        static unsigned int buildLinearSystemProjections(\
                        const std::vector<cv::Mat>& poses, \
                        const std::vector<cv::Point3f>& landmarks, \
                        const std::vector<Measurement>& measurements, \
                        const std::vector<Association>& associations, \
                        const cv::Mat& K, \
                        cv::Mat& H, cv::Mat& b, \
                        float& chi_tot, \
                        const float& kernel_threshold, \
                        const float& threshold_to_ignore);

        static bool computeProjectionErrorAndJacobian(const cv::Mat& pose, \
                        const cv::Point3f& landmark_pose, \
                        const cv::KeyPoint& img_point, const cv::Mat& K, \
                        cv::Mat& J_pose, cv::Mat& J_landmark, cv::Mat& error);
      
        static void computePoseErrorAndJacobian(const cv::Mat& pose_1, \
                        const cv::Mat& pose_2, \
                        const cv::Mat& pose2_wrt_pose1, \
                        cv::Mat& J_1, cv::Mat& J_2, cv::Mat& error);
            
        static inline unsigned int poseMatrixIdx(const unsigned int&  idx) {
            return idx*6;
        };

        static inline unsigned int landmarkMatrixIdx(const unsigned int&  idx, \
                                                    const unsigned int& n_poses) {
            return (n_poses*6) + (idx*3);
        };
     
        // The vector containing all the poses, ordered by time
        std::vector<cv::Mat> _poses;

        // The vector containing all the measurements, ordered by time
        std::vector<Measurement> _measurements;

        // In this vector we have the informations of each observation
        std::vector<Association> _associations;

        // The vector containing all the triangulated points, ordered
        // by insertion
        std::vector<cv::Point3f> _landmarks;

        // Camera matrix
        cv::Mat _K;

    };

} // namespace SLucAM



#endif // SLUCAM_STATE_H