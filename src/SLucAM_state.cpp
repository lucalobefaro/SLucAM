//
// SLucAM_state.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_state.h>
#include <SLucAM_geometry.h>

// TODO: delete this
#include <iostream>
using namespace std;



// -----------------------------------------------------------------------------
// Implementation of State class methods
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This constructor allows us to reserve some expected space for the vector
    * of poses and the vector of landmarks, just for optimization. It also need
    * the camera matrix K.
    * TODO: do shrink to fit somewhere. 
    */
    State::State(cv::Mat& K, \
                const unsigned int expected_poses, \
                const unsigned int expected_landmarks) {
        
        this->_K = K;
        this->_poses.reserve(expected_poses);
        this->_landmarks.reserve(expected_landmarks);

    }



    /*
    * This function initialize the state with two new poses, the landmarks
    * vector with the new triangulated points and the associations vector.
    * Inputs:
    *   new_pose: pose_2 to add to the pose vector (the first is assumed 
    *               at the origin)
    *   new_landmarks: list of landmarks to add to the state
    *   points1/points2: measured points in the initialization
    *   matches: matches between points1 and points2
    *   idxs: only the matches contained in this vector are considered during
    *       triangulation, so for each element in the idxs vector we have 
    *       a match in the matches vector from wich we can retrieve the points
    *       indices observed in measure1_idx and measure2_idx and we have
    *       the corresponding triangulated point in the new_landmarks vector
    *   measure1_idx/measure2_idx: index in the measures vector where to find 
    *       the points observed in points1/points2
    */
    void State::initializeObservations(cv::Mat& new_pose, \
                                    std::vector<cv::Point3f>& new_landmarks, \
                                    std::vector<cv::KeyPoint>& points1, \
                                    std::vector<cv::KeyPoint>& points2, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const std::vector<unsigned int>& idxs, \
                                    const unsigned int& measure1_idx, \
                                    const unsigned int& measure2_idx) {
        
        // Initialization
        const unsigned int n_observations = idxs.size();

        // Add all the landmarks in the landmarks vector
        

        // For each observation
        for(unsigned int i=0; i<n_observations; ++i) {

            // 
        } 
    }



    /*
    * Implementation of the boxplus operator for bundle adjustement.
    * It applies a perturbation (dx) to the state.
    * Inputs:
    *   dx: perturbation vector, the poses' perturbation come first, 
    *       then the landmarks.
    */
    void State::boxPlus(cv::Mat& dx) {
        
        // Initialization
        const unsigned int n_poses = this->_poses.size();
        const unsigned int n_landmarks = this->_landmarks.size();

        // Update poses
        for(unsigned int i=0; i<n_poses; ++i) {
            apply_perturbation_Tmatrix(dx, this->_poses[i], poseMatrixIdx(i));
        }

        // Update landmarks
        unsigned int real_idx;
        for(unsigned int i=0; i<n_landmarks; ++i) {
            real_idx = landmarkMatrixIdx(i, n_poses);
            this->_landmarks[i].x += dx.at<float>(real_idx, 0);
            this->_landmarks[i].y += dx.at<float>(real_idx+1, 0);
            this->_landmarks[i].z += dx.at<float>(real_idx+2, 0);
        }

    }


    /*
    * Function that linearize the robot-landmark measurement, useful for bundle
    * adjustment.
    * Inputs:
    *   poses: all the poses in the state
    *   landmarks: all the triangulated landmarks in the state
    *   measurements: all the measurements in the system
    *   associations: associations vector for measurements (see the _associations
    *       attribute of State class for details)
    *   K: the camera matrix
    *   H: (output) the resulting H matrix for Least-Square
    *       (we assume it is already initialized with dimension NxN
    *       where n= (6*#poses) + (3*#landmarks))
    *   b: (output) the resulting b vector for Least-Square
    *       (we assume it is already initialized with dimension Nx1
    *       where n= (6*#poses) + (3*#landmarks))
    *   chi_tot: (output) chi error of the current iteration of Least-Square
    *   kernel_threshold: robust kernel threshold
    *   threshold_to_ignore: error threshold that determine if an outlier is too
    *               much outlier to be considered
    * Outputs:
    *   n_inliers: #inliers
    */
    unsigned int State::buildLinearSystemProjections(\
                        const std::vector<cv::Mat>& poses, \
                        const std::vector<cv::Point3f>& landmarks, \
                        const std::vector<Measurement>& measurements, \
                        const std::vector<std::tuple<unsigned int, \
                            unsigned int, unsigned int>>& associations, \
                        const cv::Mat& K, \
                        cv::Mat& H, cv::Mat& b, \
                        float& chi_tot, \
                        const float& kernel_threshold, \
                        const float& threshold_to_ignore) {
        
        // Initialization
        unsigned int n_inliers = 0;
        chi_tot = 0.0;
        float current_chi = 0.0;
        const unsigned int n_measurements = associations.size();
        const unsigned int n_poses = poses.size();
        cv::Mat J_pose, J_landmark;
        cv::Mat error = cv::Mat::zeros(2,1,CV_32F);

        // For each measurement
        for(unsigned int i=0; i<n_measurements; ++i) {

            // Get the index of the observer pose
            const unsigned int& pose_idx = std::get<0>(associations[i]);
            
            // Get the elements of the measurement
            const cv::Mat& current_pose = poses[pose_idx];
            const cv::KeyPoint& current_measure = \
                        measurements[std::get<1>(associations[i])]\
                        .getPoints()[std::get<2>(associations[i])];
            const cv::Point3f& current_landmark = landmarks[i];

            // Compute error and Jacobian
            if(!computeProjectionErrorAndJacobian(current_pose, current_landmark, \
                        current_measure, K, J_pose, J_landmark, error)) {
                continue;   // The measurement is not valid, so ignore it
            }

            // Compute the chi error
            const float& error_1 = error.at<float>(0,0);
            const float& error_2 = error.at<float>(1,0);
            current_chi = (error_1*error_1) + (error_2*error_2);

            // Robust kernel
            if(current_chi > threshold_to_ignore)
                continue;
            if(current_chi > kernel_threshold) {
                error = error*sqrt(kernel_threshold/current_chi);
                current_chi = kernel_threshold;
            } else {
                ++n_inliers;
            }

            // Update the error evolution
            chi_tot += current_chi;

            // Some reference to jacobians
            const float& J_pose_11 = J_pose.at<float>(0,0);
            const float& J_pose_12 = J_pose.at<float>(0,1);
            const float& J_pose_13 = J_pose.at<float>(0,2);
            const float& J_pose_14 = J_pose.at<float>(0,3);
            const float& J_pose_15 = J_pose.at<float>(0,4);
            const float& J_pose_16 = J_pose.at<float>(0,5);
            const float& J_pose_21 = J_pose.at<float>(1,0);
            const float& J_pose_22 = J_pose.at<float>(1,1);
            const float& J_pose_23 = J_pose.at<float>(1,2);
            const float& J_pose_24 = J_pose.at<float>(1,3);
            const float& J_pose_25 = J_pose.at<float>(1,4);
            const float& J_pose_26 = J_pose.at<float>(1,5);
            const float& J_landmark_11 = J_landmark.at<float>(0,0);
            const float& J_landmark_12 = J_landmark.at<float>(0,1);
            const float& J_landmark_13 = J_landmark.at<float>(0,2);
            const float& J_landmark_21 = J_landmark.at<float>(1,0);
            const float& J_landmark_22 = J_landmark.at<float>(1,1);
            const float& J_landmark_23 = J_landmark.at<float>(1,2);

            // Retrieve the indices in the matrix H and vector b
            unsigned int pose_matrix_idx = poseMatrixIdx(pose_idx);
            unsigned int landmark_matrix_idx = landmarkMatrixIdx(i, n_poses);

            // Update the H matrix
            // J_pose.t()*J_pose
            H.at<float>(pose_matrix_idx, pose_matrix_idx) += \
                J_pose_11*J_pose_11 + J_pose_21*J_pose_21;
            H.at<float>(pose_matrix_idx, pose_matrix_idx+1) += \
                J_pose_11*J_pose_12 + J_pose_21*J_pose_22;
            H.at<float>(pose_matrix_idx, pose_matrix_idx+2) += \
                J_pose_11*J_pose_13 + J_pose_21*J_pose_23;
            H.at<float>(pose_matrix_idx, pose_matrix_idx+3) += \
                J_pose_11*J_pose_14 + J_pose_21*J_pose_24;
            H.at<float>(pose_matrix_idx, pose_matrix_idx+4) += \
                J_pose_11*J_pose_15 + J_pose_21*J_pose_25;
            H.at<float>(pose_matrix_idx, pose_matrix_idx+5) += \
                J_pose_11*J_pose_16 + J_pose_21*J_pose_26;
            H.at<float>(pose_matrix_idx+1, pose_matrix_idx) += \
                J_pose_12*J_pose_11 + J_pose_22*J_pose_21;
            H.at<float>(pose_matrix_idx+1, pose_matrix_idx+1) += \
                J_pose_12*J_pose_12 + J_pose_22*J_pose_22;
            H.at<float>(pose_matrix_idx+1, pose_matrix_idx+2) += \
                J_pose_12*J_pose_13 + J_pose_22*J_pose_23;
            H.at<float>(pose_matrix_idx+1, pose_matrix_idx+3) += \
                J_pose_12*J_pose_14 + J_pose_22*J_pose_24;
            H.at<float>(pose_matrix_idx+1, pose_matrix_idx+4) += \
                J_pose_12*J_pose_15 + J_pose_22*J_pose_25;
            H.at<float>(pose_matrix_idx+1, pose_matrix_idx+5) += \
                J_pose_12*J_pose_16 + J_pose_22*J_pose_26;
            H.at<float>(pose_matrix_idx+2, pose_matrix_idx) += \
                J_pose_13*J_pose_11 + J_pose_23*J_pose_21;
            H.at<float>(pose_matrix_idx+2, pose_matrix_idx+1) += \
                J_pose_13*J_pose_12 + J_pose_23*J_pose_22;
            H.at<float>(pose_matrix_idx+2, pose_matrix_idx+2) += \
                J_pose_13*J_pose_13 + J_pose_23*J_pose_23;
            H.at<float>(pose_matrix_idx+2, pose_matrix_idx+3) += \
                J_pose_13*J_pose_14 + J_pose_23*J_pose_24;
            H.at<float>(pose_matrix_idx+2, pose_matrix_idx+4) += \
                J_pose_13*J_pose_15 + J_pose_23*J_pose_25;
            H.at<float>(pose_matrix_idx+2, pose_matrix_idx+5) += \
                J_pose_13*J_pose_16 + J_pose_23*J_pose_26;
            H.at<float>(pose_matrix_idx+3, pose_matrix_idx) += \
                J_pose_14*J_pose_11 + J_pose_24*J_pose_21;
            H.at<float>(pose_matrix_idx+3, pose_matrix_idx+1) += \
                J_pose_14*J_pose_12 + J_pose_24*J_pose_22;
            H.at<float>(pose_matrix_idx+3, pose_matrix_idx+2) += \
                J_pose_14*J_pose_13 + J_pose_24*J_pose_23;
            H.at<float>(pose_matrix_idx+3, pose_matrix_idx+3) += \
                J_pose_14*J_pose_14 + J_pose_24*J_pose_24;
            H.at<float>(pose_matrix_idx+3, pose_matrix_idx+4) += \
                J_pose_14*J_pose_15 + J_pose_24*J_pose_25;
            H.at<float>(pose_matrix_idx+3, pose_matrix_idx+5) += \
                J_pose_14*J_pose_16 + J_pose_24*J_pose_26;
            H.at<float>(pose_matrix_idx+4, pose_matrix_idx) += \
                J_pose_15*J_pose_11 + J_pose_25*J_pose_21;
            H.at<float>(pose_matrix_idx+4, pose_matrix_idx+1) += \
                J_pose_15*J_pose_12 + J_pose_25*J_pose_22;
            H.at<float>(pose_matrix_idx+4, pose_matrix_idx+2) += \
                J_pose_15*J_pose_13 + J_pose_25*J_pose_23;
            H.at<float>(pose_matrix_idx+4, pose_matrix_idx+3) += \
                J_pose_15*J_pose_14 + J_pose_25*J_pose_24;
            H.at<float>(pose_matrix_idx+4, pose_matrix_idx+4) += \
                J_pose_15*J_pose_15 + J_pose_25*J_pose_25;
            H.at<float>(pose_matrix_idx+4, pose_matrix_idx+5) += \
                J_pose_15*J_pose_16 + J_pose_25*J_pose_26;
            H.at<float>(pose_matrix_idx+5, pose_matrix_idx) += \
                J_pose_16*J_pose_11 + J_pose_26*J_pose_21;
            H.at<float>(pose_matrix_idx+5, pose_matrix_idx+1) += \
                J_pose_16*J_pose_12 + J_pose_26*J_pose_22;
            H.at<float>(pose_matrix_idx+5, pose_matrix_idx+2) += \
                J_pose_16*J_pose_13 + J_pose_26*J_pose_23;
            H.at<float>(pose_matrix_idx+5, pose_matrix_idx+3) += \
                J_pose_16*J_pose_14 + J_pose_26*J_pose_24;
            H.at<float>(pose_matrix_idx+5, pose_matrix_idx+4) += \
                J_pose_16*J_pose_15 + J_pose_26*J_pose_25;
            H.at<float>(pose_matrix_idx+5, pose_matrix_idx+5) += \
                J_pose_16*J_pose_16 + J_pose_26*J_pose_26;
            
            // J_pose.t()*J_landmark
            H.at<float>(pose_matrix_idx, landmark_matrix_idx) += \
                J_pose_11*J_landmark_11 + J_pose_21*J_landmark_21;
            H.at<float>(pose_matrix_idx, landmark_matrix_idx+1) += \
                J_pose_11*J_landmark_12 + J_pose_21*J_landmark_22;
            H.at<float>(pose_matrix_idx, landmark_matrix_idx+2) += \
                J_pose_11*J_landmark_13 + J_pose_21*J_landmark_23;  
            H.at<float>(pose_matrix_idx+1, landmark_matrix_idx) += \
                J_pose_12*J_landmark_11 + J_pose_22*J_landmark_21;
            H.at<float>(pose_matrix_idx+1, landmark_matrix_idx+1) += \
                J_pose_12*J_landmark_12 + J_pose_22*J_landmark_22;
            H.at<float>(pose_matrix_idx+1, landmark_matrix_idx+2) += \
                J_pose_12*J_landmark_13 + J_pose_22*J_landmark_23;
            H.at<float>(pose_matrix_idx+2, landmark_matrix_idx) += \
                J_pose_13*J_landmark_11 + J_pose_23*J_landmark_21;
            H.at<float>(pose_matrix_idx+2, landmark_matrix_idx+1) += \
                J_pose_13*J_landmark_12 + J_pose_23*J_landmark_22;
            H.at<float>(pose_matrix_idx+2, landmark_matrix_idx+2) += \
                J_pose_13*J_landmark_13 + J_pose_23*J_landmark_23;
            H.at<float>(pose_matrix_idx+3, landmark_matrix_idx) += \
                J_pose_14*J_landmark_11 + J_pose_24*J_landmark_21;
            H.at<float>(pose_matrix_idx+3, landmark_matrix_idx+1) += \
                J_pose_14*J_landmark_12 + J_pose_24*J_landmark_22;
            H.at<float>(pose_matrix_idx+3, landmark_matrix_idx+2) += \
                J_pose_14*J_landmark_13 + J_pose_24*J_landmark_23;
            H.at<float>(pose_matrix_idx+4, landmark_matrix_idx) += \
                J_pose_15*J_landmark_11 + J_pose_25*J_landmark_21;
            H.at<float>(pose_matrix_idx+4, landmark_matrix_idx+1) += \
                J_pose_15*J_landmark_12 + J_pose_25*J_landmark_22;
            H.at<float>(pose_matrix_idx+4, landmark_matrix_idx+2) += \
                J_pose_15*J_landmark_13 + J_pose_25*J_landmark_23;
            H.at<float>(pose_matrix_idx+5, landmark_matrix_idx) += \
                J_pose_16*J_landmark_11 + J_pose_26*J_landmark_21;
            H.at<float>(pose_matrix_idx+5, landmark_matrix_idx+1) += \
                J_pose_16*J_landmark_12 + J_pose_26*J_landmark_22;
            H.at<float>(pose_matrix_idx+5, landmark_matrix_idx+2) += \
                J_pose_16*J_landmark_13 + J_pose_26*J_landmark_23;
            
            // J_landmark.t()*J_landmark
            H.at<float>(landmark_matrix_idx, landmark_matrix_idx) += \
                J_landmark_11*J_landmark_11 + J_landmark_21*J_landmark_21;
            H.at<float>(landmark_matrix_idx, landmark_matrix_idx+1) += \
                J_landmark_11*J_landmark_12 + J_landmark_21*J_landmark_22;
            H.at<float>(landmark_matrix_idx, landmark_matrix_idx+2) += \
                J_landmark_11*J_landmark_13 + J_landmark_21*J_landmark_23;
            H.at<float>(landmark_matrix_idx+1, landmark_matrix_idx) += \
                J_landmark_12*J_landmark_11 + J_landmark_22*J_landmark_21;
            H.at<float>(landmark_matrix_idx+1, landmark_matrix_idx+1) += \
                J_landmark_12*J_landmark_12 + J_landmark_22*J_landmark_22;
            H.at<float>(landmark_matrix_idx+1, landmark_matrix_idx+2) += \
                J_landmark_12*J_landmark_13 + J_landmark_22*J_landmark_23;
            H.at<float>(landmark_matrix_idx+2, landmark_matrix_idx) += \
                J_landmark_13*J_landmark_11 + J_landmark_23*J_landmark_21;
            H.at<float>(landmark_matrix_idx+2, landmark_matrix_idx+1) += \
                J_landmark_13*J_landmark_12 + J_landmark_23*J_landmark_22;
            H.at<float>(landmark_matrix_idx+2, landmark_matrix_idx+2) += \
                J_landmark_13*J_landmark_13 + J_landmark_23*J_landmark_23;

            // J_landmark.t()*J_pose
            H.at<float>(landmark_matrix_idx, pose_matrix_idx) += \
                J_landmark_11*J_pose_11 + J_landmark_21*J_pose_21;
            H.at<float>(landmark_matrix_idx, pose_matrix_idx+1) += \
                J_landmark_11*J_pose_12 + J_landmark_21*J_pose_22;
            H.at<float>(landmark_matrix_idx, pose_matrix_idx+2) += \
                J_landmark_11*J_pose_13 + J_landmark_21*J_pose_23;
            H.at<float>(landmark_matrix_idx, pose_matrix_idx+3) += \
                J_landmark_11*J_pose_14 + J_landmark_21*J_pose_24;
            H.at<float>(landmark_matrix_idx, pose_matrix_idx+4) += \
                J_landmark_11*J_pose_15 + J_landmark_21*J_pose_25;
            H.at<float>(landmark_matrix_idx, pose_matrix_idx+5) += \
                J_landmark_11*J_pose_16 + J_landmark_21*J_pose_26;
            H.at<float>(landmark_matrix_idx+1, pose_matrix_idx) += \
                J_landmark_12*J_pose_11 + J_landmark_22*J_pose_21;
            H.at<float>(landmark_matrix_idx+1, pose_matrix_idx+1) += \
                J_landmark_12*J_pose_12 + J_landmark_22*J_pose_22;
            H.at<float>(landmark_matrix_idx+1, pose_matrix_idx+2) += \
                J_landmark_12*J_pose_13 + J_landmark_22*J_pose_23;
            H.at<float>(landmark_matrix_idx+1, pose_matrix_idx+3) += \
                J_landmark_12*J_pose_14 + J_landmark_22*J_pose_24;
            H.at<float>(landmark_matrix_idx+1, pose_matrix_idx+4) += \
                J_landmark_12*J_pose_15 + J_landmark_22*J_pose_25;
            H.at<float>(landmark_matrix_idx+1, pose_matrix_idx+5) += \
                J_landmark_12*J_pose_16 + J_landmark_22*J_pose_26;
            H.at<float>(landmark_matrix_idx+2, pose_matrix_idx) += \
                J_landmark_13*J_pose_11 + J_landmark_23*J_pose_21;
            H.at<float>(landmark_matrix_idx+2, pose_matrix_idx+1) += \
                J_landmark_13*J_pose_12 + J_landmark_23*J_pose_22;
            H.at<float>(landmark_matrix_idx+2, pose_matrix_idx+2) += \
                J_landmark_13*J_pose_13 + J_landmark_23*J_pose_23;
            H.at<float>(landmark_matrix_idx+2, pose_matrix_idx+3) += \
                J_landmark_13*J_pose_14 + J_landmark_23*J_pose_24;
            H.at<float>(landmark_matrix_idx+2, pose_matrix_idx+4) += \
                J_landmark_13*J_pose_15 + J_landmark_23*J_pose_25;
            H.at<float>(landmark_matrix_idx+2, pose_matrix_idx+5) += \
                J_landmark_13*J_pose_16 + J_landmark_23*J_pose_26;

            // Update the b vector
            // J_pose.t()*error
            b.at<float>(pose_matrix_idx, pose_matrix_idx) += \
                J_pose_11*error_1 + J_pose_21*error_2; 
            b.at<float>(pose_matrix_idx, pose_matrix_idx+1) += \
                J_pose_12*error_1 + J_pose_22*error_2;
            b.at<float>(pose_matrix_idx, pose_matrix_idx+2) += \
                J_pose_13*error_1 + J_pose_23*error_2;
            b.at<float>(pose_matrix_idx, pose_matrix_idx+3) += \
                J_pose_14*error_1 + J_pose_24*error_2;
            b.at<float>(pose_matrix_idx, pose_matrix_idx+4) += \
                J_pose_15*error_1 + J_pose_25*error_2;
            b.at<float>(pose_matrix_idx, pose_matrix_idx+5) += \
                J_pose_16*error_1 + J_pose_26*error_2;

            // J_landmark.t()*error
            b.at<float>(landmark_matrix_idx, landmark_matrix_idx) += \
                J_landmark_11*error_1 + J_landmark_21*error_2;
            b.at<float>(landmark_matrix_idx, landmark_matrix_idx+1) += \
                J_landmark_12*error_1 + J_landmark_22*error_2;
            b.at<float>(landmark_matrix_idx, landmark_matrix_idx+2) += \
                J_landmark_13*error_1 + J_landmark_23*error_2;

        }

        return n_inliers;        
    
    }


    /*
    * This function compute the error and jacobian for the projection measurements
    * useful for bundle adjustment.
    * Inputs:
    *   pose: pose of the camera observer in world frame
    *   landmark_pose: believed pose of the landmark
    *   img_point: observation of the landmarks from the camera
    *   K: camera matrix
    *   J_pose: output 2x6 derivative w.r.t. the error and a perturbation 
    *           of the pose
    *   J_landmark: output 2x3 derivative w.r.t. the error and a perturbation
    *               on the landmark pose
    *   error: 2x1 difference between prediction and measurement, we assume it is
    *           already instantiated as a 2x1 cv::Mat
    *  Outputs:
    *   true if the projection is valide, false otherwise
    */
    bool State::computeProjectionErrorAndJacobian(const cv::Mat& pose, \
                        const cv::Point3f& landmark_pose, \
                        const cv::KeyPoint& img_point, const cv::Mat& K,   
                        cv::Mat& J_pose, cv::Mat& J_landmark, cv::Mat& error) {

        // Some reference to save time
        const float& landmark_pose_x = landmark_pose.x;
        const float& landmark_pose_y = landmark_pose.y;
        const float& landmark_pose_z = landmark_pose.z;
        const float& t_x = pose.at<float>(0,3);
        const float& t_y = pose.at<float>(1,3);
        const float& t_z = pose.at<float>(2,3);
        const float& K11 = K.at<float>(0,0);
        const float& K12 = K.at<float>(0,1);
        const float& K13 = K.at<float>(0,2);
        const float& K21 = K.at<float>(1,0);
        const float& K22 = K.at<float>(1,1);
        const float& K23 = K.at<float>(1,2);
        const float& K31 = K.at<float>(2,0);
        const float& K32 = K.at<float>(2,1);
        const float& K33 = K.at<float>(2,2);

        // Get references to the transpose of the rotational part of
        // the pose (the 3x3 submatrix of pose where we have the R matrix)
        const float& iR11 = pose.at<float>(0,0);
        const float& iR12 = pose.at<float>(1,0);
        const float& iR13 = pose.at<float>(2,0);
        const float& iR21 = pose.at<float>(0,1);
        const float& iR22 = pose.at<float>(1,1);
        const float& iR23 = pose.at<float>(2,1);
        const float& iR31 = pose.at<float>(0,2);
        const float& iR32 = pose.at<float>(1,2);
        const float& iR33 = pose.at<float>(2,2);

        // Compute the point prediction in world coordinates
        // p_world = R'*landmark_pose + (-R'*pose)
        // where pose is the translation subvector of pose
        const float p_world_x = iR11*landmark_pose_x + \
                                 iR12*landmark_pose_y + \
                                 iR13*landmark_pose_z + \
                                 -(iR11*t_x + iR12*t_y + iR13*t_z);
        const float p_world_y = iR21*landmark_pose_x + \
                                 iR22*landmark_pose_y + \
                                 iR23*landmark_pose_z + \
                                 -(iR21*t_x + iR22*t_y + iR23*t_z);
        const float p_world_z = iR31*landmark_pose_x + \
                                 iR32*landmark_pose_y + \
                                 iR33*landmark_pose_z + \
                                 -(iR31*t_x + iR32*t_y + iR33*t_z);

        // Check if the prediction is in front of camera
        if(p_world_z<0)
            return false;

        // Jacobian of landmark prediction for the ICP part w.r.t. the 
        // part of the state that refers to the pose and then 
        // w.r.t. the part of the state that refers to the landmark pose
        cv::Mat Jw_pose = cv::Mat::zeros(3,6,CV_32F);
        // -R'
        Jw_pose.at<float>(0,0) = -iR11;
        Jw_pose.at<float>(0,1) = -iR12;
        Jw_pose.at<float>(0,2) = -iR13;
        Jw_pose.at<float>(1,0) = -iR21;
        Jw_pose.at<float>(1,1) = -iR22;
        Jw_pose.at<float>(1,2) = -iR23; 
        Jw_pose.at<float>(2,0) = -iR31;
        Jw_pose.at<float>(2,1) = -iR32;
        Jw_pose.at<float>(2,2) = -iR33;
        // R'*skew(landmark_pose)
        Jw_pose.at<float>(0,3) = iR12*(landmark_pose_z)+iR13*(-landmark_pose_y);
        Jw_pose.at<float>(0,4) = iR11*(-landmark_pose_z)+iR13*(landmark_pose_x);
        Jw_pose.at<float>(0,5) = iR11*(landmark_pose_y)+iR12*(-landmark_pose_x);
        Jw_pose.at<float>(1,3) = iR22*(landmark_pose_z)+iR23*(-landmark_pose_y);
        Jw_pose.at<float>(1,4) = iR21*(-landmark_pose_z)+iR23*(landmark_pose_x);
        Jw_pose.at<float>(1,5) = iR21*(landmark_pose_y)+iR22*(-landmark_pose_x);
        Jw_pose.at<float>(2,3) = iR32*(landmark_pose_z)+iR33*(-landmark_pose_y);
        Jw_pose.at<float>(2,4) = iR31*(-landmark_pose_z)+iR33*(landmark_pose_x);
        Jw_pose.at<float>(2,5) = iR31*(landmark_pose_y)+iR32*(-landmark_pose_x);
        // R'
        cv::Mat Jw_landmark = cv::Mat::zeros(3,3,CV_32F);
        Jw_landmark.at<float>(0,0) = iR11;
        Jw_landmark.at<float>(0,1) = iR12;
        Jw_landmark.at<float>(0,2) = iR13;
        Jw_landmark.at<float>(1,0) = iR21;
        Jw_landmark.at<float>(1,1) = iR22;
        Jw_landmark.at<float>(1,2) = iR23;
        Jw_landmark.at<float>(2,0) = iR31;
        Jw_landmark.at<float>(2,1) = iR32;
        Jw_landmark.at<float>(2,2) = iR33;

        // Point prediction, in camera frame
        const float p_cam_x = K11*p_world_x + \
                              K12*p_world_y + \
                              K13*p_world_z;
        const float p_cam_y = K21*p_world_x + \
                              K22*p_world_y + \
                              K23*p_world_z;
        const float p_cam_z = K31*p_world_x + \
                              K32*p_world_y + \
                              K33*p_world_z;

        // Point prediction, on projection plane
        const float iz = 1/(p_cam_z);
        const float z_hat_x = p_cam_x*iz;
        const float z_hat_y = p_cam_y*iz;

        // Check if the point prediction on projection plane is inside 
        // the camera frustum
        if (z_hat_x < 0 || 
            z_hat_x > (2*K23) ||
            z_hat_y < 0 || 
            z_hat_y > (2*K13))
            return false;
        
        // compute the derivative of the projection function
        const float iz2 = iz*iz;
        cv::Mat Jp = cv::Mat::zeros(2,3,CV_32F);
        Jp.at<float>(0,0) = iz;
        Jp.at<float>(0,2) = -p_cam_x*iz2;
        Jp.at<float>(1,1) = iz;
        Jp.at<float>(1,2) = -p_cam_y*iz2;

        // Compute the error
        error.at<float>(0,0) = z_hat_x - img_point.pt.x;
        error.at<float>(1,0) = z_hat_y - img_point.pt.y;

        // Compute the final jacobians
        J_pose = Jp*K*Jw_pose;
        J_landmark = Jp*K*Jw_landmark;

        return true;        
    }


    /*
    * This function compute the error and jacobian for the measured pose
    * useful for bundle adjustment.
    * Inputs:
    *   pose_1: the observing robot pose (4x4 homogeneous matrix)
    *   pose_2: the observed robot pose (4x4 homogeneous matrix)
    *   pose2_wrt_pose1: the relative transform measured between pose-1 
    *                       and pose_2
    *   J_1: 12x6 derivative w.r.t the error and a perturbation of the
    *           first pose
    *   J_2: 12x6 derivative w.r.t the error and a perturbation of the
    *           second pose
    *   error: 12x1 difference between prediction and measurement, vectorized
    */
    void State::computePoseErrorAndJacobian(const cv::Mat& pose_1, \
                        const cv::Mat& pose_2, \
                        const cv::Mat& pose2_wrt_pose1, \
                        cv::Mat& J_1, cv::Mat& J_2, cv::Mat& error) {
        
        // Reference to the rotational part of pose_1, transposed
        const float& iR1_11 = pose_1.at<float>(0,0);
        const float& iR1_12 = pose_1.at<float>(1,0);
        const float& iR1_13 = pose_1.at<float>(2,0);
        const float& iR1_21 = pose_1.at<float>(0,1);
        const float& iR1_22 = pose_1.at<float>(1,1);
        const float& iR1_23 = pose_1.at<float>(2,1);
        const float& iR1_31 = pose_1.at<float>(0,2);
        const float& iR1_32 = pose_1.at<float>(1,2);
        const float& iR1_33 = pose_1.at<float>(2,2);

        // Reference to the rotational part of pose_2
        const float& R2_11 = pose_2.at<float>(0,0);
        const float& R2_12 = pose_2.at<float>(0,1);
        const float& R2_13 = pose_2.at<float>(0,2);
        const float& R2_21 = pose_2.at<float>(1,0);
        const float& R2_22 = pose_2.at<float>(1,1);
        const float& R2_23 = pose_2.at<float>(1,2);
        const float& R2_31 = pose_2.at<float>(2,0);
        const float& R2_32 = pose_2.at<float>(2,1);
        const float& R2_33 = pose_2.at<float>(2,2);

        // Reference to the translational part of pose_2
        const float& t2_x = pose_2.at<float>(0,3);
        const float& t2_y = pose_2.at<float>(1,3);
        const float& t2_z = pose_2.at<float>(2,3);

        // Difference of translation between the two poses
        const float t_diff_x = t2_x - pose_1.at<float>(0,3);
        const float t_diff_y = t2_y - pose_1.at<float>(1,3);
        const float t_diff_z = t2_z - pose_1.at<float>(2,3);

        // Compute J2
        J_2 = cv::Mat::zeros(12,6,CV_32F);

        // iR1*Rx0*R2 where Rx0 is the matrix:
        //  [0 0 0; 0 0 -1; 0 1 0] (MATLAB notation)
        J_2.at<float>(0,3) = iR1_13*R2_21 - iR1_12*R2_31;
        J_2.at<float>(3,3) = iR1_13*R2_22 - iR1_12*R2_32;
        J_2.at<float>(6,3) = iR1_13*R2_23 - iR1_12*R2_33;
        J_2.at<float>(1,3) = iR1_23*R2_21 - iR1_22*R2_31;
        J_2.at<float>(4,3) = iR1_23*R2_22 - iR1_22*R2_32;
        J_2.at<float>(7,3) = iR1_23*R2_23 - iR1_22*R2_33;
        J_2.at<float>(2,3) = iR1_33*R2_21 - iR1_32*R2_31;
        J_2.at<float>(5,3) = iR1_33*R2_22 - iR1_32*R2_32;
        J_2.at<float>(8,3) = iR1_33*R2_23 - iR1_32*R2_33;

        // iR1*Ry0*R2 where Rx0 is the matrix:
        //  [0 0 1; 0 0 0; -1 0 0] (MATLAB notation)
        J_2.at<float>(0,4) = R2_31*iR1_11 - R2_11*iR1_13;
        J_2.at<float>(3,4) = R2_32*iR1_11 - R2_12*iR1_13;
        J_2.at<float>(6,4) = R2_33*iR1_11 - R2_13*iR1_13;  
        J_2.at<float>(1,4) = R2_31*iR1_21 - R2_11*iR1_23;
        J_2.at<float>(4,4) = R2_32*iR1_21 - R2_12*iR1_23;
        J_2.at<float>(7,4) = R2_33*iR1_21 - R2_13*iR1_23;
        J_2.at<float>(2,4) = R2_31*iR1_31 - R2_11*iR1_33;
        J_2.at<float>(5,4) = R2_32*iR1_31 - R2_12*iR1_33;
        J_2.at<float>(8,4) = R2_33*iR1_31 - R2_13*iR1_33;

        // iR1*Rz0*R2 where Rx0 is the matrix:
        //  [0 -1 0; 1 0 0; 0 0 0] (MATLAB notation)
        J_2.at<float>(0,5) = R2_11*iR1_12 - R2_21*iR1_11;
        J_2.at<float>(3,5) = R2_12*iR1_12 - R2_22*iR1_11;
        J_2.at<float>(6,5) = R2_13*iR1_12 - R2_23*iR1_11;
        J_2.at<float>(1,5) = R2_11*iR1_22 - R2_21*iR1_21;
        J_2.at<float>(4,5) = R2_12*iR1_22 - R2_22*iR1_21;
        J_2.at<float>(7,5) = R2_13*iR1_22 - R2_23*iR1_21;
        J_2.at<float>(2,5) = R2_11*iR1_32 - R2_21*iR1_31;
        J_2.at<float>(5,5) = R2_12*iR1_32 - R2_22*iR1_31;
        J_2.at<float>(8,5) = R2_13*iR1_32 - R2_23*iR1_31;

        // iR1
        J_2.at<float>(9,0) = iR1_11;
        J_2.at<float>(9,1) = iR1_12;
        J_2.at<float>(9,2) = iR1_13;
        J_2.at<float>(10,0) = iR1_21;
        J_2.at<float>(10,1) = iR1_22;
        J_2.at<float>(10,2) = iR1_23;
        J_2.at<float>(11,0) = iR1_31;
        J_2.at<float>(11,1) = iR1_32;
        J_2.at<float>(11,2) = iR1_33;

        // -iR1*skew(t2)
        J_2.at<float>(9,3) = iR1_13*t2_y - iR1_12*t2_z;
        J_2.at<float>(9,4) = iR1_11*t2_z - iR1_13*t2_x;
        J_2.at<float>(9,5) = iR1_12*t2_x - iR1_11*t2_y;
        J_2.at<float>(10,3) = iR1_23*t2_y - iR1_22*t2_z;
        J_2.at<float>(10,4) = iR1_21*t2_z - iR1_23*t2_x;
        J_2.at<float>(10,5) = iR1_22*t2_x - iR1_21*t2_y;
        J_2.at<float>(11,3) = iR1_33*t2_y - iR1_32*t2_z;
        J_2.at<float>(11,4) = iR1_31*t2_z - iR1_33*t2_x;
        J_2.at<float>(11,5) = iR1_32*t2_x - iR1_31*t2_y;

        // Compute J_1 
        // TODO: can we avoid this computation?
        J_1 = -J_2;

        // Compute the error in the rotation
        error = cv::Mat::zeros(12,1,CV_32F);
        error.at<float>(0,0) = (iR1_11*R2_11 + iR1_12*R2_21 + iR1_13*R2_31) \
                                - pose2_wrt_pose1.at<float>(0,0);
        error.at<float>(3,0) = (iR1_11*R2_12 + iR1_12*R2_22 + iR1_13*R2_32) \
                                - pose2_wrt_pose1.at<float>(0,1);
        error.at<float>(6,0) = (iR1_11*R2_13 + iR1_12*R2_23 + iR1_13*R2_33) \
                                - pose2_wrt_pose1.at<float>(0,2);
        error.at<float>(1,0) = (iR1_21*R2_11 + iR1_22*R2_21 + iR1_23*R2_31) \
                                - pose2_wrt_pose1.at<float>(1,0);
        error.at<float>(4,0) = (iR1_21*R2_12 + iR1_22*R2_22 + iR1_23*R2_32) \
                                - pose2_wrt_pose1.at<float>(1,1);
        error.at<float>(7,0) = (iR1_21*R2_13 + iR1_22*R2_23 + iR1_23*R2_33) \
                                - pose2_wrt_pose1.at<float>(1,2);
        error.at<float>(2,0) = (iR1_31*R2_11 + iR1_32*R2_21 + iR1_33*R2_31) \
                                - pose2_wrt_pose1.at<float>(2,0);
        error.at<float>(5,0) = (iR1_31*R2_12 + iR1_32*R2_22 + iR1_33*R2_32) \
                                - pose2_wrt_pose1.at<float>(2,1);
        error.at<float>(8,0) = (iR1_31*R2_13 + iR1_32*R2_23 + iR1_33*R2_33) \
                                - pose2_wrt_pose1.at<float>(2,2);

        // Compute the error in the translation
        error.at<float>(9,0) = (iR1_11*t_diff_x + iR1_12*t_diff_y + iR1_13*t_diff_z) \
                                - pose2_wrt_pose1.at<float>(0,3);
        error.at<float>(10,0) = (iR1_21*t_diff_x + iR1_22*t_diff_y + iR1_23*t_diff_z) \
                                - pose2_wrt_pose1.at<float>(1,3);
        error.at<float>(11,0) = (iR1_31*t_diff_x + iR1_32*t_diff_y + iR1_33*t_diff_z) \
                                - pose2_wrt_pose1.at<float>(2,3);

    }


} // namespace SLucAM