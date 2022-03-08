//
// SLucAM_state.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_state.h>
#include <SLucAM_geometry.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>


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
    State::State(cv::Mat& K, std::vector<Measurement>& measurements, \
                const unsigned int expected_poses, \
                const unsigned int expected_landmarks) {
        
        this->_K = K;
        this->_measurements = measurements;
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
    void State::initializeState(cv::Mat& new_pose, \
                                std::vector<cv::Point3f>& new_landmarks, \
                                const std::vector<cv::KeyPoint>& points1, \
                                const std::vector<cv::KeyPoint>& points2, \
                                const std::vector<cv::DMatch>& matches, \
                                const std::vector<unsigned int>& idxs, \
                                const unsigned int& measure1_idx, \
                                const unsigned int& measure2_idx) {
        
        // Initialization
        const unsigned int n_observations = idxs.size();

        // Add the first two poses
        this->_poses.emplace_back(cv::Mat::eye(4,4,CV_32F));
        this->_poses.emplace_back(new_pose);

        // Add the observation "first pose observes second pose"
        this->_pose_observations.emplace_back(0,1);
        this->_poses_measurements.emplace_back(\
                    this->_poses[0].inv()*this->_poses[1]); // TODO: check if this has to be computed here, so constant or at each cycle of bundle adjustment

        // For each observation
        for(unsigned int i=0; i<n_observations; ++i) {

            // If the landmark is not triangulated in a good way
            if(new_landmarks[i].x == 0 && \
                new_landmarks[i].y == 0 && \
                new_landmarks[i].z == 0) {
                continue;                   // Ignore this association
            }

            this->_landmarks.emplace_back(new_landmarks[i]);

            // Add the observation made from pose 1
            this->_landmark_observations.emplace_back(0, this->_landmarks.size()-1, \
                        measure1_idx, matches[idxs[i]].queryIdx);

            // Add the same observation made from pose 2
            this->_landmark_observations.emplace_back(1, this->_landmarks.size()-1, \
                        measure2_idx, matches[idxs[i]].trainIdx);
        } 
    }


    /*
    * Implementation of the Bundle Adjustment optimization with Least-Squares
    * method.
    * Inputs:
    *   n_iterations: the number of iterations for Least-Squares
    *   damping_factor: damping factor (in case system not spd)
    *   kernel_threshold: robust kernel threshold
    *   threshold_to_ignore: error threshold that determine if an outlier is 
    *           too outlier to be considered
    */
    void State::performBundleAdjustment(const float& n_iterations, \
                                        const float& damping_factor, \
                                        const float& kernel_threshold, \
                                        const float& threshold_to_ignore){
        
        // Initialization
        const float img_rows = 2*this->_K.at<float>(1, 2);
        const float img_cols = 2*this->_K.at<float>(0, 2);
        std::vector<unsigned int> n_inliers_projection(n_iterations, 0);
        std::vector<unsigned int> n_inliers_pose(n_iterations, 0);
        std::vector<float> chi_stats_projection(n_iterations, 0.0);
        std::vector<float> chi_stats_pose(n_iterations, 0.0);
        unsigned int n_landmarks = this->_landmarks.size();
        unsigned int system_size = (6*this->_poses.size()) + (3*n_landmarks);
        cv::Mat H, b;
        const cv::Mat DampingMatrix = \
                    cv::Mat::eye(system_size, system_size, CV_32F)*damping_factor;

        // Start optimization process
        for(unsigned int iter=0; iter<n_iterations; ++iter) {

            // Reset to zero the H and b matrices
            H = cv::Mat::zeros(system_size, system_size, CV_32F);
            b = cv::Mat::zeros(system_size, 1, CV_32F);

            // If we have landmarks in the state
            if(n_landmarks > 0) {

                // Add to H and b matrices the values from projections
                n_inliers_projection[iter] = \
                    State::buildLinearSystemProjections(this->_poses, \
                                                        this->_landmarks, 
                                                        this->_measurements, \
                                                        this->_landmark_observations, \
                                                        this->_K, H, b, \
                                                        chi_stats_projection[iter], \
                                                        kernel_threshold, threshold_to_ignore, \
                                                        img_rows, img_cols);

            }

            // Add to H and b matrices the values from poses
            n_inliers_pose[iter] = \
                State::buildLinearSystemPoses(this->_poses, this->_pose_observations, \
                                        this->_poses_measurements, \
                                        H, b, chi_stats_pose[iter], kernel_threshold);
            
            // Damping of the H matrix
            H += DampingMatrix;

            // The linear system is underdetermined so we need
            // to eliminate redundant variables. For this reasor we
            // solve the linear system blocking the first pose,
            // this corresponds to "remove" from H and b the blocks
            // of the 1st pose, while solving the system (using Eigen)
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> \
                        H_Eigen(H.ptr<float>(), H.rows, H.cols);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> \
                        b_Eigen(b.ptr<float>(), b.rows, b.cols);
            Eigen::VectorXf dx_Eigen = H_Eigen.block(6,6,system_size-6,system_size-6)\
                        .ldlt().solve(-b_Eigen.block(6,0,system_size-6,1));
            cv::Mat dx(dx_Eigen.rows(), dx_Eigen.cols(), CV_32F, dx_Eigen.data());

            // Apply the perturbation (we assume that the first 6 elements of dx 
            // are zero)
            this->boxPlus(dx);
        }
        
    }


    /*
    * Implementation of the boxplus operator for bundle adjustement.
    * It applies a perturbation (dx) to the state.
    * Inputs:
    *   dx: perturbation vector, the poses' perturbation come first, 
    *       then the landmarks (we assume that the first 6 elements of 
    *       dx are zeros, so the first pose is not updated).
    */
    void State::boxPlus(cv::Mat& dx) {
        
        // Initialization
        const unsigned int n_poses = this->_poses.size();
        const unsigned int n_landmarks = this->_landmarks.size();

        // Update poses (starting from the pose with idx 1, the first one
        // is not updated because we assume dx has the first 6 elements
        // as zeros)
        for(unsigned int i=1; i<n_poses; ++i) {
            apply_perturbation_Tmatrix(dx, this->_poses[i], poseMatrixIdx(i)-6);
        }

        // Update landmarks
        unsigned int real_idx;
        for(unsigned int i=0; i<n_landmarks; ++i) {
            real_idx = landmarkMatrixIdx(i, n_poses);
            this->_landmarks[i].x += dx.at<float>((real_idx)-6, 0);
            this->_landmarks[i].y += dx.at<float>((real_idx+1)-6, 0);
            this->_landmarks[i].z += dx.at<float>((real_idx+2)-6, 0);
        }

    }


    /*
    * Function that linearize the pose-landmark measurements, useful for bundle
    * adjustment.
    * Inputs:
    *   poses: all the poses in the state
    *   landmarks: all the triangulated landmarks in the state
    *   measurements: all the measurements in the system
    *   associations: associations vector for measurements (see the _associations
    *       attribute of State class for details)
    *   K: the camera matrix
    *   H: (output) the resulting H matrix for Least-Square
    *       (we assume it is already initialized with zeros and dimension NxN
    *       where n= (6*#poses) + (3*#landmarks))
    *   b: (output) the resulting b vector for Least-Square
    *       (we assume it is already initialized with zeros and dimension Nx1
    *       where n= (6*#poses) + (3*#landmarks))
    *   chi_tot: (output) chi error of the current iteration of Least-Square
    *   kernel_threshold: robust kernel threshold
    *   threshold_to_ignore: error threshold that determine if an outlier is too
    *               much outlier to be considered
    *   img_rows: #rows in the image plane pixels matrix
    *   img_cols: #cols in the image plane pixels matrix
    * Outputs:
    *   n_inliers: #inliers
    */
    unsigned int State::buildLinearSystemProjections(\
                        const std::vector<cv::Mat>& poses, \
                        const std::vector<cv::Point3f>& landmarks, \
                        const std::vector<Measurement>& measurements, \
                        const std::vector<LandmarkObservation>& associations, \
                        const cv::Mat& K, \
                        cv::Mat& H, cv::Mat& b, \
                        float& chi_tot, \
                        const float& kernel_threshold, \
                        const float& threshold_to_ignore, \
                        const float& img_rows, \
                        const float& img_cols) {
        
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

            // Get the index of the observer pose and the index of 
            // the landmark observed
            const unsigned int& pose_idx = associations[i].pose_idx;
            const unsigned int& landmark_idx = associations[i].landmark_idx;
            
            // Get the elements of the measurement
            const cv::Mat& current_pose = poses[pose_idx];
            const cv::Point3f& current_landmark = landmarks[landmark_idx];
            const cv::KeyPoint& current_measure = \
                        measurements[associations[i].measurement_idx]\
                        .getPoints()[associations[i].point_idx];

            // Compute error and Jacobian
            if(!computeProjectionErrorAndJacobian(current_pose, current_landmark, \
                        current_measure, K, J_pose, J_landmark, error, \
                        img_rows, img_cols)) {
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
            unsigned int landmark_matrix_idx = landmarkMatrixIdx(landmark_idx, n_poses);

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
            b.at<float>(pose_matrix_idx, 0) += \
                J_pose_11*error_1 + J_pose_21*error_2; 
            b.at<float>(pose_matrix_idx+1, 0) += \
                J_pose_12*error_1 + J_pose_22*error_2;
            b.at<float>(pose_matrix_idx+2, 0) += \
                J_pose_13*error_1 + J_pose_23*error_2;
            b.at<float>(pose_matrix_idx+3, 0) += \
                J_pose_14*error_1 + J_pose_24*error_2;
            b.at<float>(pose_matrix_idx+4, 0) += \
                J_pose_15*error_1 + J_pose_25*error_2;
            b.at<float>(pose_matrix_idx+5, 0) += \
                J_pose_16*error_1 + J_pose_26*error_2;

            // J_landmark.t()*error
            b.at<float>(landmark_matrix_idx, 0) += \
                J_landmark_11*error_1 + J_landmark_21*error_2;
            b.at<float>(landmark_matrix_idx+1, 0) += \
                J_landmark_12*error_1 + J_landmark_22*error_2;
            b.at<float>(landmark_matrix_idx+2, 0) += \
                J_landmark_13*error_1 + J_landmark_23*error_2;

        }

        return n_inliers;        
    
    }


    /*
    * Function that linearize the pose-pose measurements, useful for 
    * bundle adjustment.
    * Inputs:
    *   poses: all the poses in the state
    *   landmarks: all the triangulated landmarks in the state
    *   H: (output) the resulting H matrix for Least-Square
    *       (we assume it is already initialized with zeros and dimension NxN
    *       where n= (6*#poses) + (3*#landmarks))
    *   b: (output) the resulting b vector for Least-Square
    *       (we assume it is already initialized with zeros and dimension Nx1
    *       where n= (6*#poses) + (3*#landmarks))
    *   chi_tot: (output) chi error of the current iteration of Least-Square
    *   kernel_threshold: robust kernel threshold
    * Outputs:
    *   n_inliers: #inliers
    */
    unsigned int State::buildLinearSystemPoses(\
                        const std::vector<cv::Mat>& poses, \
                        const std::vector<PoseObservation>& associations, \
                        const std::vector<cv::Mat>& poses_measurements, \
                        cv::Mat& H, cv::Mat& b, \
                        float& chi_tot, \
                        const float& kernel_threshold) {
        
        // Initialization
        unsigned int n_inliers = 0;
        chi_tot = 0.0;
        float current_chi = 0.0;
        const unsigned int n_measurements = associations.size();
        cv::Mat J_2 = cv::Mat::zeros(12,6,CV_32F);  // J_1 is -J_2
        cv::Mat error = cv::Mat::zeros(12,1,CV_32F);
        float current_H_11, current_H_12, current_H_13, current_H_14, current_H_15, \
            current_H_16, current_H_21, current_H_22, current_H_23, current_H_24, \
            current_H_25, current_H_26, current_H_31, current_H_32, current_H_33, \
            current_H_34, current_H_35, current_H_36, current_H_41, current_H_42, \
            current_H_43, current_H_44, current_H_45, current_H_46, current_H_51, \
            current_H_52, current_H_53, current_H_54, current_H_55, current_H_56, \
            current_H_61, current_H_62, current_H_63, current_H_64, current_H_65, \
            current_H_66, \
            current_b_1, current_b_2, current_b_3, current_b_4, current_b_5, \
            current_b_6;

        // Diagonal values of the Omega matrix for the rotational part
        float omega_r;

        // // Diagonal values of the Omega matrix for the translational part
        float omega_t;

        // For each measurement
        for(unsigned int i=0; i<n_measurements; ++i) {

            // Reset the omega values
            omega_r = 1e3;
            omega_t = 1;
            
            // Get the indices of the observer and the observed poses
            const unsigned int& pose_1_idx = associations[i].observer_pose_idx;
            const unsigned int& pose_2_idx = associations[i].measured_pose_idx;

            // Get the poses
            const cv::Mat& pose_1 = poses[pose_1_idx];
            const cv::Mat& pose_2 = poses[pose_2_idx];

            // How the pose 1 "sees" the pose 2
            const cv::Mat& pose2_wrt_pose1 = poses_measurements[i];

            // Compute error and Jacobian
            // (here we take J_1 as -J_2)
            computePoseErrorAndJacobian(pose_1, pose_2, pose2_wrt_pose1, J_2, error);

            // Some reference to the error
            const float& e_1 = error.at<float>(0,0);
            const float& e_2 = error.at<float>(1,0);
            const float& e_3 = error.at<float>(2,0);
            const float& e_4 = error.at<float>(3,0);
            const float& e_5 = error.at<float>(4,0);
            const float& e_6 = error.at<float>(5,0);
            const float& e_7 = error.at<float>(6,0);
            const float& e_8 = error.at<float>(7,0);
            const float& e_9 = error.at<float>(8,0);
            const float& e_10 = error.at<float>(9,0);
            const float& e_11 = error.at<float>(10,0);
            const float& e_12 = error.at<float>(11,0);

            // Compute the chi error
            current_chi = (e_1*omega_r)*e_1 + (e_2*omega_r)*e_2 + (e_3*omega_r)*e_3 + \
                            (e_4*omega_r)*e_4 + (e_5*omega_r)*e_5 + (e_6*omega_r)*e_6 + \
                            (e_7*omega_r)*e_7 + (e_8*omega_r)*e_8 + (e_9*omega_r)*e_9 + \
                            e_10*e_10 + e_11*e_11 + e_12*e_12;
            
            // Robust kernel
            if(current_chi > kernel_threshold) {
                omega_t = sqrt(kernel_threshold/current_chi);
                omega_r *= omega_t;
                current_chi = kernel_threshold;
            } else {
                ++n_inliers;
            }

            // Update the error evolution
            chi_tot += current_chi;

            // Some reference to jacobians (to take J_1 references just put a minus
            // sign in front of this references)
            const float& J_11 = J_2.at<float>(0,0);
            const float& J_12 = J_2.at<float>(0,1);
            const float& J_13 = J_2.at<float>(0,2);
            const float& J_14 = J_2.at<float>(0,3);
            const float& J_15 = J_2.at<float>(0,4);
            const float& J_16 = J_2.at<float>(0,5);
            const float& J_21 = J_2.at<float>(1,0);
            const float& J_22 = J_2.at<float>(1,1);
            const float& J_23 = J_2.at<float>(1,2);
            const float& J_24 = J_2.at<float>(1,3);
            const float& J_25 = J_2.at<float>(1,4);
            const float& J_26 = J_2.at<float>(1,5);
            const float& J_31 = J_2.at<float>(2,0);
            const float& J_32 = J_2.at<float>(2,1);
            const float& J_33 = J_2.at<float>(2,2);
            const float& J_34 = J_2.at<float>(2,3);
            const float& J_35 = J_2.at<float>(2,4);
            const float& J_36 = J_2.at<float>(2,5);
            const float& J_41 = J_2.at<float>(3,0);
            const float& J_42 = J_2.at<float>(3,1);
            const float& J_43 = J_2.at<float>(3,2);
            const float& J_44 = J_2.at<float>(3,3);
            const float& J_45 = J_2.at<float>(3,4);
            const float& J_46 = J_2.at<float>(3,5);
            const float& J_51 = J_2.at<float>(4,0);
            const float& J_52 = J_2.at<float>(4,1);
            const float& J_53 = J_2.at<float>(4,2);
            const float& J_54 = J_2.at<float>(4,3);
            const float& J_55 = J_2.at<float>(4,4);
            const float& J_56 = J_2.at<float>(4,5);
            const float& J_61 = J_2.at<float>(5,0);
            const float& J_62 = J_2.at<float>(5,1);
            const float& J_63 = J_2.at<float>(5,2);
            const float& J_64 = J_2.at<float>(5,3);
            const float& J_65 = J_2.at<float>(5,4);
            const float& J_66 = J_2.at<float>(5,5);
            const float& J_71 = J_2.at<float>(6,0);
            const float& J_72 = J_2.at<float>(6,1);
            const float& J_73 = J_2.at<float>(6,2);
            const float& J_74 = J_2.at<float>(6,3);
            const float& J_75 = J_2.at<float>(6,4);
            const float& J_76 = J_2.at<float>(6,5);
            const float& J_81 = J_2.at<float>(7,0);
            const float& J_82 = J_2.at<float>(7,1);
            const float& J_83 = J_2.at<float>(7,2);
            const float& J_84 = J_2.at<float>(7,3);
            const float& J_85 = J_2.at<float>(7,4);
            const float& J_86 = J_2.at<float>(7,5);
            const float& J_91 = J_2.at<float>(8,0);
            const float& J_92 = J_2.at<float>(8,1);
            const float& J_93 = J_2.at<float>(8,2);
            const float& J_94 = J_2.at<float>(8,3);
            const float& J_95 = J_2.at<float>(8,4);
            const float& J_96 = J_2.at<float>(8,5);
            const float& J_101 = J_2.at<float>(9,0);
            const float& J_102 = J_2.at<float>(9,1);
            const float& J_103 = J_2.at<float>(9,2);
            const float& J_104 = J_2.at<float>(9,3);
            const float& J_105 = J_2.at<float>(9,4);
            const float& J_106 = J_2.at<float>(9,5);
            const float& J_111 = J_2.at<float>(10,0);
            const float& J_112 = J_2.at<float>(10,1);
            const float& J_113 = J_2.at<float>(10,2);
            const float& J_114 = J_2.at<float>(10,3);
            const float& J_115 = J_2.at<float>(10,4);
            const float& J_116 = J_2.at<float>(10,5);
            const float& J_121 = J_2.at<float>(11,0);
            const float& J_122 = J_2.at<float>(11,1);
            const float& J_123 = J_2.at<float>(11,2);
            const float& J_124 = J_2.at<float>(11,3);
            const float& J_125 = J_2.at<float>(11,4);
            const float& J_126 = J_2.at<float>(11,5);

            // Retrieve the indices in the matrix H and vector b
            const unsigned int pose_1_matrix_idx = poseMatrixIdx(pose_1_idx);
            const unsigned int pose_2_matrix_idx = poseMatrixIdx(pose_2_idx);

            // Update the H matrix
            // J_1.t()*omega_r*J_1
            current_H_11 = \
                (-J_11*omega_r)*(-J_11) + (-J_21*omega_r)*(-J_21) + \
                (-J_31*omega_r)*(-J_31) + (-J_41*omega_r)*(-J_41) + \
                (-J_51*omega_r)*(-J_51) + (-J_61*omega_r)*(-J_61) + \
                (-J_71*omega_r)*(-J_71) + (-J_81*omega_r)*(-J_81) + \
                (-J_91*omega_r)*(-J_91) + (-J_101*omega_t)*(-J_101) + \
                (-J_111*omega_t)*(-J_111) + (-J_121*omega_t)*(-J_121);
            H.at<float>(pose_1_matrix_idx, pose_1_matrix_idx) += current_H_11;
            current_H_12 = \
                (-J_11*omega_r)*(-J_12) + (-J_21*omega_r)*(-J_22) + \
                (-J_31*omega_r)*(-J_32) + (-J_41*omega_r)*(-J_42) + \
                (-J_51*omega_r)*(-J_52) + (-J_61*omega_r)*(-J_62) + \
                (-J_71*omega_r)*(-J_72) + (-J_81*omega_r)*(-J_82) + \
                (-J_91*omega_r)*(-J_92) + (-J_101*omega_t)*(-J_102) + \
                (-J_111*omega_t)*(-J_112) + (-J_121*omega_t)*(-J_122);
            H.at<float>(pose_1_matrix_idx, pose_1_matrix_idx+1) += current_H_12; 
            current_H_13 = \
                (-J_11*omega_r)*(-J_13) + (-J_21*omega_r)*(-J_23) + \
                (-J_31*omega_r)*(-J_33) + (-J_41*omega_r)*(-J_43) + \
                (-J_51*omega_r)*(-J_53) + (-J_61*omega_r)*(-J_63) + \
                (-J_71*omega_r)*(-J_73) + (-J_81*omega_r)*(-J_83) + \
                (-J_91*omega_r)*(-J_93) + (-J_101*omega_t)*(-J_103) + \
                (-J_111*omega_t)*(-J_113) + (-J_121*omega_t)*(-J_123);
            H.at<float>(pose_1_matrix_idx, pose_1_matrix_idx+2) += current_H_13;
            current_H_14 = \
                (-J_11*omega_r)*(-J_14) + (-J_21*omega_r)*(-J_24) + \
                (-J_31*omega_r)*(-J_34) + (-J_41*omega_r)*(-J_44) + \
                (-J_51*omega_r)*(-J_54) + (-J_61*omega_r)*(-J_64) + \
                (-J_71*omega_r)*(-J_74) + (-J_81*omega_r)*(-J_84) + \
                (-J_91*omega_r)*(-J_94) + (-J_101*omega_t)*(-J_104) + \
                (-J_111*omega_t)*(-J_114) + (-J_121*omega_t)*(-J_124);
            H.at<float>(pose_1_matrix_idx, pose_1_matrix_idx+3) += current_H_14; 
            current_H_15 = \
                (-J_11*omega_r)*(-J_15) + (-J_21*omega_r)*(-J_25) + \
                (-J_31*omega_r)*(-J_35) + (-J_41*omega_r)*(-J_45) + \
                (-J_51*omega_r)*(-J_55) + (-J_61*omega_r)*(-J_65) + \
                (-J_71*omega_r)*(-J_75) + (-J_81*omega_r)*(-J_85) + \
                (-J_91*omega_r)*(-J_95) + (-J_101*omega_t)*(-J_105) + \
                (-J_111*omega_t)*(-J_115) + (-J_121*omega_t)*(-J_125);
            H.at<float>(pose_1_matrix_idx, pose_1_matrix_idx+4) += current_H_15; 
            current_H_16 = \
                (-J_11*omega_r)*(-J_16) + (-J_21*omega_r)*(-J_26) + \
                (-J_31*omega_r)*(-J_36) + (-J_41*omega_r)*(-J_46) + \
                (-J_51*omega_r)*(-J_56) + (-J_61*omega_r)*(-J_66) + \
                (-J_71*omega_r)*(-J_76) + (-J_81*omega_r)*(-J_86) + \
                (-J_91*omega_r)*(-J_96) + (-J_101*omega_t)*(-J_106) + \
                (-J_111*omega_t)*(-J_116) + (-J_121*omega_t)*(-J_126);
            H.at<float>(pose_1_matrix_idx, pose_1_matrix_idx+5) += current_H_16;
            current_H_21 = \
                (-J_12*omega_r)*(-J_11) + (-J_22*omega_r)*(-J_21) + \
                (-J_32*omega_r)*(-J_31) + (-J_42*omega_r)*(-J_41) + \
                (-J_52*omega_r)*(-J_51) + (-J_62*omega_r)*(-J_61) + \
                (-J_72*omega_r)*(-J_71) + (-J_82*omega_r)*(-J_81) + \
                (-J_92*omega_r)*(-J_91) + (-J_102*omega_t)*(-J_101) + \
                (-J_112*omega_t)*(-J_111) + (-J_122*omega_t)*(-J_121);
            H.at<float>(pose_1_matrix_idx+1, pose_1_matrix_idx) += current_H_21;
            current_H_22 = \
                (-J_12*omega_r)*(-J_12) + (-J_22*omega_r)*(-J_22) + \
                (-J_32*omega_r)*(-J_32) + (-J_42*omega_r)*(-J_42) + \
                (-J_52*omega_r)*(-J_52) + (-J_62*omega_r)*(-J_62) + \
                (-J_72*omega_r)*(-J_72) + (-J_82*omega_r)*(-J_82) + \
                (-J_92*omega_r)*(-J_92) + (-J_102*omega_t)*(-J_102) + \
                (-J_112*omega_t)*(-J_112) + (-J_122*omega_t)*(-J_122);
            H.at<float>(pose_1_matrix_idx+1, pose_1_matrix_idx+1) += current_H_22;
            current_H_23 = \
                (-J_12*omega_r)*(-J_13) + (-J_22*omega_r)*(-J_23) + \
                (-J_32*omega_r)*(-J_33) + (-J_42*omega_r)*(-J_43) + \
                (-J_52*omega_r)*(-J_53) + (-J_62*omega_r)*(-J_63) + \
                (-J_72*omega_r)*(-J_73) + (-J_82*omega_r)*(-J_83) + \
                (-J_92*omega_r)*(-J_93) + (-J_102*omega_t)*(-J_103) + \
                (-J_112*omega_t)*(-J_113) + (-J_122*omega_t)*(-J_123);
            H.at<float>(pose_1_matrix_idx+1, pose_1_matrix_idx+2) += current_H_23;
            current_H_24 = \
                (-J_12*omega_r)*(-J_14) + (-J_22*omega_r)*(-J_24) + \
                (-J_32*omega_r)*(-J_34) + (-J_42*omega_r)*(-J_44) + \
                (-J_52*omega_r)*(-J_54) + (-J_62*omega_r)*(-J_64) + \
                (-J_72*omega_r)*(-J_74) + (-J_82*omega_r)*(-J_84) + \
                (-J_92*omega_r)*(-J_94) + (-J_102*omega_t)*(-J_104) + \
                (-J_112*omega_t)*(-J_114) + (-J_122*omega_t)*(-J_124);
            H.at<float>(pose_1_matrix_idx+1, pose_1_matrix_idx+3) += current_H_24;
            current_H_25 = \
                (-J_12*omega_r)*(-J_15) + (-J_22*omega_r)*(-J_25) + \
                (-J_32*omega_r)*(-J_35) + (-J_42*omega_r)*(-J_45) + \
                (-J_52*omega_r)*(-J_55) + (-J_62*omega_r)*(-J_65) + \
                (-J_72*omega_r)*(-J_75) + (-J_82*omega_r)*(-J_85) + \
                (-J_92*omega_r)*(-J_95) + (-J_102*omega_t)*(-J_105) + \
                (-J_112*omega_t)*(-J_115) + (-J_122*omega_t)*(-J_125);
            H.at<float>(pose_1_matrix_idx+1, pose_1_matrix_idx+4) += current_H_25;
            current_H_26 = \
                (-J_12*omega_r)*(-J_16) + (-J_22*omega_r)*(-J_26) + \
                (-J_32*omega_r)*(-J_36) + (-J_42*omega_r)*(-J_46) + \
                (-J_52*omega_r)*(-J_56) + (-J_62*omega_r)*(-J_66) + \
                (-J_72*omega_r)*(-J_76) + (-J_82*omega_r)*(-J_86) + \
                (-J_92*omega_r)*(-J_96) + (-J_102*omega_t)*(-J_106) + \
                (-J_112*omega_t)*(-J_116) + (-J_122*omega_t)*(-J_126);
            H.at<float>(pose_1_matrix_idx+1, pose_1_matrix_idx+5) += current_H_26;
            current_H_31 = \
                (-J_13*omega_r)*(-J_11) + (-J_23*omega_r)*(-J_21) + \
                (-J_33*omega_r)*(-J_31) + (-J_43*omega_r)*(-J_41) + \
                (-J_53*omega_r)*(-J_51) + (-J_63*omega_r)*(-J_61) + \
                (-J_73*omega_r)*(-J_71) + (-J_83*omega_r)*(-J_81) + \
                (-J_93*omega_r)*(-J_91) + (-J_103*omega_t)*(-J_101) + \
                (-J_113*omega_t)*(-J_111) + (-J_123*omega_t)*(-J_121);
            H.at<float>(pose_1_matrix_idx+2, pose_1_matrix_idx) += current_H_31;
            current_H_32 = \
                (-J_13*omega_r)*(-J_12) + (-J_23*omega_r)*(-J_22) + \
                (-J_33*omega_r)*(-J_32) + (-J_43*omega_r)*(-J_42) + \
                (-J_53*omega_r)*(-J_52) + (-J_63*omega_r)*(-J_62) + \
                (-J_73*omega_r)*(-J_72) + (-J_83*omega_r)*(-J_82) + \
                (-J_93*omega_r)*(-J_92) + (-J_103*omega_t)*(-J_102) + \
                (-J_113*omega_t)*(-J_112) + (-J_123*omega_t)*(-J_122);
            H.at<float>(pose_1_matrix_idx+2, pose_1_matrix_idx+1) += current_H_32;
            current_H_33 = \
                (-J_13*omega_r)*(-J_13) + (-J_23*omega_r)*(-J_23) + \
                (-J_33*omega_r)*(-J_33) + (-J_43*omega_r)*(-J_43) + \
                (-J_53*omega_r)*(-J_53) + (-J_63*omega_r)*(-J_63) + \
                (-J_73*omega_r)*(-J_73) + (-J_83*omega_r)*(-J_83) + \
                (-J_93*omega_r)*(-J_93) + (-J_103*omega_t)*(-J_103) + \
                (-J_113*omega_t)*(-J_113) + (-J_123*omega_t)*(-J_123);
            H.at<float>(pose_1_matrix_idx+2, pose_1_matrix_idx+2) += current_H_33;
            current_H_34 = \
                (-J_13*omega_r)*(-J_14) + (-J_23*omega_r)*(-J_24) + \
                (-J_33*omega_r)*(-J_34) + (-J_43*omega_r)*(-J_44) + \
                (-J_53*omega_r)*(-J_54) + (-J_63*omega_r)*(-J_64) + \
                (-J_73*omega_r)*(-J_74) + (-J_83*omega_r)*(-J_84) + \
                (-J_93*omega_r)*(-J_94) + (-J_103*omega_t)*(-J_104) + \
                (-J_113*omega_t)*(-J_114) + (-J_123*omega_t)*(-J_124);
            H.at<float>(pose_1_matrix_idx+2, pose_1_matrix_idx+3) += current_H_34;
            current_H_35 = \
                (-J_13*omega_r)*(-J_15) + (-J_23*omega_r)*(-J_25) + \
                (-J_33*omega_r)*(-J_35) + (-J_43*omega_r)*(-J_45) + \
                (-J_53*omega_r)*(-J_55) + (-J_63*omega_r)*(-J_65) + \
                (-J_73*omega_r)*(-J_75) + (-J_83*omega_r)*(-J_85) + \
                (-J_93*omega_r)*(-J_95) + (-J_103*omega_t)*(-J_105) + \
                (-J_113*omega_t)*(-J_115) + (-J_123*omega_t)*(-J_125);
            H.at<float>(pose_1_matrix_idx+2, pose_1_matrix_idx+4) += current_H_35;
            current_H_36 = \
                (-J_13*omega_r)*(-J_16) + (-J_23*omega_r)*(-J_26) + \
                (-J_33*omega_r)*(-J_36) + (-J_43*omega_r)*(-J_46) + \
                (-J_53*omega_r)*(-J_56) + (-J_63*omega_r)*(-J_66) + \
                (-J_73*omega_r)*(-J_76) + (-J_83*omega_r)*(-J_86) + \
                (-J_93*omega_r)*(-J_96) + (-J_103*omega_t)*(-J_106) + \
                (-J_113*omega_t)*(-J_116) + (-J_123*omega_t)*(-J_126);
            H.at<float>(pose_1_matrix_idx+2, pose_1_matrix_idx+5) += current_H_36;
            current_H_41 = \
                (-J_14*omega_r)*(-J_11) + (-J_24*omega_r)*(-J_21) + \
                (-J_34*omega_r)*(-J_31) + (-J_44*omega_r)*(-J_41) + \
                (-J_54*omega_r)*(-J_51) + (-J_64*omega_r)*(-J_61) + \
                (-J_74*omega_r)*(-J_71) + (-J_84*omega_r)*(-J_81) + \
                (-J_94*omega_r)*(-J_91) + (-J_104*omega_t)*(-J_101) + \
                (-J_114*omega_t)*(-J_111) + (-J_124*omega_t)*(-J_121);
            H.at<float>(pose_1_matrix_idx+3, pose_1_matrix_idx) += current_H_41;
            current_H_42 = \
                (-J_14*omega_r)*(-J_12) + (-J_24*omega_r)*(-J_22) + \
                (-J_34*omega_r)*(-J_32) + (-J_44*omega_r)*(-J_42) + \
                (-J_54*omega_r)*(-J_52) + (-J_64*omega_r)*(-J_62) + \
                (-J_74*omega_r)*(-J_72) + (-J_84*omega_r)*(-J_82) + \
                (-J_94*omega_r)*(-J_92) + (-J_104*omega_t)*(-J_102) + \
                (-J_114*omega_t)*(-J_112) + (-J_124*omega_t)*(-J_122);
            H.at<float>(pose_1_matrix_idx+3, pose_1_matrix_idx+1) += current_H_42;
            current_H_43 = \
                (-J_14*omega_r)*(-J_13) + (-J_24*omega_r)*(-J_23) + \
                (-J_34*omega_r)*(-J_33) + (-J_44*omega_r)*(-J_43) + \
                (-J_54*omega_r)*(-J_53) + (-J_64*omega_r)*(-J_63) + \
                (-J_74*omega_r)*(-J_73) + (-J_84*omega_r)*(-J_83) + \
                (-J_94*omega_r)*(-J_93) + (-J_104*omega_t)*(-J_103) + \
                (-J_114*omega_t)*(-J_113) + (-J_124*omega_t)*(-J_123);
            H.at<float>(pose_1_matrix_idx+3, pose_1_matrix_idx+2) += current_H_43;
            current_H_44 = \
                (-J_14*omega_r)*(-J_14) + (-J_24*omega_r)*(-J_24) + \
                (-J_34*omega_r)*(-J_34) + (-J_44*omega_r)*(-J_44) + \
                (-J_54*omega_r)*(-J_54) + (-J_64*omega_r)*(-J_64) + \
                (-J_74*omega_r)*(-J_74) + (-J_84*omega_r)*(-J_84) + \
                (-J_94*omega_r)*(-J_94) + (-J_104*omega_t)*(-J_104) + \
                (-J_114*omega_t)*(-J_114) + (-J_124*omega_t)*(-J_124);
            H.at<float>(pose_1_matrix_idx+3, pose_1_matrix_idx+3) += current_H_44;
            current_H_45 = \
                (-J_14*omega_r)*(-J_15) + (-J_24*omega_r)*(-J_25) + \
                (-J_34*omega_r)*(-J_35) + (-J_44*omega_r)*(-J_45) + \
                (-J_54*omega_r)*(-J_55) + (-J_64*omega_r)*(-J_65) + \
                (-J_74*omega_r)*(-J_75) + (-J_84*omega_r)*(-J_85) + \
                (-J_94*omega_r)*(-J_95) + (-J_104*omega_t)*(-J_105) + \
                (-J_114*omega_t)*(-J_115) + (-J_124*omega_t)*(-J_125);
            H.at<float>(pose_1_matrix_idx+3, pose_1_matrix_idx+4) += current_H_45;
            current_H_46 = \
                (-J_14*omega_r)*(-J_16) + (-J_24*omega_r)*(-J_26) + \
                (-J_34*omega_r)*(-J_36) + (-J_44*omega_r)*(-J_46) + \
                (-J_54*omega_r)*(-J_56) + (-J_64*omega_r)*(-J_66) + \
                (-J_74*omega_r)*(-J_76) + (-J_84*omega_r)*(-J_86) + \
                (-J_94*omega_r)*(-J_96) + (-J_104*omega_t)*(-J_106) + \
                (-J_114*omega_t)*(-J_116) + (-J_124*omega_t)*(-J_126);
            H.at<float>(pose_1_matrix_idx+3, pose_1_matrix_idx+5) += current_H_46;
            current_H_51 = \
                (-J_15*omega_r)*(-J_11) + (-J_25*omega_r)*(-J_21) + \
                (-J_35*omega_r)*(-J_31) + (-J_45*omega_r)*(-J_41) + \
                (-J_55*omega_r)*(-J_51) + (-J_65*omega_r)*(-J_61) + \
                (-J_75*omega_r)*(-J_71) + (-J_85*omega_r)*(-J_81) + \
                (-J_95*omega_r)*(-J_91) + (-J_105*omega_t)*(-J_101) + \
                (-J_115*omega_t)*(-J_111) + (-J_125*omega_t)*(-J_121);
            H.at<float>(pose_1_matrix_idx+4, pose_1_matrix_idx) += current_H_51;
            current_H_52 = \
                (-J_15*omega_r)*(-J_12) + (-J_25*omega_r)*(-J_22) + \
                (-J_35*omega_r)*(-J_32) + (-J_45*omega_r)*(-J_42) + \
                (-J_55*omega_r)*(-J_52) + (-J_65*omega_r)*(-J_62) + \
                (-J_75*omega_r)*(-J_72) + (-J_85*omega_r)*(-J_82) + \
                (-J_95*omega_r)*(-J_92) + (-J_105*omega_t)*(-J_102) + \
                (-J_115*omega_t)*(-J_112) + (-J_125*omega_t)*(-J_122);
            H.at<float>(pose_1_matrix_idx+4, pose_1_matrix_idx+1) += current_H_52;
            current_H_53 = \
                (-J_15*omega_r)*(-J_13) + (-J_25*omega_r)*(-J_23) + \
                (-J_35*omega_r)*(-J_33) + (-J_45*omega_r)*(-J_43) + \
                (-J_55*omega_r)*(-J_53) + (-J_65*omega_r)*(-J_63) + \
                (-J_75*omega_r)*(-J_73) + (-J_85*omega_r)*(-J_83) + \
                (-J_95*omega_r)*(-J_93) + (-J_105*omega_t)*(-J_103) + \
                (-J_115*omega_t)*(-J_113) + (-J_125*omega_t)*(-J_123);
            H.at<float>(pose_1_matrix_idx+4, pose_1_matrix_idx+2) += current_H_53;
            current_H_54 = \
                (-J_15*omega_r)*(-J_14) + (-J_25*omega_r)*(-J_24) + \
                (-J_35*omega_r)*(-J_34) + (-J_45*omega_r)*(-J_44) + \
                (-J_55*omega_r)*(-J_54) + (-J_65*omega_r)*(-J_64) + \
                (-J_75*omega_r)*(-J_74) + (-J_85*omega_r)*(-J_84) + \
                (-J_95*omega_r)*(-J_94) + (-J_105*omega_t)*(-J_104) + \
                (-J_115*omega_t)*(-J_114) + (-J_125*omega_t)*(-J_124);
            H.at<float>(pose_1_matrix_idx+4, pose_1_matrix_idx+3) += current_H_54;
            current_H_55 = \
                (-J_15*omega_r)*(-J_15) + (-J_25*omega_r)*(-J_25) + \
                (-J_35*omega_r)*(-J_35) + (-J_45*omega_r)*(-J_45) + \
                (-J_55*omega_r)*(-J_55) + (-J_65*omega_r)*(-J_65) + \
                (-J_75*omega_r)*(-J_75) + (-J_85*omega_r)*(-J_85) + \
                (-J_95*omega_r)*(-J_95) + (-J_105*omega_t)*(-J_105) + \
                (-J_115*omega_t)*(-J_115) + (-J_125*omega_t)*(-J_125);
            H.at<float>(pose_1_matrix_idx+4, pose_1_matrix_idx+4) += current_H_55;
            current_H_56 = \
                (-J_15*omega_r)*(-J_16) + (-J_25*omega_r)*(-J_26) + \
                (-J_35*omega_r)*(-J_36) + (-J_45*omega_r)*(-J_46) + \
                (-J_55*omega_r)*(-J_56) + (-J_65*omega_r)*(-J_66) + \
                (-J_75*omega_r)*(-J_76) + (-J_85*omega_r)*(-J_86) + \
                (-J_95*omega_r)*(-J_96) + (-J_105*omega_t)*(-J_106) + \
                (-J_115*omega_t)*(-J_116) + (-J_125*omega_t)*(-J_126);
            H.at<float>(pose_1_matrix_idx+4, pose_1_matrix_idx+5) += current_H_56;
            current_H_61 = \
                (-J_16*omega_r)*(-J_11) + (-J_26*omega_r)*(-J_21) + \
                (-J_36*omega_r)*(-J_31) + (-J_46*omega_r)*(-J_41) + \
                (-J_56*omega_r)*(-J_51) + (-J_66*omega_r)*(-J_61) + \
                (-J_76*omega_r)*(-J_71) + (-J_86*omega_r)*(-J_81) + \
                (-J_96*omega_r)*(-J_91) + (-J_106*omega_t)*(-J_101) + \
                (-J_116*omega_t)*(-J_111) + (-J_126*omega_t)*(-J_121);
            H.at<float>(pose_1_matrix_idx+5, pose_1_matrix_idx) += current_H_61;
            current_H_62 = \
                (-J_16*omega_r)*(-J_12) + (-J_26*omega_r)*(-J_22) + \
                (-J_36*omega_r)*(-J_32) + (-J_46*omega_r)*(-J_42) + \
                (-J_56*omega_r)*(-J_52) + (-J_66*omega_r)*(-J_62) + \
                (-J_76*omega_r)*(-J_72) + (-J_86*omega_r)*(-J_82) + \
                (-J_96*omega_r)*(-J_92) + (-J_106*omega_t)*(-J_102) + \
                (-J_116*omega_t)*(-J_112) + (-J_126*omega_t)*(-J_122);
            H.at<float>(pose_1_matrix_idx+5, pose_1_matrix_idx+1) += current_H_62;
            current_H_63 = \
                (-J_16*omega_r)*(-J_13) + (-J_26*omega_r)*(-J_23) + \
                (-J_36*omega_r)*(-J_33) + (-J_46*omega_r)*(-J_43) + \
                (-J_56*omega_r)*(-J_53) + (-J_66*omega_r)*(-J_63) + \
                (-J_76*omega_r)*(-J_73) + (-J_86*omega_r)*(-J_83) + \
                (-J_96*omega_r)*(-J_93) + (-J_106*omega_t)*(-J_103) + \
                (-J_116*omega_t)*(-J_113) + (-J_126*omega_t)*(-J_123);
            H.at<float>(pose_1_matrix_idx+5, pose_1_matrix_idx+2) += current_H_63;
            current_H_64 = \
                (-J_16*omega_r)*(-J_14) + (-J_26*omega_r)*(-J_24) + \
                (-J_36*omega_r)*(-J_34) + (-J_46*omega_r)*(-J_44) + \
                (-J_56*omega_r)*(-J_54) + (-J_66*omega_r)*(-J_64) + \
                (-J_76*omega_r)*(-J_74) + (-J_86*omega_r)*(-J_84) + \
                (-J_96*omega_r)*(-J_94) + (-J_106*omega_t)*(-J_104) + \
                (-J_116*omega_t)*(-J_114) + (-J_126*omega_t)*(-J_124);
            H.at<float>(pose_1_matrix_idx+5, pose_1_matrix_idx+3) += current_H_64;
            current_H_65 = \
                (-J_16*omega_r)*(-J_15) + (-J_26*omega_r)*(-J_25) + \
                (-J_36*omega_r)*(-J_35) + (-J_46*omega_r)*(-J_45) + \
                (-J_56*omega_r)*(-J_55) + (-J_66*omega_r)*(-J_65) + \
                (-J_76*omega_r)*(-J_75) + (-J_86*omega_r)*(-J_85) + \
                (-J_96*omega_r)*(-J_95) + (-J_106*omega_t)*(-J_105) + \
                (-J_116*omega_t)*(-J_115) + (-J_126*omega_t)*(-J_125);
            H.at<float>(pose_1_matrix_idx+5, pose_1_matrix_idx+4) += current_H_65;
            current_H_66 = \
                (-J_16*omega_r)*(-J_16) + (-J_26*omega_r)*(-J_26) + \
                (-J_36*omega_r)*(-J_36) + (-J_46*omega_r)*(-J_46) + \
                (-J_56*omega_r)*(-J_56) + (-J_66*omega_r)*(-J_66) + \
                (-J_76*omega_r)*(-J_76) + (-J_86*omega_r)*(-J_86) + \
                (-J_96*omega_r)*(-J_96) + (-J_106*omega_t)*(-J_106) + \
                (-J_116*omega_t)*(-J_116) + (-J_126*omega_t)*(-J_126);
            H.at<float>(pose_1_matrix_idx+5, pose_1_matrix_idx+5) += current_H_66;

            // J_1.t()*omega_r*J_2
            H.at<float>(pose_1_matrix_idx, pose_2_matrix_idx) -= current_H_11;
            H.at<float>(pose_1_matrix_idx, pose_2_matrix_idx+1) -= current_H_12;
            H.at<float>(pose_1_matrix_idx, pose_2_matrix_idx+2) -= current_H_13;
            H.at<float>(pose_1_matrix_idx, pose_2_matrix_idx+3) -= current_H_14;
            H.at<float>(pose_1_matrix_idx, pose_2_matrix_idx+4) -= current_H_15;
            H.at<float>(pose_1_matrix_idx, pose_2_matrix_idx+5) -= current_H_16;
            H.at<float>(pose_1_matrix_idx+1, pose_2_matrix_idx) -= current_H_21;
            H.at<float>(pose_1_matrix_idx+1, pose_2_matrix_idx+1) -= current_H_22;
            H.at<float>(pose_1_matrix_idx+1, pose_2_matrix_idx+2) -= current_H_23;
            H.at<float>(pose_1_matrix_idx+1, pose_2_matrix_idx+3) -= current_H_24;
            H.at<float>(pose_1_matrix_idx+1, pose_2_matrix_idx+4) -= current_H_25;
            H.at<float>(pose_1_matrix_idx+1, pose_2_matrix_idx+5) -= current_H_26;
            H.at<float>(pose_1_matrix_idx+2, pose_2_matrix_idx) -= current_H_31;
            H.at<float>(pose_1_matrix_idx+2, pose_2_matrix_idx+1) -= current_H_32;
            H.at<float>(pose_1_matrix_idx+2, pose_2_matrix_idx+2) -= current_H_33;
            H.at<float>(pose_1_matrix_idx+2, pose_2_matrix_idx+3) -= current_H_34;
            H.at<float>(pose_1_matrix_idx+2, pose_2_matrix_idx+4) -= current_H_35;
            H.at<float>(pose_1_matrix_idx+2, pose_2_matrix_idx+5) -= current_H_36;
            H.at<float>(pose_1_matrix_idx+3, pose_2_matrix_idx) -= current_H_41;
            H.at<float>(pose_1_matrix_idx+3, pose_2_matrix_idx+1) -= current_H_42;
            H.at<float>(pose_1_matrix_idx+3, pose_2_matrix_idx+2) -= current_H_43;
            H.at<float>(pose_1_matrix_idx+3, pose_2_matrix_idx+3) -= current_H_44;
            H.at<float>(pose_1_matrix_idx+3, pose_2_matrix_idx+4) -= current_H_45;
            H.at<float>(pose_1_matrix_idx+3, pose_2_matrix_idx+5) -= current_H_46;
            H.at<float>(pose_1_matrix_idx+4, pose_2_matrix_idx) -= current_H_51;
            H.at<float>(pose_1_matrix_idx+4, pose_2_matrix_idx+1) -= current_H_52;
            H.at<float>(pose_1_matrix_idx+4, pose_2_matrix_idx+2) -= current_H_53;
            H.at<float>(pose_1_matrix_idx+4, pose_2_matrix_idx+3) -= current_H_54;
            H.at<float>(pose_1_matrix_idx+4, pose_2_matrix_idx+4) -= current_H_55;
            H.at<float>(pose_1_matrix_idx+4, pose_2_matrix_idx+5) -= current_H_56;
            H.at<float>(pose_1_matrix_idx+5, pose_2_matrix_idx) -= current_H_61;
            H.at<float>(pose_1_matrix_idx+5, pose_2_matrix_idx+1) -= current_H_62;
            H.at<float>(pose_1_matrix_idx+5, pose_2_matrix_idx+2) -= current_H_63;
            H.at<float>(pose_1_matrix_idx+5, pose_2_matrix_idx+3) -= current_H_64;
            H.at<float>(pose_1_matrix_idx+5, pose_2_matrix_idx+4) -= current_H_65;
            H.at<float>(pose_1_matrix_idx+5, pose_2_matrix_idx+5) -= current_H_66;

            // J_2.t()*omega_r*J_1
            H.at<float>(pose_2_matrix_idx, pose_1_matrix_idx) += current_H_11;
            H.at<float>(pose_2_matrix_idx, pose_1_matrix_idx+1) += current_H_12;
            H.at<float>(pose_2_matrix_idx, pose_1_matrix_idx+2) += current_H_13;
            H.at<float>(pose_2_matrix_idx, pose_1_matrix_idx+3) += current_H_14;
            H.at<float>(pose_2_matrix_idx, pose_1_matrix_idx+4) += current_H_15;
            H.at<float>(pose_2_matrix_idx, pose_1_matrix_idx+5) += current_H_16;
            H.at<float>(pose_2_matrix_idx+1, pose_1_matrix_idx) += current_H_21;
            H.at<float>(pose_2_matrix_idx+1, pose_1_matrix_idx+1) += current_H_22;
            H.at<float>(pose_2_matrix_idx+1, pose_1_matrix_idx+2) += current_H_23;
            H.at<float>(pose_2_matrix_idx+1, pose_1_matrix_idx+3) += current_H_24;
            H.at<float>(pose_2_matrix_idx+1, pose_1_matrix_idx+4) += current_H_25;
            H.at<float>(pose_2_matrix_idx+1, pose_1_matrix_idx+5) += current_H_26;
            H.at<float>(pose_2_matrix_idx+2, pose_1_matrix_idx) += current_H_31;
            H.at<float>(pose_2_matrix_idx+2, pose_1_matrix_idx+1) += current_H_32;
            H.at<float>(pose_2_matrix_idx+2, pose_1_matrix_idx+2) += current_H_33;
            H.at<float>(pose_2_matrix_idx+2, pose_1_matrix_idx+3) += current_H_34;
            H.at<float>(pose_2_matrix_idx+2, pose_1_matrix_idx+4) += current_H_35;
            H.at<float>(pose_2_matrix_idx+2, pose_1_matrix_idx+5) += current_H_36;
            H.at<float>(pose_2_matrix_idx+3, pose_1_matrix_idx) += current_H_41;
            H.at<float>(pose_2_matrix_idx+3, pose_1_matrix_idx+1) += current_H_42;
            H.at<float>(pose_2_matrix_idx+3, pose_1_matrix_idx+2) += current_H_43;
            H.at<float>(pose_2_matrix_idx+3, pose_1_matrix_idx+3) += current_H_44;
            H.at<float>(pose_2_matrix_idx+3, pose_1_matrix_idx+4) += current_H_45;
            H.at<float>(pose_2_matrix_idx+3, pose_1_matrix_idx+5) += current_H_46;
            H.at<float>(pose_2_matrix_idx+4, pose_1_matrix_idx) += current_H_51;
            H.at<float>(pose_2_matrix_idx+4, pose_1_matrix_idx+1) += current_H_52;
            H.at<float>(pose_2_matrix_idx+4, pose_1_matrix_idx+2) += current_H_53;
            H.at<float>(pose_2_matrix_idx+4, pose_1_matrix_idx+3) += current_H_54;
            H.at<float>(pose_2_matrix_idx+4, pose_1_matrix_idx+4) += current_H_55;
            H.at<float>(pose_2_matrix_idx+4, pose_1_matrix_idx+5) += current_H_56;
            H.at<float>(pose_2_matrix_idx+5, pose_1_matrix_idx) += current_H_61;
            H.at<float>(pose_2_matrix_idx+5, pose_1_matrix_idx+1) += current_H_62;
            H.at<float>(pose_2_matrix_idx+5, pose_1_matrix_idx+2) += current_H_63;
            H.at<float>(pose_2_matrix_idx+5, pose_1_matrix_idx+3) += current_H_64;
            H.at<float>(pose_2_matrix_idx+5, pose_1_matrix_idx+4) += current_H_65;
            H.at<float>(pose_2_matrix_idx+5, pose_1_matrix_idx+5) += current_H_66;

            // J_2.t()*omega_r*J_2
            H.at<float>(pose_2_matrix_idx, pose_2_matrix_idx) += current_H_11;
            H.at<float>(pose_2_matrix_idx, pose_2_matrix_idx+1) += current_H_12;
            H.at<float>(pose_2_matrix_idx, pose_2_matrix_idx+2) += current_H_13;
            H.at<float>(pose_2_matrix_idx, pose_2_matrix_idx+3) += current_H_14;
            H.at<float>(pose_2_matrix_idx, pose_2_matrix_idx+4) += current_H_15;
            H.at<float>(pose_2_matrix_idx, pose_2_matrix_idx+5) += current_H_16;
            H.at<float>(pose_2_matrix_idx+1, pose_2_matrix_idx) += current_H_21;
            H.at<float>(pose_2_matrix_idx+1, pose_2_matrix_idx+1) += current_H_22;
            H.at<float>(pose_2_matrix_idx+1, pose_2_matrix_idx+2) += current_H_23;
            H.at<float>(pose_2_matrix_idx+1, pose_2_matrix_idx+3) += current_H_24;
            H.at<float>(pose_2_matrix_idx+1, pose_2_matrix_idx+4) += current_H_25;
            H.at<float>(pose_2_matrix_idx+1, pose_2_matrix_idx+5) += current_H_26;
            H.at<float>(pose_2_matrix_idx+2, pose_2_matrix_idx) += current_H_31;
            H.at<float>(pose_2_matrix_idx+2, pose_2_matrix_idx+1) += current_H_32;
            H.at<float>(pose_2_matrix_idx+2, pose_2_matrix_idx+2) += current_H_33;
            H.at<float>(pose_2_matrix_idx+2, pose_2_matrix_idx+3) += current_H_34;
            H.at<float>(pose_2_matrix_idx+2, pose_2_matrix_idx+4) += current_H_35;
            H.at<float>(pose_2_matrix_idx+2, pose_2_matrix_idx+5) += current_H_36;
            H.at<float>(pose_2_matrix_idx+3, pose_2_matrix_idx) += current_H_41;
            H.at<float>(pose_2_matrix_idx+3, pose_2_matrix_idx+1) += current_H_42;
            H.at<float>(pose_2_matrix_idx+3, pose_2_matrix_idx+2) += current_H_43;
            H.at<float>(pose_2_matrix_idx+3, pose_2_matrix_idx+3) += current_H_44;
            H.at<float>(pose_2_matrix_idx+3, pose_2_matrix_idx+4) += current_H_45;
            H.at<float>(pose_2_matrix_idx+3, pose_2_matrix_idx+5) += current_H_46;
            H.at<float>(pose_2_matrix_idx+4, pose_2_matrix_idx) += current_H_51;
            H.at<float>(pose_2_matrix_idx+4, pose_2_matrix_idx+1) += current_H_52;
            H.at<float>(pose_2_matrix_idx+4, pose_2_matrix_idx+2) += current_H_53;
            H.at<float>(pose_2_matrix_idx+4, pose_2_matrix_idx+3) += current_H_54;
            H.at<float>(pose_2_matrix_idx+4, pose_2_matrix_idx+4) += current_H_55;
            H.at<float>(pose_2_matrix_idx+4, pose_2_matrix_idx+5) += current_H_56;
            H.at<float>(pose_2_matrix_idx+5, pose_2_matrix_idx) += current_H_61;
            H.at<float>(pose_2_matrix_idx+5, pose_2_matrix_idx+1) += current_H_62;
            H.at<float>(pose_2_matrix_idx+5, pose_2_matrix_idx+2) += current_H_63;
            H.at<float>(pose_2_matrix_idx+5, pose_2_matrix_idx+3) += current_H_64;
            H.at<float>(pose_2_matrix_idx+5, pose_2_matrix_idx+4) += current_H_65;
            H.at<float>(pose_2_matrix_idx+5, pose_2_matrix_idx+5) += current_H_66;

            // Update the b vector
            // J_1.t()*omega_r*error
            
            current_b_1 = \
                (-J_11*omega_r)*e_1 + (-J_21*omega_r)*e_2 + (-J_31*omega_r)*e_3 + \
                (-J_41*omega_r)*e_4 + (-J_51*omega_r)*e_5 + (-J_61*omega_r)*e_6 + \
                (-J_71*omega_r)*e_7 + (-J_81*omega_r)*e_8 + (-J_91*omega_r)*e_9 + \
                (-J_101*omega_t)*e_10 + (-J_111*omega_t)*e_11 + (-J_121*omega_t)*e_12;
            current_b_2 = \
                (-J_12*omega_r)*e_1 + (-J_22*omega_r)*e_2 + (-J_32*omega_r)*e_3 + \
                (-J_42*omega_r)*e_4 + (-J_52*omega_r)*e_5 + (-J_62*omega_r)*e_6 + \
                (-J_72*omega_r)*e_7 + (-J_82*omega_r)*e_8 + (-J_92*omega_r)*e_9 + \
                (-J_102*omega_t)*e_10 + (-J_112*omega_t)*e_11 + (-J_122*omega_t)*e_12;
            current_b_3 = \
                (-J_13*omega_r)*e_1 + (-J_23*omega_r)*e_2 + (-J_33*omega_r)*e_3 + \
                (-J_43*omega_r)*e_4 + (-J_53*omega_r)*e_5 + (-J_63*omega_r)*e_6 + \
                (-J_73*omega_r)*e_7 + (-J_83*omega_r)*e_8 + (-J_93*omega_r)*e_9 + \
                (-J_103*omega_t)*e_10 + (-J_113*omega_t)*e_11 + (-J_123*omega_t)*e_12;
            current_b_4 = \
                (-J_14*omega_r)*e_1 + (-J_24*omega_r)*e_2 + (-J_34*omega_r)*e_3 + \
                (-J_44*omega_r)*e_4 + (-J_54*omega_r)*e_5 + (-J_64*omega_r)*e_6 + \
                (-J_74*omega_r)*e_7 + (-J_84*omega_r)*e_8 + (-J_94*omega_r)*e_9 + \
                (-J_104*omega_t)*e_10 + (-J_114*omega_t)*e_11 + (-J_124*omega_t)*e_12;
            current_b_5 = \
                (-J_15*omega_r)*e_1 + (-J_25*omega_r)*e_2 + (-J_35*omega_r)*e_3 + \
                (-J_45*omega_r)*e_4 + (-J_55*omega_r)*e_5 + (-J_65*omega_r)*e_6 + \
                (-J_75*omega_r)*e_7 + (-J_85*omega_r)*e_8 + (-J_95*omega_r)*e_9 + \
                (-J_105*omega_t)*e_10 + (-J_115*omega_t)*e_11 + (-J_125*omega_t)*e_12;
            current_b_6 = \
                (-J_16*omega_r)*e_1 + (-J_26*omega_r)*e_2 + (-J_36*omega_r)*e_3 + \
                (-J_46*omega_r)*e_4 + (-J_56*omega_r)*e_5 + (-J_66*omega_r)*e_6 + \
                (-J_76*omega_r)*e_7 + (-J_86*omega_r)*e_8 + (-J_96*omega_r)*e_9 + \
                (-J_106*omega_t)*e_10 + (-J_116*omega_t)*e_11 + (-J_126*omega_t)*e_12;
            b.at<float>(pose_1_matrix_idx, 0) += current_b_1;
            b.at<float>(pose_1_matrix_idx+1, 0) += current_b_2;
            b.at<float>(pose_1_matrix_idx+2, 0) += current_b_3;
            b.at<float>(pose_1_matrix_idx+3, 0) += current_b_4;
            b.at<float>(pose_1_matrix_idx+4, 0) += current_b_5;
            b.at<float>(pose_1_matrix_idx+5, 0) += current_b_6;

            // J_2.t()*omega_r*error
            b.at<float>(pose_2_matrix_idx, 0) -= current_b_1;
            b.at<float>(pose_2_matrix_idx+1, 0) -= current_b_2;
            b.at<float>(pose_2_matrix_idx+2, 0) -= current_b_3;
            b.at<float>(pose_2_matrix_idx+3, 0) -= current_b_4;
            b.at<float>(pose_2_matrix_idx+4, 0) -= current_b_5;
            b.at<float>(pose_2_matrix_idx+5, 0) -= current_b_6;
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
    *   img_rows: #rows in the image plane pixels matrix
    *   img_cols: #cols in the image plane pixels matrix
    *  Outputs:
    *   true if the projection is valid, false otherwise
    */
    bool State::computeProjectionErrorAndJacobian(const cv::Mat& pose, \
                        const cv::Point3f& landmark_pose, \
                        const cv::KeyPoint& img_point, const cv::Mat& K,   
                        cv::Mat& J_pose, cv::Mat& J_landmark, cv::Mat& error, \
                        const float& img_rows, const float& img_cols) {

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
        // TODO: assicurati che img_cols e img_rows siano corretti
        if (z_hat_x < 0 || 
            z_hat_x > img_cols ||
            z_hat_y < 0 || 
            z_hat_y > img_rows)
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
    *           first pose (we do not compute it explicitly, it is -J_2)
    *   J_2: 12x6 derivative w.r.t the error and a perturbation of the
    *           second pose (we assume it is already instantiated as a 12x6 cv::Mat)
    *   error: 12x1 difference between prediction and measurement, vectorized,
    *           (we assume it is already instantiated as a 12x1 cv::Mat)
    */
    void State::computePoseErrorAndJacobian(const cv::Mat& pose_1, \
                        const cv::Mat& pose_2, \
                        const cv::Mat& pose2_wrt_pose1, \
                        cv::Mat& J_2, cv::Mat& error) {
        
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
        // (it is -J_2, so it is not computed explicitly)

        // Compute the error in the rotation
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