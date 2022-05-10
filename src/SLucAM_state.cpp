//
// SLucAM_state.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_state.h>
#include <SLucAM_keyframe.h>
#include <SLucAM_initialization.h>
#include <SLucAM_geometry.h>
#include <SLucAM_image.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>
#include <map>
#include <set>



// -----------------------------------------------------------------------------
// Implementation of State class methods
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Base constructor.
    */
    State::State() {
        this->_next_measurement_idx = 0;
        this->_distorsion_coefficients = cv::Mat();
    }


    /*
    * This constructor allows us to reserve some expected space for the vector
    * of poses and the vector of landmarks, just for optimization. It also need
    * the camera matrix K.
    */
    State::State(cv::Mat& K, std::vector<Measurement>& measurements, \
                const unsigned int expected_poses, \
                const unsigned int expected_landmarks) 
        : State() {
        
        this->_K = K;
        this->_measurements = measurements;
        this->_poses.reserve(expected_poses);
        this->_keypoints.reserve(expected_landmarks);
        this->_keyframes.reserve(expected_poses);

    }


    /*
    * This constructor is like the previous one but with the distorsion
    * coefficients.
    */
    State::State(cv::Mat& K, cv::Mat& distorsion_coefficients, \
                std::vector<Measurement>& measurements, \
                const unsigned int expected_poses, \
                const unsigned int expected_landmarks)  
        : State(K, measurements, expected_poses, expected_landmarks) {

        this->_distorsion_coefficients = distorsion_coefficients;

    }



    /*
    * This function performs the initialization of the state. It try to 
    * perform initialization between the measurement in position 0 of the 
    * measurements vector and the first measurement that have enough translation
    * between them. If no such measurement is found then is returned false.  
    */
    bool State::initializeState(Matcher& matcher, \
                                const unsigned int n_iters_ransac, \
                                const float& parallax_threshold, \
                                const bool verbose) {

        // If we do not have enough measurements refuse initialization
        if(this->_measurements.size() < 2) return false; 

        // Initialization
        const cv::Mat& K = this->_K;
        bool initialization_performed = false;
        cv::Mat predicted_pose;
        std::vector<cv::Point3f> triangulated_points;
        std::vector<std::pair<unsigned int, unsigned int>> meas1_points_associations;
        std::vector<std::pair<unsigned int, unsigned int>> meas2_points_associations;
        std::vector<cv::DMatch> matches;
        std::vector<unsigned int> matches_filter;

        // Take the first measurement
        const Measurement& meas1 = getNextMeasurement();

        // While a good measurement couple is not found try to find it
        while(!initialization_performed && (reaminingMeasurements() != 0)) {
            
            const Measurement& meas2 = getNextMeasurement();
            initialization_performed = initialize(meas1, meas2, matcher, \
                        K, predicted_pose, matches, matches_filter, \
                        triangulated_points, n_iters_ransac, \
                        parallax_threshold, verbose);
            
            if(verbose && initialization_performed) {
                if(!visualize_matches(meas1, meas2, matches, matches_filter)) {
                    std::cout << "Unable to visualize the match" << std::endl;
                }
            }

        }

        // If we close the loop because we have no more measurements, return false
        if(!initialization_performed) return false;

        // Save the 3D points filtering out the invalid triangulations
        // and create the associations vector for each measurement
        associateNewKeypoints(triangulated_points, matches, matches_filter, \
                                this->_keypoints, meas1_points_associations, \
                                meas2_points_associations, verbose);

        // Create the first pose (we assume it at the origin) and use it as new keyframe
        this->_poses.emplace_back(cv::Mat::eye(4,4,CV_32F));
        addKeyFrame(0, 0, meas1_points_associations, -1, verbose);

        // Save the predicted pose and use it as new keyframe
        this->_poses.emplace_back(predicted_pose);
        addKeyFrame(this->_next_measurement_idx-1, this->_poses.size()-1, \
                    meas2_points_associations, this->_keyframes.size()-1, \
                    verbose);
        
        // Optimize initial map
        performTotalBA(20, false);

        // Scale compute pose's position and points to avoid too large distances
        cv::Mat& pose1 = this->_poses[1];
        const unsigned int n_points = this->_keypoints.size();
        const float scale_factor = compute_median_distance_cam_points(this->_keypoints, pose1);
        pose1.at<float>(0,3) /= scale_factor;
        pose1.at<float>(1,3) /= scale_factor;
        pose1.at<float>(2,3) /= scale_factor;
        for(unsigned int j=0; j<n_points; ++j) {
            this->_keypoints[j].setPosition(this->_keypoints[j].getPosition()/scale_factor);
        }

        return true;

    }



    /*
    * This function takes the next measurement to analyze and:
    * 1. Check what landmarks that we already know are seen from the
    *    new measurement (the ones that match with the previous one)
    *    and update the _landmark_observations structure
    * 2. Perform Projective ICP to determine the new pose and update
    *    the _pose_observations and _poses_measurements structures
    *    (it also add it to the _poses vector)
    * 3. If specified (by triangulate_new_points) detect new landmarks
    *    seen from the new measurement never seen before and triangulate 
    *    them
    * Outputs:
    *   false in case of error (when there is no more measurement to
    *   integrate)
    */
    bool State::integrateNewMeasurement(Matcher& matcher, \
                                        const bool& triangulate_new_points, \
                                        const unsigned int& local_map_size, \
                                        const float& kernel_threshold_POSIT, \
                                        const float& inliers_threshold_POSIT, \
                                        const float& parallax_threshold, \
                                        const float& new_landmark_threshold, \
                                        const bool verbose) {

        // If we have no more measurement to integrate, return error
        if(this->reaminingMeasurements() == 0) return false;

        if(verbose) {
            std::cout << "-> INTEGRATING NEW MEASUREMENT (" << \
                    this->_next_measurement_idx << "/" << \
                    this->_measurements.size()-1 << "): ";
        }

        // Take the measurement to integrate
        const SLucAM::Measurement& meas_to_integrate = getNextMeasurement();

        // Predict the new pose (use previous keyframe's pose as initial guess)
        std::vector<std::pair<unsigned int, unsigned int>> points_associations;
        const cv::Mat& pose_1 = this->_poses[this->_keyframes.back().getPoseIdx()];
        cv::Mat predicted_pose = pose_1.clone();
        if(!predictPose(predicted_pose, meas_to_integrate, points_associations, \
                        matcher, this->_keyframes, \
                        this->_keypoints, this->_measurements, \
                        this->_poses, this->_K, \
                        kernel_threshold_POSIT, \
                        inliers_threshold_POSIT, \
                        verbose)) {
            return false;
        }

        // Check that the new measurement can be used as a good keyframe
        if(!this->canBeSpawnedAsKeyframe(predicted_pose, points_associations, verbose))
            return false;

        if(verbose) {
            std::cout << std::endl << "\t";
        }

        // Add the new pose
        this->_poses.emplace_back(predicted_pose);

        // Use the new pose/measure as keyframe
        addKeyFrame(this->_next_measurement_idx-1, this->_poses.size()-1, \
                points_associations, this->_keyframes.size()-1, verbose);

        // If requested add new landmarks triangulating 
        // new matches between the last integrated keyframe and the last n 
        // (specified by local_map_size) keyframes
        if(triangulate_new_points) {
            
            triangulateNewPoints(this->_keyframes, \
                                this->_keypoints, \
                                this->_measurements, \
                                this->_poses, \
                                matcher, \
                                this->_K, \
                                local_map_size, \
                                new_landmark_threshold, \
                                parallax_threshold, \
                                verbose);

            if(verbose)
                std::cout << "TOTAL POINTS: " << this->_keypoints.size() << std::endl;

        }

        return true;

    }



    /*
    * Function that performs total Bundle Adjustment using g2o.
    * Here we assume that all the landmarks and keyframes are added in order
    * when they are restored in the original format.
    */
    void State::performTotalBA(const unsigned int& n_iters, const bool verbose) {

        // Initialization
        unsigned int vertex_id = 0;
        const unsigned int n_keyframes = this->_keyframes.size();
        const unsigned int n_keypoints = this->_keypoints.size();
        const float& fx = this->_K.at<float>(0,0);
        const float& fy = this->_K.at<float>(1,1);
        const float& cx = this->_K.at<float>(0,2);
        const float& cy = this->_K.at<float>(1,2);

        // Create optimizer
        g2o::SparseOptimizer optimizer;
        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
        linearSolver= g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
        g2o::OptimizationAlgorithmLevenberg* solver =
            new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
        optimizer.setAlgorithm(solver);        

        // --- Set keypoints vertices ---
        for(unsigned int i=0; i<n_keypoints; ++i) {

            // Get the reference to the current landmark
            const cv::Point3f& current_landmark = this->_keypoints[i].getPosition();

            // Create the new vertex
            g2o::VertexPointXYZ* vl = new g2o::VertexPointXYZ();
            vl->setId(vertex_id);
            vl->setEstimate(point_3d_to_vector_3d(current_landmark));
            vl->setMarginalized(true);
            optimizer.addVertex(vl);

            // Increment vertex_id
            ++vertex_id;

        }

        // --- Set Keyframes vertices ---
        for(unsigned int i=0; i<n_keyframes; ++i) {

            // Get the reference to the current keyframe, its pose and its measurement
            const Keyframe& current_keyframe = this->_keyframes[i];
            const cv::Mat& current_pose = this->_poses[current_keyframe.getPoseIdx()];
            const Measurement& current_meas = this->_measurements[current_keyframe.getMeasIdx()];

            // Create the new vertex
            g2o::VertexSE3Expmap* vk = new g2o::VertexSE3Expmap();
            vk->setEstimate(transformation_matrix_to_SE3Quat(current_pose));
            vk->setId(vertex_id);
            vk->setFixed(i==0);         // Block the first keyframe/pose
            optimizer.addVertex(vk);

            // Increment vertex_id
            ++vertex_id;

            // --- Set edges keyframe -> landmark ---
            const std::vector<std::pair<unsigned int, unsigned int>>& current_points_associations = \
                    current_keyframe.getPointsAssociations();
            const unsigned int n_associations = current_points_associations.size();
            
            for(unsigned int association_idx=0; association_idx<n_associations; ++association_idx) {

                // Take references for this association
                const std::pair<unsigned int, unsigned int>& current_association = \
                        current_points_associations[association_idx];
                const unsigned int& point_2d_idx = current_association.first;
                const unsigned int& landmark_idx = current_association.second;

                // Take the measured point of this association
                const cv::KeyPoint& z = current_meas.getPoints()[point_2d_idx];

                // Create the edge
                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                e->setVertex(0, \
                        dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(landmark_idx)) );
                e->setVertex(1, \
                        dynamic_cast<g2o::OptimizableGraph::Vertex*>(vk));
                e->setMeasurement(point_2d_to_vector_2d(z));
                e->information() = Eigen::Matrix2d::Identity();
                g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
                e->setRobustKernel(robust_kernel);
                robust_kernel->setDelta(sqrt(5.99));
                e->fx = fx; // Camera parameters
                e->fy = fy;
                e->cx = cx;
                e->cy = cy;
                optimizer.addEdge(e);

            }

        }

        // Optimize
        if(verbose) {
            std::cout << "BUNDLE ADJUSTMENT STARTED" << std::endl;
        }
        optimizer.initializeOptimization();
        optimizer.setVerbose(verbose);
        optimizer.optimize(n_iters);

        // --- Recover optimized data ---
        vertex_id = 0;

        // Recover landmarks
        for(unsigned int i=0; i<n_keypoints; ++i) {
            g2o::VertexPointXYZ* current_vertex = \
                    static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(vertex_id));
            this->_keypoints[i].setPosition( vector_3d_to_point_3d(current_vertex->estimate()) );
            ++vertex_id;
        }

        // Recover keyframes' poses
        for(unsigned int i=0; i<n_keyframes; ++i) {
            g2o::VertexSE3Expmap* current_vertex = \
                    static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(vertex_id));
            this->_poses[this->_keyframes[i].getPoseIdx()] = \
                    SE3Quat_to_transformation_matrix(current_vertex->estimate());
            ++vertex_id;
        }

        // Clear
        optimizer.clear();

    }



    /*
    * Simply the Local Bundle Adjustment.
    */
    void State::performLocalBA(const unsigned int& n_iters, const bool verbose) {

        // Initialization
        const unsigned int last_keyframe_idx = this->_keyframes.size()-1;
        const float& fx = this->_K.at<float>(0,0);
        const float& fy = this->_K.at<float>(1,1);
        const float& cx = this->_K.at<float>(0,2);
        const float& cy = this->_K.at<float>(1,2);
        unsigned int vertex_id = 0;

        // Get the local map of the last keyframe inserted in the state
        std::vector<unsigned int> observed_keypoints, near_local_keyframes, \
                                    far_local_keyframes;
        this->getLocalMap(last_keyframe_idx, observed_keypoints, \
                            near_local_keyframes, far_local_keyframes);
        const unsigned int n_keypoints = observed_keypoints.size();
        const unsigned int n_near_keyframes = near_local_keyframes.size();
        const unsigned int n_far_keyframes = far_local_keyframes.size();

        if(n_near_keyframes == 0)
            return;

        // Create optimizer
        g2o::SparseOptimizer optimizer;
        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
        linearSolver= g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
        g2o::OptimizationAlgorithmLevenberg* solver =
            new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
        optimizer.setAlgorithm(solver);

        // Create a map that, given the idx of a keypoint, returns the associated
        // vertex idx in the optimizer graph
        std::map<unsigned int, unsigned int> keypoint_idx2vertex_idx;

        // --- Set keypoints vertices ---
        for(unsigned int i=0; i<n_keypoints; ++i) {

            // Get the reference to the current landmark
            const cv::Point3f& p = this->_keypoints[observed_keypoints[i]].getPosition();

            // Create the new vertex
            g2o::VertexPointXYZ* vl = new g2o::VertexPointXYZ();
            vl->setId(vertex_id);
            keypoint_idx2vertex_idx[observed_keypoints[i]] = vertex_id;
            vl->setEstimate(point_3d_to_vector_3d(p));
            vl->setMarginalized(true);
            optimizer.addVertex(vl);

            // Increment vertex_id
            ++vertex_id;

        }

        // --- Set near Keyframes vertices ---
        for(unsigned int i=0; i<n_near_keyframes; ++i) {

            // Get the reference to the current keyframe, its pose and its measurement
            const Keyframe& current_keyframe = this->_keyframes[near_local_keyframes[i]];
            const cv::Mat& current_pose = this->_poses[current_keyframe.getPoseIdx()];
            const Measurement& current_meas = this->_measurements[current_keyframe.getMeasIdx()];

            // Create the new vertex
            g2o::VertexSE3Expmap* vk = new g2o::VertexSE3Expmap();
            vk->setEstimate(transformation_matrix_to_SE3Quat(current_pose));
            vk->setId(vertex_id);
            vk->setFixed(near_local_keyframes[i]==0);         // Block eventually the pose 0
            optimizer.addVertex(vk);

            // Increment vertex_id
            ++vertex_id;

            // --- Set edges keyframe -> landmark ---
            const std::vector<std::pair<unsigned int, unsigned int>>& current_points_associations = \
                    current_keyframe.getPointsAssociations();
            const unsigned int n_associations = current_points_associations.size();
            
            for(unsigned int association_idx=0; association_idx<n_associations; ++association_idx) {

                // Take references for this association
                const std::pair<unsigned int, unsigned int>& current_association = \
                        current_points_associations[association_idx];

                // If this association refers to a keypoint not in the local map, ignore it
                if(!keypoint_idx2vertex_idx.count(current_association.second) )
                    continue;    
                
                // Take references to points of the associations
                const unsigned int& point_2d_idx = current_association.first;
                const unsigned int& keypoint_vertex_idx = keypoint_idx2vertex_idx[current_association.second];

                // Take the measured point of this association
                const cv::KeyPoint& z = current_meas.getPoints()[point_2d_idx];

                // Create the edge
                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                e->setVertex(0, \
                        dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(keypoint_vertex_idx)) );
                e->setVertex(1, \
                        dynamic_cast<g2o::OptimizableGraph::Vertex*>(vk));
                e->setMeasurement(point_2d_to_vector_2d(z));
                e->information() = Eigen::Matrix2d::Identity();
                g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
                e->setRobustKernel(robust_kernel);
                robust_kernel->setDelta(sqrt(5.99));
                e->fx = fx; // Camera parameters
                e->fy = fy;
                e->cx = cx;
                e->cy = cy;
                optimizer.addEdge(e);

            }

        }

        // --- Set far Keyframes vertices ---
        for(unsigned int i=0; i<n_far_keyframes; ++i) {

            // Get the reference to the current keyframe, its pose and its measurement
            const Keyframe& current_keyframe = this->_keyframes[far_local_keyframes[i]];
            const cv::Mat& current_pose = this->_poses[current_keyframe.getPoseIdx()];
            const Measurement& current_meas = this->_measurements[current_keyframe.getMeasIdx()];

            // Create the new vertex
            g2o::VertexSE3Expmap* vk = new g2o::VertexSE3Expmap();
            vk->setEstimate(transformation_matrix_to_SE3Quat(current_pose));
            vk->setId(vertex_id);
            vk->setFixed(true);         // Block the far poses
            optimizer.addVertex(vk);

            // Increment vertex_id
            ++vertex_id;

            // --- Set edges keyframe -> landmark ---
            const std::vector<std::pair<unsigned int, unsigned int>>& current_points_associations = \
                    current_keyframe.getPointsAssociations();
            const unsigned int n_associations = current_points_associations.size();
            
            for(unsigned int association_idx=0; association_idx<n_associations; ++association_idx) {

                // Take references for this association
                const std::pair<unsigned int, unsigned int>& current_association = \
                        current_points_associations[association_idx];

                // If this association refers to a keypoint not in the local map, ignore it
                if(!keypoint_idx2vertex_idx.count(current_association.second) )
                    continue;
                
                // Take references to points of the associations
                const unsigned int& point_2d_idx = current_association.first;
                const unsigned int& keypoint_vertex_idx = keypoint_idx2vertex_idx[current_association.second];

                // Take the measured point of this association
                const cv::KeyPoint& z = current_meas.getPoints()[point_2d_idx];

                // Create the edge
                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                e->setVertex(0, \
                        dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(keypoint_vertex_idx)) );
                e->setVertex(1, \
                        dynamic_cast<g2o::OptimizableGraph::Vertex*>(vk));
                e->setMeasurement(point_2d_to_vector_2d(z));
                e->information() = Eigen::Matrix2d::Identity();
                g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber;
                e->setRobustKernel(robust_kernel);
                robust_kernel->setDelta(sqrt(5.99));
                e->fx = fx; // Camera parameters
                e->fy = fy;
                e->cx = cx;
                e->cy = cy;
                optimizer.addEdge(e);

            }

        }

        // Optimize
        if(verbose) {
            std::cout << "LOCAL BUNDLE ADJUSTMENT STARTED" << std::endl;
        }
        optimizer.initializeOptimization();
        optimizer.setVerbose(verbose);
        optimizer.optimize(n_iters);

        // --- Recover optimized data ---
        vertex_id = 0;

        // Recover landmarks
        for(unsigned int i=0; i<n_keypoints; ++i) {
            g2o::VertexPointXYZ* current_vertex = \
                    static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(vertex_id));
            this->_keypoints[i].setPosition( vector_3d_to_point_3d(current_vertex->estimate()) );
            ++vertex_id;
        }

        // Recover near keyframes' poses
        for(unsigned int i=0; i<n_near_keyframes; ++i) {
            g2o::VertexSE3Expmap* current_vertex = \
                    static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(vertex_id));
            this->_poses[this->_keyframes[near_local_keyframes[i]].getPoseIdx()] = \
                    SE3Quat_to_transformation_matrix(current_vertex->estimate());
            ++vertex_id;
        }

        // Recover far keyframes' poses
        for(unsigned int i=0; i<n_far_keyframes; ++i) {
            g2o::VertexSE3Expmap* current_vertex = \
                    static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(vertex_id));
            this->_poses[this->_keyframes[far_local_keyframes[i]].getPoseIdx()] = \
                    SE3Quat_to_transformation_matrix(current_vertex->estimate());
            ++vertex_id;
        }

        // Clear
        optimizer.clear();

    }




    /*
    * This function simply allows to add a new keyframe.
    * Setting observer_keyframe_idx to -1 means that we have no observer for
    * this pose.
    * It also sets the connections 3D point (keypoint) -> 2D point
    */
    void State::addKeyFrame(const unsigned int& meas_idx, const unsigned int& pose_idx, \
                            std::vector<std::pair<unsigned int, unsigned int>>& points_associations, \
                            const int& observer_keyframe_idx, const bool verbose) {
        
        // Initialization
        const unsigned int n_associations = points_associations.size();
        const unsigned int n_points_meas = this->_measurements[meas_idx].getPoints().size();

        // Add a new keyframe
        this->_keyframes.emplace_back(meas_idx, pose_idx, points_associations, n_points_meas);

        // Add the reference to the observer (if any)
        if(observer_keyframe_idx != -1) {
            this->_keyframes[observer_keyframe_idx].addKeyframeObserved(this->_keyframes.size()-1);
            this->_keyframes[this->_keyframes.size()-1].addObserverKeyframe(observer_keyframe_idx);
        }

        // Add the association 3D keypoints -> 2D points and update the keypoint
        // descriptor
        addAssociationKeypoints(this->_keypoints, points_associations, \
                                this->_keyframes.size()-1, this->_keyframes, \
                                this->_measurements);

        if(verbose) {
            std::cout << "NEW KEYFRAME ADDED (meas:" << meas_idx << \
                ", pose:" << pose_idx << ")" << std::endl; 
        }

    }



    /*
    * This function tells us if a new measurement (represented by its predicted
    * points associations) can be used as new keyframe. In particular these
    * conditions must be met in order to insert a new keyframe:
    *   1. The new measurement has more than 50 points associations (inliers) 
    *       but, in order to track fast rotations, if the rotation is "too high"
    *       we need at least 20 points associations
    *   2. It has less than 90% of common points with the last keyframe (the 90%
    *       of associations that the last keyframe has) (TURNED OFF NOW)
    *   3. If the new measurement's predicted pose is "too near" to the last keyframe
    *       and not has enough rotation (this allows to spawn a lot of keyframes during
    *       rotations, to not loose track)
    */
    bool State::canBeSpawnedAsKeyframe(const cv::Mat& pose, \
                                        const std::vector<std::pair<unsigned int, unsigned int>> \
                                            points_associations, \
                                        const bool verbose) {
        
        // Compute the angle between the two poses
        const cv::Mat& last_keyframe_pose = this->_poses[this->_keyframes.back().getPoseIdx()];
        const float poses_angle = compute_poses_angle(pose, last_keyframe_pose);

        // Check condition 1
        if(points_associations.size() < 50 && poses_angle < 0.049) {
            if(verbose)
                std::cout << ", not enough inliers...";
            return false;
        }
        if(points_associations.size() < 20) {
            if(verbose)
                std::cout << ", not enough inliers...";
            return false;
        }
        
        /* Check condition 2
        unsigned int n_common_keypoints = 0;
        std::map<unsigned int, unsigned int> keypoints_counter;
        const std::vector<std::pair<unsigned int, unsigned int>>& ref_associations = \
                this->_keyframes.back().getPointsAssociations();
        for(const auto& obs: ref_associations)
            keypoints_counter[obs.second]++;        
        for(const auto& obs: points_associations)
            keypoints_counter[obs.second]++;        
        for(const auto& el: keypoints_counter)
            if(el.second == 2)
                ++n_common_keypoints;
        if(n_common_keypoints >= 0.9*ref_associations.size())
            return false;
        */

        // Check condition 3
        if(compute_poses_distance(pose, last_keyframe_pose) < 0.03 && poses_angle < 0.03) {
            if(verbose)
                std::cout << ", not enough displacement...";
            return false;
        }

        return true;
        
    }



    /*
    * This function, given two keyframe ids, returns all the ids of the keypoints
    * seen from both of them.
    */
    void State::getCommonKeypoints(const unsigned int& k1_idx, \
                                    const unsigned int& k2_idx, \
                                    std::vector<unsigned int>& commond_keypoints_ids) {
        
        // Initialization
        const std::vector<std::pair<unsigned int, unsigned int>>& k1_ass = \
                    this->_keyframes[k1_idx].getPointsAssociations();
        const std::vector<std::pair<unsigned int, unsigned int>>& k2_ass = \
                    this->_keyframes[k2_idx].getPointsAssociations();

        // Create a "counter" for keypoints
        std::map<unsigned int, unsigned int> counter;
        for(auto& ass: k1_ass)
            counter[ass.second]++;
        for(auto& ass: k2_ass)
            counter[ass.second]++;

        // Save only keypoints seen from both
        commond_keypoints_ids.reserve(counter.size());
        for(auto& el: counter)
            if(el.second == 2)
                commond_keypoints_ids.emplace_back(el.first);
        commond_keypoints_ids.shrink_to_fit();
        
    }



    /*
    * This function computes the local map of a given keyframe. In particular
    * it computes:
    *   - the vector of keypoints seen from the given keyframe (observed_keypoints)
    *       TODO: insert also the keypoints seen from the near_local_keyframes?
    *   - the vector of keyframes that shares a certain number of keypoints
    *       seen with the given keyframe (near_local_keyframes)
    *   - the vector of keyframes that observe some keypoints sen from the given
    *       keyframe, but for which the number is not enough (they will be frozen
    *       in the local BA) (far_local_keyframes)
    */
    void State::getLocalMap(const unsigned int& keyframe_idx, \
                            std::vector<unsigned int>& observed_keypoints, \
                            std::vector<unsigned int>& near_local_keyframes, \
                            std::vector<unsigned int>& far_local_keyframes) {

        // Initialization
        const Keyframe& reference_keyframe = this->_keyframes[keyframe_idx];
        const std::vector<std::pair<unsigned int, unsigned int>>& observations_ref = \
                            reference_keyframe.getPointsAssociations();
        const unsigned int n_observations_ref = observations_ref.size();
        std::map<unsigned int, unsigned int> keyframes_observations_counter;
        const unsigned int common_points_threshold = 10;
        std::set<unsigned int> observer_keypoints_set;

        // Get all the keypoints seen from the reference keyframe
        for(unsigned int i=0; i<n_observations_ref; ++i) {
            observer_keypoints_set.insert(observations_ref[i].second);
        }
        const unsigned int n_observed_keypoints = observer_keypoints_set.size();

        // Count the number of common observation per local keyframe
        for(const auto& keypoint_idx: observer_keypoints_set) {
            const std::vector<std::pair<unsigned int, unsigned int>>& current_observers = \
                    this->_keypoints[keypoint_idx].getObservers();
            const unsigned int n_current_observers = current_observers.size();
            for(unsigned int obs_idx=0; obs_idx<n_current_observers; ++obs_idx) {
                keyframes_observations_counter[ current_observers[obs_idx].first ]++;
            }
        }

        // Build the near and far local keyframes vectors
        near_local_keyframes.reserve(keyframes_observations_counter.size());
        far_local_keyframes.reserve(keyframes_observations_counter.size());
        for(const auto& el: keyframes_observations_counter) {
            if(el.second >= common_points_threshold)
                near_local_keyframes.emplace_back(el.first);
            else
                far_local_keyframes.emplace_back(el.first);
        }
        near_local_keyframes.shrink_to_fit();
        far_local_keyframes.shrink_to_fit();

        // Add all the keypoints seen from the near keyframes
        for(const auto& keyframe_idx: near_local_keyframes) {
            const std::vector<std::pair<unsigned int, unsigned int>>& observations = \
                this->_keyframes[keyframe_idx].getPointsAssociations();
            for(const auto& obs: observations) {
                observer_keypoints_set.insert(obs.second);
            }
        }

        observed_keypoints = std::vector<unsigned int>(observer_keypoints_set.begin(), \
                                                        observer_keypoints_set.end());
        
    }



    /*
    * This function, given a measurement, predict its pose, by using an 
    * initial guess and all the seen landmarks, already triangulated, from 
    * such measurement. It uses my projective ICP implementation.
    * Inputs:
    *   guessedPose: initial guess for projective ICP (output)
    *   measurement: measurement for which predict the pose
    *   points_associations_inliers: a vector containing all the associations 2D points
    *       <-> 3D points for the measurement analyzed (it contains only "inliers")
    *   all the useful state infos and the matcher
    *   local_map_size: the number of keyframes to analyze in order to
    *       understand which landmarks are seen from the current measurement
    *       (starting from the last keyframe)
    * Output:
    *   true if the pose has been predicted, false otherwise
    */
    bool State::predictPose(cv::Mat& guessed_pose, \
                            const Measurement& meas_to_predict, \
                            std::vector<std::pair<unsigned int, unsigned int>>& \
                                        points_associations_inliers, \
                            Matcher& matcher, \
                            const std::vector<Keyframe>& keyframes, \
                            const std::vector<Keypoint>& keypoints, \
                            const std::vector<Measurement>& measurements, \
                            const std::vector<cv::Mat>& poses, \
                            const cv::Mat& K, \
                            const float& kernel_threshold_POSIT, \
                            const float& inliers_threshold_POSIT, \
                            const bool verbose) {

        // Get all the associations 2D point in the current image -> 3D keypoint
        // in the map
        std::vector<std::pair<unsigned int, unsigned int>> points_associations;
        projectAssociations(meas_to_predict, guessed_pose, K, keypoints, \
                            keyframes, measurements, points_associations);
        const unsigned int n_points_associations = points_associations.size();
        if(verbose) {
            std::cout << n_points_associations << " points already seen, ";
        }

        // If we do not have enough points associations, return error
        if(n_points_associations < 3) {
            if(verbose) {
                std::cout << "too few associations..." << std::endl;
            }
            return false;
        }

        // Predict the pose using projective ICP
        std::vector<bool> points_associations_filter(n_points_associations, false);
        const unsigned int n_inliers = perform_Posit(guessed_pose, \
                                                        meas_to_predict, \
                                                        points_associations_filter, \
                                                        points_associations, \
                                                        keypoints, \
                                                        K, \
                                                        kernel_threshold_POSIT, \
                                                        inliers_threshold_POSIT);

        if(verbose) {
            std::cout << n_inliers << " inliers";
        }

        // Mantain only the inliers
        points_associations_inliers.reserve(n_inliers);
        const unsigned int n_unfiltered_associations = points_associations.size();
        for(unsigned int i=0; i<n_unfiltered_associations; ++i) {
            if(points_associations_filter[i]) {
                points_associations_inliers.emplace_back(points_associations[i]);
            }
        }
        
        return true;

    }



    /* 
    * This function add new landmarks triangulating new matches between the last 
    * integrated keyframe and the last n (specified by triangulation_window) 
    * keyframes
    */
    void State::triangulateNewPoints(std::vector<Keyframe>& keyframes, \
                                    std::vector<Keypoint>& keypoints, \
                                    const std::vector<Measurement>& measurements, \
                                    const std::vector<cv::Mat>& poses, \
                                    Matcher& matcher, \
                                    const cv::Mat& K, \
                                    const unsigned int& triangulation_window, \
                                    const float& new_landmark_threshold, \
                                    const float& parallax_threshold, \
                                    const bool verbose) {
        
        // Initialization
        const unsigned int& n_keyframes = keyframes.size()-1;
        const float matching_distance_threshold = 8;   // 8 for ANMS, 12 for ORB
        unsigned int n_triangulated = 0;

        // Adjust the window size according to the number of keyframes
        // present
        unsigned int window = triangulation_window;
        if(triangulation_window > n_keyframes) {
            window = n_keyframes;
        }

        // Take the reference to the last integrated keyframe
        const unsigned int keyframe2_idx = keyframes.size()-1;
        Keyframe& last_keyframe = keyframes[keyframe2_idx];
        const Measurement& meas2 = measurements[last_keyframe.getMeasIdx()];
        const cv::Mat& pose2 = poses[last_keyframe.getPoseIdx()];

        if(verbose) {
            std::cout << "\tTRIANGULATING NEW POINTS: ";
        }

        // For each measurement in the window, triangulate new points
        for(unsigned int windows_idx=1; windows_idx<window+1; ++windows_idx) {
            
            // Take the reference to the keyframe with which triangulate
            const unsigned int keyframe1_idx = n_keyframes-windows_idx;
            Keyframe& current_keyframe = keyframes[keyframe1_idx];
            const Measurement& meas1 = measurements[current_keyframe.getMeasIdx()];
            const cv::Mat& pose1 = poses[current_keyframe.getPoseIdx()];

            // Matches the two keyframes
            std::vector<cv::DMatch> matches;
            matcher.match_measurements(meas1, meas2, matches, matching_distance_threshold);
            const unsigned int n_matches = matches.size();

            // If we have no match just ignore this measurement
            if(n_matches == 0) {
                continue;
            }

            // Build a matches filter, to take into account only those
            // matched 2D points for which we don't have already a 3D point associated
            std::vector<unsigned int> matches_filter;
            matches_filter.reserve(n_matches);
            for(unsigned int match_idx=0; match_idx<n_matches; ++match_idx) {
                
                // Take references to the two points in the current match
                const unsigned int& p1 = matches[match_idx].queryIdx;
                const unsigned int& p2 = matches[match_idx].trainIdx;
                const int p1_3dpoint_idx = current_keyframe.point2Landmark(p1);
                const int p2_3dpoint_idx = last_keyframe.point2Landmark(p2);

                // If we do not have a 3D point associated to this match
                // use it to triangulate a new point
                if(p1_3dpoint_idx == -1 && p2_3dpoint_idx == -1) {
                        matches_filter.emplace_back(match_idx);
                }

            }
            matches_filter.shrink_to_fit();

            // Triangulate new points
            std::vector<cv::Point3f> triangulated_points;
            triangulate_points(meas1.getPoints(), meas2.getPoints(), \
                                matches, matches_filter, \
                                pose1, pose2, K, \
                                triangulated_points);
                        
            // Add new triangulated points to the state
            // (in keypoints vector and in corresponding keyframes)
            std::vector<std::pair<unsigned int, unsigned int>> new_points_associations1;
            std::vector<std::pair<unsigned int, unsigned int>> new_points_associations2;
            associateNewKeypoints(triangulated_points, matches, matches_filter, \
                                keypoints, new_points_associations1, \
                                new_points_associations2, false);
            addAssociationKeypoints(keypoints, new_points_associations1, \
                                        keyframe1_idx, keyframes, measurements);
            addAssociationKeypoints(keypoints, new_points_associations2, \
                                        keyframe2_idx, keyframes, measurements);
            current_keyframe.addPointsAssociations(new_points_associations1);
            last_keyframe.addPointsAssociations(new_points_associations2);
            n_triangulated += new_points_associations1.size();

        }

        if(verbose)
            std::cout << n_triangulated << " new points, ";

    }



    /*
    * This function, given a set of keypoints, a set of associations 2D point -> 3D point
    * and the measure at which these associations are correlated, add the association
    * 3D point -> 2D point inside the keypoints vector.
    */
    void State::addAssociationKeypoints(std::vector<Keypoint>& keypoints, \
                                        const std::vector<std::pair<unsigned int, unsigned int>>& \
                                                    points_associations, \
                                        const unsigned int keyframe_idx, \
                                        const std::vector<Keyframe>& keyframes, \
                                        const std::vector<Measurement>& measurements) {
        
        // Initialization
        const unsigned int n_associations = points_associations.size();

        // Add each association
        for(unsigned int i=0; i<n_associations; ++i) {
            const std::pair<unsigned int, unsigned int>& current_association = \
                    points_associations[i];
            Keypoint& kp = keypoints[current_association.second];
            kp.addObserver(keyframe_idx, current_association.first);
            kp.updateDescriptor(keyframes, measurements);
        }

    }



    /*
    * This function takes matches between two measurements (filtered) and a set
    * of triangulated points between them and create a keypoint for each new
    * valid triangulated points, creating, in the meanwhile, the association
    * vector 2D point <-> 3D point between each of the two measurements.
    * Inputs:
    *   predicted_landmarks: the set of triangulated point, one for each filtered
    *       matches.
    *   matches
    *   matches_filter
    *   keypoints: the vector where to add only the valid predicted landmarks (output)
    *       (it can be empty or already filled and we assume it already "reserved").
    *   meas1_points_associations/meas2_points_associations: the associations vectors
    *       2D point <-> 3D point (outputs).
    */
    void State::associateNewKeypoints(const std::vector<cv::Point3f>& predicted_landmarks, \
                                        const std::vector<cv::DMatch>& matches, \
                                        const std::vector<unsigned int>& matches_filter, \
                                        std::vector<Keypoint>& keypoints, \
                                        std::vector<std::pair<unsigned int, \
                                                unsigned int>>& meas1_points_associations, \
                                        std::vector<std::pair<unsigned int, \
                                                unsigned int>>& meas2_points_associations, \
                                        const bool verbose) {
        
        // Initialization
        const unsigned int n_associations = matches_filter.size();
        meas1_points_associations.reserve(n_associations);
        meas2_points_associations.reserve(n_associations);
        unsigned int n_invalid_keypoints = 0;
        unsigned int n_added_keypoints = 0;
        unsigned int current_kp_idx, current_meas1_point_idx, current_meas2_point_idx;

        for(unsigned int i=0; i<n_associations; ++i) {

            // Get the current 3D point
            const cv::Point3f& current_point = predicted_landmarks[i];
            
            // If the landmark is not triangulated in a good way
            // ignore this association
            if(current_point.x == 0 && \
                current_point.y == 0 && \
                current_point.z == 0) {
                if(verbose) n_invalid_keypoints++;
                continue;                   
            }

            // Get the 2D points ids that sees the current 3D point
            current_meas1_point_idx = matches[matches_filter[i]].queryIdx;
            current_meas2_point_idx = matches[matches_filter[i]].trainIdx;

            // Create a new keypoint with the current valid 3D point
            keypoints.emplace_back(Keypoint(current_point));
            
            // Save the association 2D point -> 3D point
            current_kp_idx = keypoints.size()-1;
            meas1_points_associations.emplace_back(current_meas1_point_idx, current_kp_idx);
            meas2_points_associations.emplace_back(current_meas2_point_idx, current_kp_idx);
            
            if(verbose) n_added_keypoints++;
        }
        meas1_points_associations.shrink_to_fit();
        meas2_points_associations.shrink_to_fit();

        if(verbose) {
            std::cout << "KEYPOINTS ADDED: " << n_added_keypoints << " (" \
                    << n_invalid_keypoints << " invalid)" << std::endl;
        }

    }


    /*
    * Function that tells us if a vector containing associations 2D point <-> 3D point
    * contains a given 3D point idx (landmark_idx)
    */
    bool State::containsLandmark(const std::vector<std::pair<unsigned int, \
                                                unsigned int>>& points_associations, \
                                            const unsigned int& landmark_idx) {
        const unsigned int& n_associations = points_associations.size();
        for(unsigned int i=0; i<n_associations; ++i) {
            if(points_associations[i].second == landmark_idx) {
                return true;
            }
        }
        return false;
    }



    /*
    * This function, given a measurement and a set of 3D point, computes the 
    * associations 2D point <-> 3D point by searching for those 3D points
    * that can be candidates for such association.
    * Inputs:
    *   meas: measurement to analyze
    *   keypoints: 3D points
    *   keyframes: all the keyframes in the state
    *   measurements: all the measurements in the state
    *   matcher: useful to compute distance from descriptors
    *   points_associations: output associations vector
    */
    void State::projectAssociations(const Measurement& meas, \
                                    const cv::Mat& T, const cv::Mat& K, \
                                    const std::vector<Keypoint>& keypoints, \
                                    const std::vector<Keyframe>& keyframes, \
                                    const std::vector<Measurement>& measurements, \
                                    std::vector<std::pair<unsigned int, unsigned int>>& \
                                            points_associations) {
        
        // Initialization
        const std::vector<cv::KeyPoint>& points_2d = meas.getPoints();
        const unsigned int n_points_3d = keypoints.size();
        const unsigned int n_points_2d = points_2d.size();
        const unsigned int image_width = 2*K.at<float>(0,2);
        const unsigned int image_height = 2*K.at<float>(1,2);
        float kp_cam_x, kp_cam_y, kp_cam_z, kp_img_x, kp_img_y, iz;
        const unsigned int distance_threshold = 50;
        int current_distance, best_distance;
        unsigned int best_row_idx;

        // Some reference to save time
        const float& R11 = T.at<float>(0,0);
        const float& R12 = T.at<float>(0,1);
        const float& R13 = T.at<float>(0,2);
        const float& R21 = T.at<float>(1,0);
        const float& R22 = T.at<float>(1,1);
        const float& R23 = T.at<float>(1,2);
        const float& R31 = T.at<float>(2,0);
        const float& R32 = T.at<float>(2,1);
        const float& R33 = T.at<float>(2,2);
        const float& tx = T.at<float>(0,3);
        const float& ty = T.at<float>(1,3);
        const float& tz = T.at<float>(2,3);
        const float& K11 = K.at<float>(0,0);
        const float& K12 = K.at<float>(0,1);
        const float& K13 = K.at<float>(0,2);
        const float& K21 = K.at<float>(1,0);
        const float& K22 = K.at<float>(1,1);
        const float& K23 = K.at<float>(1,2);
        const float& K31 = K.at<float>(2,0);
        const float& K32 = K.at<float>(2,1);
        const float& K33 = K.at<float>(2,2);

        // Build the associations matrix where a row contains all the 
        // distances from a given 3D point to all 2D points in the measurement
        std::vector<std::vector<int>> associations_matrix;
        std::map<unsigned int, unsigned int> row2keypointidx;
        unsigned int current_row = 0;
        associations_matrix.reserve(n_points_3d);
        for(unsigned int i=0; i<n_points_3d; ++i) {
            
            // Get the current keypoint
            const Keypoint& kp = keypoints[i];
            const float& kp_w_x = kp.getPosition().x;
            const float& kp_w_y = kp.getPosition().y;
            const float& kp_w_z = kp.getPosition().z;

            // Bring it in camera frame and check if it is in front of cam
            kp_cam_x = R11*kp_w_x + R12*kp_w_y + R13*kp_w_z + tx;
            kp_cam_y = R21*kp_w_x + R22*kp_w_y + R23*kp_w_z + ty;
            kp_cam_z = R31*kp_w_x + R32*kp_w_y + R33*kp_w_z + tz;
            if(kp_cam_z <= 0)
                continue;

            // Project the point on image plane
            iz = K31*kp_cam_x + K32*kp_cam_y + K33*kp_cam_z;
            kp_img_x = (K11*kp_cam_x + K12*kp_cam_y + K13*kp_cam_z)/iz;
            kp_img_y = (K21*kp_cam_x + K22*kp_cam_y + K23*kp_cam_z)/iz;

            // Check that the point is inside the image plane
            if(kp_img_x < 0 || kp_img_x > image_width ||
                kp_img_y < 0 || kp_img_y > image_height)
                continue;
            
            // Get the representaitve descriptor of the current 3D point
            const cv::Mat& point_3d_descriptor = kp.getDescriptor();
            
            // The current keypoint is valid and can be added to the
            // associations matrix by building a vector of scores,
            // one element for each 2D point in the image
            row2keypointidx[current_row++] = i;
            std::vector<int> current_associations;
            current_associations.reserve(n_points_2d);
            for(unsigned int p_idx=0; p_idx<n_points_2d; ++p_idx) {
                current_associations.emplace_back(\
                        Matcher::compute_descriptors_distance(\
                                meas.getDescriptor(p_idx), point_3d_descriptor) \
                );
            }
            current_associations.shrink_to_fit();

            // Save it in the matrix
            associations_matrix.emplace_back(current_associations);

        }
        associations_matrix.shrink_to_fit();
        const unsigned int n_rows = associations_matrix.size();

        // Choose, for each 2D point, the best association, avoiding
        // duplicates
        std::vector<bool> associated_points(n_points_3d, false);
        points_associations.reserve(n_points_2d);
        for(unsigned int p_2d_idx=0; p_2d_idx<n_points_2d; ++p_2d_idx) {
            
            // Set the current best distance to max and best_row_idx to invalid
            best_distance = INT_MAX;
            best_row_idx = -1;
            
            // Search for the best 3D point
            for(unsigned int p_3d_row_idx=0; p_3d_row_idx<n_rows; ++p_3d_row_idx) {
                
                // If the current 3D point is already associated, ignore it
                if(associated_points[p_3d_row_idx])
                    continue;
                
                // Get the distance from the current 3D point and save
                // if it is the best one
                current_distance = associations_matrix[p_3d_row_idx][p_2d_idx];
                if(current_distance < best_distance) {
                    best_distance = current_distance;
                    best_row_idx = p_3d_row_idx;
                }
            }

            // If we found the best association and it is good enough, save it
            if(best_row_idx != -1 && best_distance < distance_threshold) {
                points_associations.emplace_back(p_2d_idx, row2keypointidx[best_row_idx]);
            }

        }
        points_associations.shrink_to_fit();

    }


} // namespace SLucAM