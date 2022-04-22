//
// SLucAM_state.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_state.h>
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



// -----------------------------------------------------------------------------
// Implementation of State class methods
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This constructor allows us to reserve some expected space for the vector
    * of poses and the vector of landmarks, just for optimization. It also need
    * the camera matrix K. It also create the matcher (default).
    */
    State::State(cv::Mat& K, std::vector<Measurement>& measurements, \
                const unsigned int expected_poses, \
                const unsigned int expected_landmarks) {
        
        this->_K = K;
        this->_measurements = measurements;
        this->_poses.reserve(expected_poses);
        this->_landmarks.reserve(expected_landmarks);
        this->_keyframes.reserve(expected_poses);
        this->_next_measurement_idx = 0;

    }



    /*
    * This function performs the initialization of the state. It try to 
    * perform initialization between the measurement in position 0 of the 
    * measurements vector and the first measurement that have enough translation
    * between them. If no such measurement is found then is returned false.  
    */
    bool State::initializeState(Matcher& matcher, \
                                const unsigned int& ransac_iters, \
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
                        triangulated_points, ransac_iters, \
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
        associateNewLandmarks(triangulated_points, matches, matches_filter, \
                                this->_landmarks, meas1_points_associations, \
                                 meas2_points_associations, false, 0, verbose);

        // Create the first pose (we assume it at the origin) and use it as new keyframe
        this->_poses.emplace_back(cv::Mat::eye(4,4,CV_32F));
        addKeyFrame(0, 0, meas1_points_associations, -1, verbose);

        // Save the predicted pose and use it as new keyframe
        this->_poses.emplace_back(predicted_pose);
        addKeyFrame(this->_next_measurement_idx-1, this->_poses.size()-1, \
                    meas2_points_associations, this->_keyframes.size()-1, \
                    verbose);
        
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
                                        const unsigned int& posit_n_iters, \
                                        const float& posit_kernel_threshold, \
                                        const float& posit_threshold_to_ignore, \
                                        const float& posit_damping_factor, \
                                        const unsigned int& local_map_size, \
                                        const float& parallax_threshold, \
                                        const float& new_landmark_threshold, \
                                        const bool verbose) {

        // If we have no more measurement to integrate, return error
        if(this->reaminingMeasurements() == 0) return false;

        if(verbose) {
            std::cout << "-> INTEGRATING NEW MEASUREMENT: ";
        }

        // Take the measurement to integrate
        const SLucAM::Measurement& meas_to_integrate = getNextMeasurement();

        // Predict the new pose (use previous keyframe's pose as initial guess)
        std::vector<std::pair<unsigned int, unsigned int>> points_associations;
        const cv::Mat& pose_1 = this->_poses[this->_keyframes.back().getPoseIdx()];
        cv::Mat predicted_pose = pose_1.clone();
        if(!predictPose(predicted_pose, meas_to_integrate, points_associations, \
                            matcher, this->_keyframes, \
                            this->_landmarks, this->_measurements, \
                            this->_poses, this->_K, \
                            local_map_size, verbose)) {
            return false;
        }
        const unsigned int n_inliers_posit = points_associations.size();

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
                                this->_landmarks, \
                                this->_measurements, \
                                this->_poses, \
                                matcher, \
                                this->_K, \
                                local_map_size, \
                                new_landmark_threshold, \
                                parallax_threshold, \
                                verbose);

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
        const unsigned int n_landmarks = this->_landmarks.size();

        // Create optimizer
        g2o::SparseOptimizer optimizer;
        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
        linearSolver= g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
        g2o::OptimizationAlgorithmLevenberg* solver =
            new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
        optimizer.setAlgorithm(solver);

        // Set camera parameters
        double focal_length = this->_K.at<float>(0,0);
        Eigen::Vector2d principal_point(this->_K.at<float>(0,2), this->_K.at<float>(1,2));
        g2o::CameraParameters* cam_params = \
                new g2o::CameraParameters(focal_length, principal_point, 0.);
        cam_params->setId(0);
        optimizer.addParameter(cam_params);

        // --- Set landmarks vertices ---
        for(unsigned int i=0; i<n_landmarks; ++i) {

            // Get the reference to the current landmark
            const cv::Point3f& current_landmark = this->_landmarks[i];

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
                g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();
                e->setVertex(0, \
                        dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(landmark_idx)) );
                e->setVertex(1, \
                        dynamic_cast<g2o::OptimizableGraph::Vertex*>(vk));
                e->setMeasurement(point_2d_to_vector_2d(z));
                e->information() = Eigen::Matrix2d::Identity();
                e->setRobustKernel(new g2o::RobustKernelHuber);
                e->setParameterId(0, 0);
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
        for(unsigned int i=0; i<n_landmarks; ++i) {
            g2o::VertexPointXYZ* current_vertex = \
                    static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(vertex_id));
            this->_landmarks[i] = vector_3d_to_point_3d(current_vertex->estimate());
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
    * This function simply allows to add a new keyframe.
    * Setting observer_keyframe_idx to -1 means that we have no observer for
    * this pose.
    */
    void State::addKeyFrame(const unsigned int& meas_idx, const unsigned int& pose_idx, \
                            std::vector<std::pair<unsigned int, unsigned int>>& points_associations, \
                            const int& observer_keyframe_idx, const bool verbose) {
        
        // Initialization
        const unsigned int n_points_meas = this->_measurements[meas_idx].getPoints().size();

        // Add a new keyframe
        this->_keyframes.emplace_back(meas_idx, pose_idx, points_associations, n_points_meas);

        // Add the reference to the observer (if any)
        if(observer_keyframe_idx != -1) {
            this->_keyframes[observer_keyframe_idx].addKeyframeAssociation(this->_keyframes.size()-1);
        }

        if(verbose) {
            std::cout << "NEW KEYFRAME ADDED (meas:" << meas_idx << \
                ", pose:" << pose_idx << ")" << std::endl; 
        }

    }



    /*
    * This function, given a measurement, predict its pose, by using an 
    * initial guess and all the seen landmarks, already triangulated, from 
    * such measurement. It uses projective ICP.
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
                            const std::vector<cv::Point3f>& landmarks, \
                            const std::vector<Measurement>& measurements, \
                            const std::vector<cv::Mat>& poses, \
                            const cv::Mat& K, \
                            const unsigned int& local_map_size, \
                            const bool& verbose) {
        
        // Initialization
        const unsigned int n_keyframes = keyframes.size();
        const float fx = K.at<float>(0,0);
        const float fy = K.at<float>(1,1);
        const float cx = K.at<float>(0,2);
        const float cy = K.at<float>(1,2);
        std::vector<std::pair<unsigned int, unsigned int>> points_associations;
        std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> graph_edges;

        // Adjust the local map size according to the number of keyframes
        unsigned int window = local_map_size;
        if(local_map_size > n_keyframes) {
            window = n_keyframes;
        }

        // Create optimizer
        g2o::SparseOptimizer optimizer;
        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
        linearSolver= g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
        g2o::OptimizationAlgorithmLevenberg* solver =
            new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
        optimizer.setAlgorithm(solver);

        // Set the pose to optimize
        g2o::VertexSE3Expmap* vPose = new g2o::VertexSE3Expmap();
        vPose->setEstimate(transformation_matrix_to_SE3Quat(guessed_pose));
        vPose->setId(0);
        vPose->setFixed(false);
        optimizer.addVertex(vPose);

        // For each measurement in the window
        for(unsigned int window_idx=1; window_idx<window+1; ++window_idx) {

            // Take the current keyframe's measurement
            const Keyframe& current_keyframe = keyframes[n_keyframes-window_idx];
            const Measurement& current_meas = measurements[current_keyframe.getMeasIdx()];
            
            // Match the current measurement with the measure to predict
            std::vector<cv::DMatch> matches;
            matcher.match_measurements(current_meas, meas_to_predict, matches);
            const unsigned int n_matches = matches.size();

            // For each match search for that for which we have already a 3D point
            // predicted     
            int current_3d_point_idx;
            points_associations.reserve(points_associations.size()+n_matches);
            graph_edges.reserve(graph_edges.size()+n_matches);
            for(unsigned int i=0; i<n_matches; ++i) {

                // Get the idx of the landmark for this association (if not we have -1)
                current_3d_point_idx = \
                    current_keyframe.point2Landmark(matches[i].queryIdx);
                
                // If we have a predicted point for this association and we have not
                // already used it
                if( (current_3d_point_idx != -1) && \
                    (!containsLandmark(points_associations, current_3d_point_idx))) {
                    
                    // Add it as point association
                    points_associations.emplace_back(matches[i].trainIdx, \
                                                current_3d_point_idx);

                    // Use this association as new edge for the optimization graph
                    const cv::KeyPoint point2D = meas_to_predict.getPoints()[matches[i].trainIdx];
                    const cv::Point3f point3D = landmarks[current_3d_point_idx];
                    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(point_2d_to_vector_2d(point2D));
                    e->setInformation(Eigen::Matrix2d::Identity());
                    e->setRobustKernel(new g2o::RobustKernelHuber);
                    e->fx = fx;
                    e->fy = fy;
                    e->cx = cx;
                    e->cy = cy;
                    e->Xw[0] = point3D.x;
                    e->Xw[1] = point3D.y;
                    e->Xw[2] = point3D.z;
                    optimizer.addEdge(e);

                    // Add the reference to a vector to keep acces to it
                    graph_edges.emplace_back(e);
                }

            }
        }
        points_associations.shrink_to_fit();
        graph_edges.shrink_to_fit();
        const unsigned int n_points_associations = points_associations.size();

        if(verbose) {
            std::cout << n_points_associations << " points already seen, ";
        }

        // If we do not have enough points associations, return error
        if(n_points_associations < 2) {
            if(verbose) {
                std::cout << "too few associations..." << std::endl;
            }
            return false;
        }

        // Optimize
        // TODO: set n_iters manually
        optimizer.initializeOptimization();
        optimizer.setVerbose(false);
        optimizer.optimize(50);

        // Mantain only the inliers
        const unsigned int n_associations = points_associations.size();
        points_associations_inliers.reserve(n_associations);
        for(unsigned int i=0; i<n_associations; ++i) {
            g2o::EdgeSE3ProjectXYZOnlyPose* current_edge = graph_edges[i];
            if(current_edge->chi2() <= 5.991) {  // TODO: set it manually
                points_associations_inliers.emplace_back(points_associations[i]);
            }
        }
        points_associations_inliers.shrink_to_fit();
        const unsigned int n_inliers = points_associations_inliers.size();

        if(verbose) {
            std::cout << n_inliers << " inliers, ";
        }

        // Check if we have at least 3 inliers (the minimal set of points
        // to have a "good" constrained problem)
        if(n_inliers < 3) {
            if(verbose) {
                std::cout << "too few inliers..." << std::endl;
            }
            return false;
        }

        // Recover optimized pose
        guessed_pose = SE3Quat_to_transformation_matrix(\
                            static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0))\
                                ->estimate() \
                        );
        
        return true;

    }



    /* 
    * This function add new landmarks triangulating new matches between the last 
    * integrated keyframe and the last n (specified by triangulation_window) 
    * keyframes
    */
    void State::triangulateNewPoints(std::vector<Keyframe>& keyframes, \
                                    std::vector<cv::Point3f>& landmarks, \
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
        const float matching_distance_threshold = 10;

        // Adjust the window size according to the number of keyframes
        // present
        unsigned int window = triangulation_window;
        if(triangulation_window > n_keyframes) {
            window = n_keyframes;
        }

        // Take the reference to the last integrated keyframe
        Keyframe& last_keyframe = keyframes.back();
        const Measurement& meas2 = measurements[last_keyframe.getMeasIdx()];
        const cv::Mat& pose2 = poses[last_keyframe.getPoseIdx()];

        if(verbose) {
            std::cout << "\tTRIANGULATING NEW POINTS" << std::endl;
        }

        // For each measurement in the window, triangulate new points
        for(unsigned int windows_idx=1; windows_idx<window+1; ++windows_idx) {

            if(verbose) {
                std::cout << "\t\tWINDOW " << windows_idx << ": ";
            }
            
            // Take the reference to the keyframe with which triangulate
            Keyframe& current_keyframe = keyframes[n_keyframes-windows_idx];
            const Measurement& meas1 = measurements[current_keyframe.getMeasIdx()];
            const cv::Mat& pose1 = poses[current_keyframe.getPoseIdx()];

            // Matches the two keyframes
            std::vector<cv::DMatch> matches;
            matcher.match_measurements(meas1, meas2, matches, matching_distance_threshold);
            const unsigned int n_matches = matches.size();
            
            if(verbose) {
                std::cout << n_matches << " matches, ";
            }

            // If we have no match just ignore this measurement
            if(n_matches == 0) {
                if(verbose) {
                    std::cout << " nothing to triangulate." << std::endl;
                }
                continue;
            }

            // Build a matches filter, to take into account only those
            // matched 2D points for which we don't have already a 3D point associated
            // In the meanwhile create a vector of common landmarks ids for parallax
            // computation
            std::vector<unsigned int> matches_filter;
            std::vector<unsigned int> common_landmarks_ids;
            matches_filter.reserve(n_matches);
            common_landmarks_ids.reserve(n_matches);
            unsigned int n_added_first = 0; 
            unsigned int n_added_second = 0;
            for(unsigned int match_idx=0; match_idx<n_matches; ++match_idx) {
                
                // Take references to the two points in the current match
                const unsigned int& p1 = matches[match_idx].queryIdx;
                const unsigned int& p2 = matches[match_idx].trainIdx;
                const int p1_3dpoint_idx = current_keyframe.point2Landmark(p1);
                const int p2_3dpoint_idx = last_keyframe.point2Landmark(p2);

                // Check if we have already a 3D point associated to p1
                if(p1_3dpoint_idx == -1) {

                    // If we do not have a 3D point associated neither to p2
                    // use this match to triangulate
                    if(p2_3dpoint_idx == -1) {
                        matches_filter.emplace_back(match_idx);
                    } else {
                        // Otherwise, add that association to the first keyframe
                        current_keyframe.addPointAssociation(p1, p2_3dpoint_idx);
                        n_added_first++;
                    }

                } else {

                    // If we do not have a 3D point associated to p2, add that association
                    // for p1
                    if(p2_3dpoint_idx == -1) {
                        last_keyframe.addPointAssociation(p2, p1_3dpoint_idx);
                        n_added_second++;
                    } else {
                        // In this case both points have already a prediction,
                        // if it is the same, consider that prediction as 
                        // common landmark
                        if(p1_3dpoint_idx == p2_3dpoint_idx) {
                            common_landmarks_ids.emplace_back(p1_3dpoint_idx);
                        }
                    }

                }

            }
            matches_filter.shrink_to_fit();
            common_landmarks_ids.shrink_to_fit();

            // If we have no common points we cannot compute the parallax
            // so we ignore this measurement
            const unsigned int n_common_landmarks = common_landmarks_ids.size();
            if(n_common_landmarks== 0) {
                if(verbose) {
                    std::cout << n_common_landmarks << " already seen, unable" << \
                                " to compute the parallax." << std::endl;
                }
                continue;
            }

            // Compute the parallax
            float parallax = computeParallax(pose1, pose2, \
                                landmarks, common_landmarks_ids);
        
            if(verbose) {
                std::cout << common_landmarks_ids.size() << " already seen " \
                            << "(" << n_added_first << " added to meas1" \
                            << ", " << n_added_second << " added to meas2), " \
                            << matches_filter.size() << " new points, " \
                            << "parallax "  << parallax << std::endl;
            }  

            // If we have enough parallax
            if(parallax > parallax_threshold) {

                // Triangulate new points
                std::vector<cv::Point3f> triangulated_points;
                triangulate_points(meas1.getPoints(), meas2.getPoints(), \
                                    matches, matches_filter, \
                                    pose1, pose2, K, \
                                    triangulated_points);
                            
                // Add new triangulated points to the state
                // (in landmarks vector and in corresponding keyframes)
                std::vector<std::pair<unsigned int, unsigned int>> new_points_associations1;
                std::vector<std::pair<unsigned int, unsigned int>> new_points_associations2;
                if(verbose) {
                    std::cout << "\t\t\t";
                }
                associateNewLandmarks(triangulated_points, matches, matches_filter, \
                                    landmarks, new_points_associations1, \
                                    new_points_associations2, true, new_landmark_threshold, \
                                    verbose);
                current_keyframe.addPointsAssociations(new_points_associations1);
                last_keyframe.addPointsAssociations(new_points_associations2);
                
            }

        }

    }



    /*
    * This function takes matches between two measurements (filtered) and a set
    * of triangulated points between them and adds to the landmarks vector only
    * valid triangulated points, creating, in the meanwhile, the association
    * vector 2D point <-> 3D point between each of the two measurements.
    * If requested (filter_near_points=true) it also filters out all those points
    * predicted that are too near to some other point in the landmarks vector (so
    * already triangulated). In such case also update correctly the points
    * associations.
    * Inputs:
    *   predicted_landmarks: the set of triangulated point, one for each filtered
    *       matches.
    *   matches
    *   matches_filter
    *   landmarks: the vector where to add only the valid predicted landmarks (output)
    *       (it can be empty or already filled and we assume it already "reserved").
    *   meas1_points_associations/meas2_points_associations: the associations vectors
    *       2D point <-> 3D point (outputs).
    *   filter_near_points: if true filters out too near 3D points
    *   new_landmark_threshold: filter for too near 3D points
    */
    void State::associateNewLandmarks(const std::vector<cv::Point3f>& predicted_landmarks, \
                                        const std::vector<cv::DMatch>& matches, \
                                        const std::vector<unsigned int>& matches_filter, \
                                        std::vector<cv::Point3f>& landmarks, \
                                        std::vector<std::pair<unsigned int, \
                                                unsigned int>>& meas1_points_associations, \
                                        std::vector<std::pair<unsigned int, \
                                                unsigned int>>& meas2_points_associations, \
                                        const bool& filter_near_points, \
                                        const float& new_landmark_threshold, \
                                        const bool verbose) {
        
        // Initialization
        const unsigned int n_associations = matches_filter.size();
        meas1_points_associations.reserve(n_associations);
        meas2_points_associations.reserve(n_associations);
        unsigned int n_invalid_points = 0;
        unsigned int n_filtered_points = 0;
        unsigned int n_added_points = 0;
        unsigned int current_landmark_idx;

        for(unsigned int i=0; i<n_associations; ++i) {

            // Get the current 3D point
            const cv::Point3f& current_point = predicted_landmarks[i];
            
            // If the landmark is not triangulated in a good way
            // ignore this association
            if(current_point.x == 0 && \
                current_point.y == 0 && \
                current_point.z == 0) {
                if(verbose) n_invalid_points++;
                continue;                   
            }

            // If the filter is requested
            if(filter_near_points) {

                // Compute the nearest point
                const std::pair<int, float> nearest_distance = \
                    nearest_3d_point(current_point, landmarks);
                
                // If the nearest point is under a threshold
                if(nearest_distance.second < new_landmark_threshold) {

                    // Associate that landmark to the points from which
                    // the alias has been triangulated
                    meas1_points_associations.emplace_back(\
                        matches[matches_filter[i]].queryIdx, nearest_distance.first);
                    meas2_points_associations.emplace_back(\
                        matches[matches_filter[i]].trainIdx, nearest_distance.first);
                    
                    if(verbose) n_filtered_points++;

                    continue;

                }

            }

            landmarks.emplace_back(current_point);
            current_landmark_idx = landmarks.size()-1;
            meas1_points_associations.emplace_back(\
                    matches[matches_filter[i]].queryIdx, current_landmark_idx);
            meas2_points_associations.emplace_back(\
                    matches[matches_filter[i]].trainIdx, current_landmark_idx);
            
            if(verbose) n_added_points++;
        }
        meas1_points_associations.shrink_to_fit();
        meas2_points_associations.shrink_to_fit();

        if(verbose) {
            std::cout << "POINTS ADDED: " << n_added_points << " (" << n_invalid_points << \
                    " invalid, " << n_filtered_points << " filtered out)" << std::endl;
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

} // namespace SLucAM