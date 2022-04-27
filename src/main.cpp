// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <SLucAM_image.h>
#include <SLucAM_geometry.h>
#include <SLucAM_initialization.h>
#include <SLucAM_measurement.h>
#include <SLucAM_state.h>
#include <SLucAM_dataset.h>
#include <SLucAM_visualization.h>

using namespace std;



int main() {

    // -----------------------------------------------------------------------------
    // Create Environment and set variables
    // -----------------------------------------------------------------------------
    bool synthetic = false;
    const bool verbose = true;
    const bool save_exploration = true;
    
    SLucAM::State state;
    std::string dataset_folder;
    if(synthetic) 
        dataset_folder = "../data/datasets/my_synthetic_dataset/";
    else
        dataset_folder = "../data/datasets/tum_dataset_2/";
    const std::string results_folder = "../results/";

    const unsigned int n_orb_features = 1000;

    const unsigned int how_many_meas_optimization = 10;
    const unsigned int n_iters_POSIT = 50;
    const unsigned int kernel_threshold_POSIT = 100;
    const float inliers_threshold_POSIT = 500;
    const float damping_factor = 1;
    const unsigned int n_iters_BA = 50;
    
    const unsigned int local_map_size = 8;
    const float parallax_threshold = 1.0;
    const float new_landmark_threshold = 0.02;
    
    unsigned int step = 0;
    std::vector<std::vector<unsigned int>> associations;

    

    // -----------------------------------------------------------------------------
    // Load Dataset
    // -----------------------------------------------------------------------------
    cv::Ptr<cv::Feature2D> orb_detector = cv::ORB::create(n_orb_features);
    cout << endl << "--- LOADING THE DATASET ---" << endl;
    bool loaded;
    if(synthetic)
        loaded = SLucAM::load_synthetic_dataset(dataset_folder, state, associations);
    else 
        loaded = SLucAM::load_TUM_dataset(dataset_folder, state, orb_detector, verbose);
    if(!loaded) {
        cout << "ERROR: unable to load the specified dataset" << endl;
        return 1;
    }
    cout << "--- DONE! ---" << endl << endl;


    // -----------------------------------------------------------------------------
    // Create Matcher
    // -----------------------------------------------------------------------------
    SLucAM::Matcher matcher;
    //SLucAM::Matcher matcher(associations);


    // -----------------------------------------------------------------------------
    // INITIALIZATION
    // -----------------------------------------------------------------------------
    cout << "--- INITIALIZATION ---" << endl;
    if(!state.initializeState(matcher, \
                                parallax_threshold, \
                                verbose)) {
        cout << "ERROR: unable to perform initialization" << endl;
        return 1;
    }
    if(save_exploration) {
        SLucAM::save_current_state(results_folder+"keyframe"+std::to_string(step)+"_", state);
        step++;
    }
    cout << "--- DONE! ---" << endl << endl;



    /* -----------------------------------------------------------------------------
    // OPTIMIZE INITIALIZATION
    // -----------------------------------------------------------------------------
    cout << "--- OPTIMIZING INITIALIZATION ---" << endl;
    if(state.reaminingMeasurements() == 0) {
        cout << "ERROR: no more measurement to optimize initialization" << endl;
        return 1;
    }
    
    for(unsigned int i=0; i<how_many_meas_optimization; ++i) {
        if(!state.integrateNewMeasurement(matcher, \
                                            false, \
                                            n_iters_POSIT, \
                                            kernel_threshold_POSIT, \
                                            inliers_threshold_POSIT, \
                                            damping_factor, \
                                            triangulation_window, \
                                            parallax_threshold, \
                                            new_landmark_threshold, \
                                            verbose)) {
            cout << "ERROR: no more measurement to integrate or no enough correspondances finded" << endl;
            return 1;
        }
    }
    state.performTotalBA(n_iters_BA, verbose);
    cout << "--- DONE! ---" << endl << endl;
    */

    /* -----------------------------------------------------------------------------
    // INTEGRATE NEW MEASUREMENT AND EXPAND MAP
    // -----------------------------------------------------------------------------
    cout << "--- ESPLORATION STARTED ---" << endl;
    while(state.reaminingMeasurements() != 0) {
        state.integrateNewMeasurement(matcher, \
                                    true, \
                                    n_iters_POSIT, \
                                    kernel_threshold_POSIT, \
                                    inliers_threshold_POSIT, \
                                    damping_factor, \
                                    local_map_size, \
                                    parallax_threshold, \
                                    new_landmark_threshold, \
                                    verbose);
        if(save_exploration) {
            SLucAM::save_current_state(results_folder+"keyframe"+std::to_string(step)+"_", state);
            step++;
        }
    }
    //state.performTotalBA(n_iters_BA, verbose);
    cout << "--- DONE! ---" << endl << endl;
    */

    /* -----------------------------------------------------------------------------
    // SAVE RESULTS
    // -----------------------------------------------------------------------------
    SLucAM::save_poses(dataset_folder, state.getPoses());
    SLucAM::save_landmarks(dataset_folder, state.getLandmarks());
    */
    if(!synthetic) SLucAM::save_TUM_results(dataset_folder, state);


    /* -----------------------------------------------------------------------------
    // TEST
    // -----------------------------------------------------------------------------
    
    // Visualize predicted poses
    const unsigned int n_poses = state.getPoses().size();
    for(unsigned int i=0; i<n_poses; ++i) {
        cout << "POSE " << i << endl;
        cout << state.getPoses()[i] << endl << endl;
    }

    // Test predicted 3D points
    cout << "#LANDMARKS PREDICTED: " << state.getLandmarks().size() << endl;
    //cout << "LANDMARKS PREDICTION ERROR: " << SLucAM::test_predicted_points(dataset_folder, \
                                                        state.getKeyframes(), \
                                                        state.getLandmarks(), \
                                                        data_associations) \
        << endl << endl;

    
    // Test keyframes
    cout << "#KEYFRAMES: " << state.getKeyframes().size() << endl;
    for(unsigned int i=0; i<state.getKeyframes().size(); ++i) {
        cout << "#" << i << ", MEAS: " << state.getKeyframes()[i].getMeasIdx() \
            << ", POSE: " << state.getKeyframes()[i].getPoseIdx() \
            << ", OBSERVED LANDMARKS:[";
            for(unsigned int j=0; j< state.getKeyframes()[i].getKeyframesAssociations().size(); ++j) {
                cout << " " << state.getKeyframes()[i].getKeyframesAssociations()[j];
            }
            cout << " ], #PREDICTED POINTS: " << state.getKeyframes()[i].getPointsAssociations().size() \
            << "/" << state.getMeasurements()[state.getKeyframes()[i].getMeasIdx()].getPoints().size() << endl;
    }
    */
    
   
    // OTHER TEST
    if(synthetic && verbose) {
        const unsigned int n_posess = state.getPoses().size();
        for(unsigned int i=0; i<n_posess; ++i) {
            cout << "POSE " << i << endl;
            cout << state.getPoses()[i] << endl << endl;
        }
        const unsigned int n_keyframes = state.getKeyframes().size();
        for(unsigned int i=0; i<n_keyframes; ++i) {
            cout << "KEYFRAME " << i << endl;
            cout << state.getKeyframes()[i] << endl << endl;
        }
        const unsigned int n_landmarks = state.getLandmarks().size();
        cout << "LANDMARKS:" << endl;
        for(unsigned int i=0; i<n_landmarks; ++i) {
            cout << "\tID: " << i << " " <<\
                state.getLandmarks()[i].x << " " <<\
                state.getLandmarks()[i].y << " " <<\
                state.getLandmarks()[i].z << " " << endl;
        }
    }
    // END TEST 
    

    /*
    // HERE THE FUNCTION FOR SPEED TEST
    start = high_resolution_clock::now();
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
    cout << "duration: " << duration2.count() << " microseconds" << endl;
    */
   
    return 0;
}







/* RANSAC OF OPENCV
    std::cout << std::endl << "RANSAC OPENCV" << std::endl;
    const SLucAM::Measurement& meas1 = state.getMeasurements()[0];
    const SLucAM::Measurement& meas2 = state.getMeasurements()[1];
    const unsigned int n_points1 = meas1.getPoints().size();
    const unsigned int n_points2 = meas2.getPoints().size();
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    points1.reserve(n_points1);
    points2.reserve(n_points2);
    cv::Mat W_mat = cv::Mat::zeros(3,3,CV_32F);
    W_mat.at<float>(0,1) = -1;
    W_mat.at<float>(1,0) = 1;
    W_mat.at<float>(2,2) = 1;
    const cv::Mat& K = state.getCameraMatrix();

    std::vector<cv::DMatch> matches;
    matcher.match_measurements(meas1, meas2, matches);
    const unsigned int n_matches = matches.size();

    for(unsigned int i=0; i<n_matches; ++i) {
        points1.emplace_back(meas1.getPoints()[matches[i].queryIdx].pt.x,
                                meas1.getPoints()[matches[i].queryIdx].pt.y);
        points2.emplace_back(meas2.getPoints()[matches[i].trainIdx].pt.x,
                                meas2.getPoints()[matches[i].trainIdx].pt.y);
    }
    points1.shrink_to_fit();
    points2.shrink_to_fit();

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.99, 1, mask);
    std::cout << E << std::endl;
    cv::Mat R, t;
    cv::recoverPose(E, points1, points2, K, R, t, mask);

    cv::Mat X = cv::Mat::eye(4,4,CV_32F);
    R.copyTo(X.rowRange(0,3).colRange(0,3));
    t.copyTo(X.rowRange(0,3).col(3));
    std::cout << X << std::endl;
    */