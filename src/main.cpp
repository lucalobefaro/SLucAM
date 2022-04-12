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

#include <chrono>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <filesystem>

#include <Eigen/Core>
#include <Eigen/Geometry>


using namespace std::chrono;
using namespace std;



int main() {

    // -----------------------------------------------------------------------------
    // Create Environment and set variables
    // -----------------------------------------------------------------------------
    const std::string dataset_folder =  "../data/datasets/tum_dataset_teddy/";
    const std::string results_folder = "../results/";

    const unsigned int n_orb_features = 1000;
    const unsigned int n_ransac_iters = 200;
    const unsigned int rotation_only_threshold_rate = 2;

    const unsigned int how_many_meas_optimization = 24;
    const unsigned int n_iters_POSIT = 50;
    const unsigned int kernel_threshold_POSIT = 1000;
    const float inliers_threshold_POSIT = 5000;

    const unsigned int n_iters_BA = 10;
    const float kernel_threshold_proj_BA = 100;
    const float inliers_threshold_proj_BA = 500;
    const float kernel_threshold_pose_BA = 10;
    const float damping_factor = 1;

    const unsigned int triangulation_window = 6;
    const float parallax_threshold = 1.0;
    const float new_landmark_threshold = 0.01;

    const bool verbose = true;
    const bool save_exploration = true;
    unsigned int step = 0;

    SLucAM::State state;


    // -----------------------------------------------------------------------------
    // Load Dataset
    // -----------------------------------------------------------------------------
    cv::Ptr<cv::Feature2D> orb_detector = cv::ORB::create(n_orb_features);
    cout << endl << "--- LOADING THE DATASET ---" << endl;
    if(!SLucAM::load_TUM_dataset(dataset_folder, state, orb_detector, verbose)) {
        cout << "ERROR: unable to load the specified dataset" << endl;
        return 1;
    }
    cout << "--- DONE! ---" << endl << endl;


    // -----------------------------------------------------------------------------
    // Create Matcher
    // -----------------------------------------------------------------------------
    SLucAM::Matcher matcher;


    // -----------------------------------------------------------------------------
    // INITIALIZATION
    // -----------------------------------------------------------------------------
    cout << "--- INITIALIZATION ---" << endl;
    if(!state.initializeState(matcher, n_ransac_iters, \
                                rotation_only_threshold_rate, \
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

    // -----------------------------------------------------------------------------
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
                                    triangulation_window, \
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
    

    /* -----------------------------------------------------------------------------
    // SAVE RESULTS
    // -----------------------------------------------------------------------------
    SLucAM::save_poses(dataset_folder, state.getPoses());
    SLucAM::save_landmarks(dataset_folder, state.getLandmarks());
    */


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
    

    /* OTHER TEST
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
    // END TEST */
    

    /*
    // HERE THE FUNCTION FOR SPEED TEST
    start = high_resolution_clock::now();
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
    cout << "duration: " << duration2.count() << " microseconds" << endl;
    */
   
    return 0;
}
