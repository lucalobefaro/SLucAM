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
    const std::string dataset_folder =  "../data/datasets/my_synthetic_dataset/";

    const unsigned int n_ransac_iters = 200;
    const unsigned int rotation_only_threshold_rate = 2;

    const unsigned int how_many_meas_optimization = 4;
    const unsigned int n_iters_POSIT = 50;
    const unsigned int kernel_threshold_POSIT = 1000;
    const float inliers_threshold_POSIT = 5000;

    const unsigned int n_iters_BA = 5;
    const float kernel_threshold_proj_BA = 100;
    const float inliers_threshold_proj_BA = 500;
    const float kernel_threshold_pose_BA = 10;
    const float damping_factor = 1;

    SLucAM::State state;

    std::vector<std::vector<unsigned int>> data_associations;


    // -----------------------------------------------------------------------------
    // Load Dataset
    // -----------------------------------------------------------------------------
    cout << endl << "LOADING THE DATASET ..." << endl;
    if(!SLucAM::load_synthetic_dataset(dataset_folder, state, data_associations)) {
        cout << "ERROR: unable to load the specified dataset" << endl;
        return 1;
    }
    cout << "DONE!" << endl << endl;


    // -----------------------------------------------------------------------------
    // Create Matcher
    // -----------------------------------------------------------------------------
    SLucAM::Matcher matcher(data_associations);


    // -----------------------------------------------------------------------------
    // INITIALIZATION
    // -----------------------------------------------------------------------------
    cout << "INITIALIZATION ..." << endl;
    if(!state.initializeState(matcher, n_ransac_iters, rotation_only_threshold_rate)) {
        cout << "ERROR: unable to perform initialization" << endl;
        return 1;
    }
    cout << "DONE!" << endl << endl;


    // -----------------------------------------------------------------------------
    // OPTIMIZE INITIALIZATION
    // -----------------------------------------------------------------------------
    cout << "OPTIMIZING INITIALIZATION ..." << endl;
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
                                            damping_factor)){
            cout << "ERROR: no more measurement to integrate" << endl;
            return 1;
        }
    }
    //state.performBundleAdjustment(n_iters_BA, \
                                    kernel_threshold_proj_BA, \
                                    inliers_threshold_proj_BA, \
                                    kernel_threshold_pose_BA, \
                                    damping_factor);
    

    // -----------------------------------------------------------------------------
    // TEST
    // -----------------------------------------------------------------------------
    const unsigned int n_poses = state.getPoses().size();
    for(unsigned int i=0; i<n_poses; ++i) {
        cout << "POSE " << i << endl;
        cout << state.getPoses()[i] << endl << endl;
    }


    /*
    // HERE THE FUNCTION FOR SPEED TEST
    start = high_resolution_clock::now();
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
    cout << "duration: " << duration2.count() << " microseconds" << endl;
    */
   
    return 0;
}
