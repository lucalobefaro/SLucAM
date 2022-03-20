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

    // TODO: create Matcher class that will take the BFMathces or 
    // data associations in the constructor in order to work

    // -----------------------------------------------------------------------------
    // Create Environment and set variables
    // -----------------------------------------------------------------------------
    const std::string dataset_folder =  "../data/datasets/my_synthetic_dataset/";
    const unsigned int n_ransac_iters = 200;
    const unsigned int rotation_only_threshold_rate = 2;
    const unsigned int how_many_meas_optimization = 5;
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
    if(!SLucAM::initialize(state, matcher, n_ransac_iters, rotation_only_threshold_rate)) {
        cout << "ERROR: unable to perform initialization" << endl;
        return 1;
    }
    cout << "DONE!" << endl << endl;


    /* -----------------------------------------------------------------------------
    // OPTIMIZE INITIALIZATION
    // -----------------------------------------------------------------------------
    cout << "OPTIMIZING INITIALIZATION ..." << endl;
    if(state.noMoreMeasurements()) {
        cout << "ERROR: no more measurement to optimize initialization" << endl;
        return 1;
    }
    
    for(unsigned int i=0; i<how_many_meas_optimization; ++i) {
        if(!state.updateState(false)){
            cout << "ERROR: no more measurement to integrate" << endl;
            return 1;
        }
    }
    state.performBundleAdjustment(20, 1000, 1000, 50);
    cout << "DONE!" << endl << endl;
    

    // -----------------------------------------------------------------------------
    // TEST
    // -----------------------------------------------------------------------------
    // Compute the pose of the first pose in the PLD dataset
    cv::Mat first_pose_quaternion = cv::Mat::zeros(4,1,CV_32F);
    first_pose_quaternion.at<float>(0,0) = 0.523157477;
    first_pose_quaternion.at<float>(1,0) = 0.506806552;
    first_pose_quaternion.at<float>(2,0) = -0.474782467;
    first_pose_quaternion.at<float>(3,0) = -0.494000465;
    cv::Mat first_pose_R;
    SLucAM::quaternion_to_matrix(first_pose_quaternion, first_pose_R);
    cv::Mat first_pose_T = cv::Mat::eye(4,4,CV_32F);
    first_pose_T.at<float>(0,3) = -1.62166202;
    first_pose_T.at<float>(1,3) = 1.66993701;
    first_pose_T.at<float>(2,3) = 0.107721001;
    first_pose_T.at<float>(0,0) = first_pose_R.at<float>(0,0);
    first_pose_T.at<float>(0,1) = first_pose_R.at<float>(0,1);
    first_pose_T.at<float>(0,2) = first_pose_R.at<float>(0,2);
    first_pose_T.at<float>(1,0) = first_pose_R.at<float>(1,0);
    first_pose_T.at<float>(1,1) = first_pose_R.at<float>(1,1);
    first_pose_T.at<float>(1,2) = first_pose_R.at<float>(1,2);
    first_pose_T.at<float>(2,0) = first_pose_R.at<float>(2,0);
    first_pose_T.at<float>(2,1) = first_pose_R.at<float>(2,1);
    first_pose_T.at<float>(2,2) = first_pose_R.at<float>(2,2);
    //SLucAM::invert_transformation_matrix(first_pose_T);

    const unsigned int n_poses = state.getPoses().size();
    const vector<cv::Mat>& poses = state.getPoses();

    for(unsigned int i=0; i<n_poses; ++i) {
        SLucAM::visualize_pose_as_quaternion(first_pose_T*poses[i]);
    }
    */

    /*
    // HERE THE FUNCTION TO SPEED TEST
    start = high_resolution_clock::now();
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
    cout << "duration: " << duration2.count() << " microseconds" << endl;
    */
   
    return 0;
}
