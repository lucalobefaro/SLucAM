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
#include <SLucAM_keyframe.h>

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
        dataset_folder = "../data/datasets/tum_dataset_teddy/";
    const std::string results_folder = "../results/";

    const unsigned int n_orb_features = 1000;
    const unsigned int n_iters_ransac = 200;

    const unsigned int how_many_meas_optimization = 10;

    const unsigned int kernel_threshold_POSIT = 1000;   // 1000 => 33 pixels (?)
    const float inliers_threshold_POSIT = kernel_threshold_POSIT;

    const unsigned int n_iters_total_BA = 20;
    
    const unsigned int local_map_size = 6;
    const float parallax_threshold = 1.0;
    const float new_landmark_threshold = 0.08;
    
    unsigned int step = 0;
    std::vector<std::vector<unsigned int>> associations;

    

    // -----------------------------------------------------------------------------
    // Load Dataset
    // -----------------------------------------------------------------------------
    SLucAM::FeatureExtractor feature_extractor = SLucAM::FeatureExtractor(false);
    cout << endl << "--- LOADING THE DATASET ---" << endl;
    bool loaded;
    if(synthetic)
        loaded = SLucAM::load_synthetic_dataset(dataset_folder, state, associations);
    else 
        loaded = SLucAM::load_TUM_dataset(dataset_folder, state, feature_extractor, verbose);
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
                                n_iters_ransac, \
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



    // -----------------------------------------------------------------------------
    // INTEGRATE NEW MEASUREMENT AND EXPAND MAP
    // -----------------------------------------------------------------------------
    cout << "--- ESPLORATION STARTED ---" << endl;
    unsigned int n_integrated = 0;
    while(state.reaminingMeasurements() != 0) {
        if(!state.integrateNewMeasurement(matcher, \
                                    true, \
                                    local_map_size, \
                                    kernel_threshold_POSIT, \
                                    inliers_threshold_POSIT, \
                                    parallax_threshold, \
                                    new_landmark_threshold, \
                                    verbose)) {
            std::cout << std::endl << \
                    "UNABLE TO CONTINUE EXPLORATION (it needs re-initialization)" \
                << std::endl << std::endl;
            break;
        }
        n_integrated++;
        if(n_integrated%30 == 0) {
            state.performTotalBA(10, false);
        }
        if(save_exploration) {
            SLucAM::save_current_state(results_folder+"keyframe"+std::to_string(step)+"_", state);
            step++;
        }
    }
    cout << "--- DONE! ---" << endl << endl;
    


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
    */
    return 0;
}