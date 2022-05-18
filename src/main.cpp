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
    const bool verbose = true;
    const bool save_exploration = true;

    //std::string features = "orb";
    std::string features = "superpoint";
    
    SLucAM::State state;
    std::string dataset_folder = "../data/datasets/tum_xyz/";
    //std::string dataset_folder = "../data/datasets/tum_desk/"
    const std::string results_folder = "../results/";

    const unsigned int kernel_threshold_POSIT = 1000;   // 1000 => 33 pixels (?)
    const float inliers_threshold_POSIT = kernel_threshold_POSIT;
    
    unsigned int step = 0;
    std::vector<std::vector<unsigned int>> associations;

    

    // -----------------------------------------------------------------------------
    // Load Dataset
    // -----------------------------------------------------------------------------
    SLucAM::FeatureExtractor feature_extractor = SLucAM::FeatureExtractor(false);
    cout << endl << "--- LOADING THE DATASET ---" << endl;
    bool loaded;
    if(features == "orb")
        loaded = SLucAM::load_TUM_dataset(dataset_folder, state, feature_extractor, verbose);
    else if(features == "superpoint")
        loaded = SLucAM::load_preextracted_TUM_dataset(dataset_folder, features+"/", state, verbose);
    else {
        std::cout << "ERROR: invalid features type: " << features << std::endl;
        return 1;
    }
    if(!loaded) {
        cout << "ERROR: unable to load the specified dataset" << endl;
        return 1;
    }
    cout << "--- DONE! ---" << endl << endl;



    // -----------------------------------------------------------------------------
    // Create Matcher
    // -----------------------------------------------------------------------------
    SLucAM::Matcher matcher(features);



    // -----------------------------------------------------------------------------
    // INITIALIZATION
    // -----------------------------------------------------------------------------
    cout << "--- INITIALIZATION ---" << endl;
    if(!state.initializeState(matcher, verbose)) {
        cout << "ERROR: unable to perform initialization" << endl;
        return 1;
    }
    if(save_exploration) {
        SLucAM::save_current_state(results_folder+"frame"+std::to_string(step)+"_", state);
        step++;
    }
    cout << "--- DONE! ---" << endl << endl;



    // -----------------------------------------------------------------------------
    // INTEGRATE NEW MEASUREMENT AND EXPAND MAP
    // -----------------------------------------------------------------------------
    cout << "--- ESPLORATION STARTED ---" << endl;
    while(state.reaminingMeasurements() != 0) {
        if(state.integrateNewMeasurement(matcher, \
                                    true, \
                                    kernel_threshold_POSIT, \
                                    inliers_threshold_POSIT, \
                                    verbose)) {
            if(save_exploration) {
                SLucAM::save_current_state(results_folder+"frame"+std::to_string(step)+"_", state);
                step++;
            }
        } else {
            std::cout << "UNABLE TO KEEP TRACKING" << std::endl;
            break;
        }
    }
    cout << "--- DONE! ---" << endl << endl;



    /* -----------------------------------------------------------------------------
    // PERFORM FINAL BUNDLE ADJUSTMENT
    // -----------------------------------------------------------------------------
    state.performTotalBA(10, verbose);
    if(save_exploration) {
        SLucAM::save_current_state(results_folder+"frame"+std::to_string(step)+"_", state);
        step++;
    }
    */


    // -----------------------------------------------------------------------------
    // SAVE RESULTS
    // -----------------------------------------------------------------------------
    SLucAM::save_TUM_results(dataset_folder, state);



    return 0;
}