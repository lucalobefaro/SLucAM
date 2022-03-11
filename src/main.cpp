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

#include <chrono>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <filesystem>

using namespace std::chrono;
using namespace std;



int main() {

    // -----------------------------------------------------------------------------
    // Create Environment
    // -----------------------------------------------------------------------------
    SLucAM::State state;


    // -----------------------------------------------------------------------------
    // Load Dataset
    // -----------------------------------------------------------------------------
    if(!SLucAM::load_my_dataset("../data/my_dataset/", state)) {
        cout << "ERROR: unable to load the specified dataset" << endl;
        return 1;
    }


    // -----------------------------------------------------------------------------
    // INITIALIZATION
    // -----------------------------------------------------------------------------
    if(!SLucAM::initialize(state, 200)) {
        cout << "ERROR: unable to perform initialization" << endl;
        return 1;
    }


    // -----------------------------------------------------------------------------
    // OPTIMIZE INITIALIZATION
    // -----------------------------------------------------------------------------
    if(state.noMoreMeasurements()) {
        cout << "ERROR: no more measurement to optimize initialization" << endl;
        return 1;
    }
    
    const unsigned int how_many_meas_optimization = 3;
    for(unsigned int i=0; i<how_many_meas_optimization; ++i) {
        if(!state.updateState(false)){
            cout << "ERROR: no more measurement to integrate" << endl;
            return 1;
        }
    }
    // TODO: determine good values for thresholds
    state.performBundleAdjustment(10, 1, 20000, 20000);
    
    /*
    start = high_resolution_clock::now();
    // HERE THE FUNCTION TO SPEED TEST
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
    cout << "duration: " << duration2.count() << " microseconds" << endl;
    */
   
    return 0;
}
