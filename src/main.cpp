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

#include <chrono>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std::chrono;
using namespace std;



int main() {
    
    // -----------------------------------------------------------------------------
    // Feature extraction
    // -----------------------------------------------------------------------------
    string filename_img1 = "../data/images/image1.jpg";
    string filename_img2 = "../data/images/image2.jpg";
    cv::Ptr<cv::Feature2D> orb_detector = cv::ORB::create();
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);

    SLucAM::Measurement meas1(filename_img1, orb_detector);
    SLucAM::Measurement meas2(filename_img2, orb_detector);

    std::vector<SLucAM::Measurement> measurements = {meas1, meas2};

    
    // -----------------------------------------------------------------------------
    // Load camera infos
    // -----------------------------------------------------------------------------
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = 852.6132831012704;
    K.at<float>(0,1) = 0;
    K.at<float>(0,2) = 481.24842468649103;
    K.at<float>(1,0) = 0;
    K.at<float>(1,1) = 859.3748334552829;
    K.at<float>(1,2) = 259.6723142881928;
    K.at<float>(2,0) = 0;
    K.at<float>(2,1) = 0;
    K.at<float>(2,2) = 1;


    // -----------------------------------------------------------------------------
    // Create state
    // -----------------------------------------------------------------------------
    SLucAM::State state(K,measurements,2,221);


    // -----------------------------------------------------------------------------
    // INITIALIZATION TEST
    // -----------------------------------------------------------------------------    
    auto start = high_resolution_clock::now();
    SLucAM::initialize(state, matcher, 0, 1);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "duration: " << duration.count() << " milliseconds" << endl << endl;
    
    cout << "First pose: " << endl << state.getPoses()[0] << endl << endl;
    cout << "Second pose (predicted): " << endl << state.getPoses()[1] << endl << endl;
    cout << "First point triangulated: " << endl << state.getLandmarks()[0] << endl << endl;
    cout << "Second point triangulated: " << endl << state.getLandmarks()[1] << endl << endl;
    cout << "----------- INITIALIZATION DONE ------------" << endl << endl << endl;


    // -----------------------------------------------------------------------------
    // BUNDLE ADJUSTMENT TEST
    // ----------------------------------------------------------------------------- 
    cv::Mat H = cv::Mat::zeros(12 + 6, 12 + 6, CV_32F); 
    cv::Mat b = cv::Mat::zeros(12 + 6, 1, CV_32F);
    float chi_tot;

    std::vector<cv::Point3f> sub_landmarks = std::vector<cv::Point3f>(\
                                                &state.getLandmarks()[0], \
                                                &state.getLandmarks()[2]);

    start = high_resolution_clock::now();
    //SLucAM::State::buildLinearSystemProjections(state.getPoses(), \
                                            sub_landmarks, \
                                            state.getMeasurements(), \
                                            state.getAssociations(), \
                                            state.getCameraMatrix(), \
                                            H, b, chi_tot);
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
    cout << "duration: " << duration2.count() << " microseconds" << endl;

    cout << endl << H << endl << endl;
    cout << endl << b << endl << endl;

    /*cout << endl << endl << "ASSOCIATIONS:" << endl;
    for(auto& a : state.getAssociations()) {
        if(a.landmark_idx > 1) break;
        cout << "Pose: " << a.pose_idx << endl;
        cout << "Landmark: " << a.landmark_idx << endl;
        cout << "Measurement: " << a.measurement_idx << endl;
        cout << "Point: " << a.point_idx << endl;
        cout << "Efective point: [" << \
            state.getMeasurements()[a.measurement_idx].getPoints()[a.point_idx].pt.x \
            << ", " << state.getMeasurements()[a.measurement_idx].getPoints()[a.point_idx].pt.y \
            << "]" << endl << endl;
    }*/

    return 0;
}



/* Load synthetic data
    std::vector<cv::KeyPoint> p_img1(274);
    std::vector<cv::KeyPoint> p_img2(305);
    std::ifstream f;
    f.open("../data/pimg1.dat", ios_base::in);
    for(unsigned int i=0; i<274; ++i )  {
        f >> p_img1[i].pt.x >> p_img1[i].pt.y;
    }
    f.close();
    f.open("../data/pimg2.dat", ios_base::in);
    for(unsigned int i=0; i<305; ++i )  {
        f >> p_img2[i].pt.x >> p_img2[i].pt.y;
    }
    f.close();
    vector<cv::DMatch> matches(221);
    f.open("../data/correspondances.dat", ios_base::in);
    for(unsigned int i=0; i<221; ++i )  {
        f >> matches[i].queryIdx >> matches[i].trainIdx;
    }
    f.close();
    
    std::vector<cv::KeyPoint> p_img1_normalized, p_img2_normalized;
    cv::Mat T1, T2;
    SLucAM::normalize_points(p_img1, p_img1_normalized, T1);
    SLucAM::normalize_points(p_img2, p_img2_normalized, T2);
    */