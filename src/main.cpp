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
    /* -----------------------------------------------------------------------------
    string filename_img1 = "../data/images/image1.jpg";
    string filename_img2 = "../data/images/image2.jpg";
    cv::Ptr<cv::Feature2D> orb_detector = cv::ORB::create();
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);

    SLucAM::Measurement meas1(filename_img1, orb_detector);
    SLucAM::Measurement meas2(filename_img2, orb_detector);
    */

    // -----------------------------------------------------------------------------
    // INITIALIZATION TEST
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
    SLucAM::State state(K,2,221);

    // Load synthetic data
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

    auto start = high_resolution_clock::now();
    SLucAM::initialize(state, p_img1, p_img2, p_img1_normalized, p_img2_normalized, T1, T2, matches, 0, 1);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "duration: " << duration.count() << " milliseconds" << endl;
    
    cout << "X predicted: " << endl << state.getPoses()[0] << endl << endl;
    cout << "First point triangulated: " << endl << state.getLandmarks()[0] << endl;
    cout << "----------- INITIALIZATION DONE ------------" << endl << endl << endl;


    // TEST BUNDLE ADJUSTMENT
    cv::Mat J_1;
    cv::Mat J_2;
    cv::Mat error;
    cv::Mat pose_1 = cv::Mat::eye(4,4,CV_32F);
    pose_1.at<float>(0,0) = 0.9848;
    pose_1.at<float>(0,1) = -0.1736;
    pose_1.at<float>(0,3) = 0.0100;
    pose_1.at<float>(1,0) = 0.1736;
    pose_1.at<float>(1,1) = 0.9848;
    pose_1.at<float>(1,3) = 0.03;
    pose_1.at<float>(2,2) = 1;
    pose_1.at<float>(3,3) = 1;
    cv::Mat pose2_wrt_pose1 =  state.getPoses()[0];

    cv::Mat pose_2 = (cv::Mat_<float>(4,4) << 1.0460,  -0.3200,   0.1800,   0.7857,\
   0.6800   ,1.0460  , 0.1800 ,  1.0132,\
   0.1800   ,0.1800  , 1.1800,   0.1800,\
   0.1800  , 0.1800  , 0.1800 ,  1.1800);

    cout << state.getPoses()[0] << endl;
    cout << state.getLandmarks()[0] << endl;
    cout << state.getLandmarks()[1] << endl;

    cv::Mat dx = (cv::Mat_<float>(12,1) << 3, 4, 5, 0.13, 0.45, 0.9, 8,56,89,13,12,14, 1, 2, 3, 10, 11, 12);

    start = high_resolution_clock::now();
    //state.boxPlus(dx);
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
    cout << "duration E&J: " << duration2.count() << " microseconds" << endl;
    
    cout << state.getPoses()[1] << endl;
    cout << state.getLandmarks()[0] << endl;
    cout << state.getLandmarks()[1] << endl;

    return 0;
}