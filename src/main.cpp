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
    /* ----------------------------------------------------------------------------- 
    start = high_resolution_clock::now();
    state.performBundleAdjustment(50, 1, 20000000000, 20000000000);
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop - start);
    cout << "duration: " << duration2.count() << " milliseconds" << endl;

    cout << endl << "First pose:" << endl << state.getPoses()[0] << endl;
    cout << endl << "Second pose:" << endl << state.getPoses()[1] << endl;
    cout << endl << "First landmark:" << endl << state.getLandmarks()[0] << endl;
    cout << endl << "Second landmark:" << endl << state.getLandmarks()[1] << endl;
    cout << "----------- BUNDLE ADJUSTMENT DONE ------------" << endl << endl << endl;
    */

    // -----------------------------------------------------------------------------
    // PROJECTIVE ICP TEST
    // -----------------------------------------------------------------------------
    
    cv::Mat Xr = (cv::Mat_<float>(4,4) << \
    0.93969262, -0.34202009, 1.0313163e-06, 0.72615826, \
    0.34201992, 0.93969268, 9.6002213e-07, 0.68752754, \
    -1.2869932e-06, -6.0532057e-07, 1, -7.8231224e-06, \
    0, 0, 0, 1);

    cv::Point3f Xl = cv::Point3f(10.0757,-6.46847,31.9809);

    cv::KeyPoint z = cv::KeyPoint(cv::Point2f(46.06, 13.875), 1);

    cv::Mat error = cv::Mat::zeros(2,1, CV_32F);
    cv::Mat J = cv::Mat::zeros(2,6, CV_32F);
    bool res = SLucAM::error_and_jacobian_Posit(Xr, Xl, z, K, 2*K.at<float>(1,2), 2*K.at<float>(0,2),\
                                    error, J);

    cout << res << endl;
    cout << error << endl;
    cout << J << endl;

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