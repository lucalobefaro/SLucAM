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
    // Load Dataset
    // -----------------------------------------------------------------------------
    SLucAM::State state;
    SLucAM::load_my_dataset("../data/my_dataset/", state);


    // -----------------------------------------------------------------------------
    // INITIALIZATION
    // -----------------------------------------------------------------------------


    
    /* -----------------------------------------------------------------------------
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
    /* -----------------------------------------------------------------------------
    
    // Create the guessed pose and landmarks
    cv::Mat guessed_pose = (cv::Mat_<float>(4,4) << \
                    0.93969262, -0.34202009, 1.0313163e-06, 0.72615826, \
                    0.34201992, 0.93969268, 9.6002213e-07, 0.68752754, \
                    -1.2869932e-06, -6.0532057e-07, 1, -7.8231224e-06, \
                    0, 0, 0, 1);
    cv::Point3f l1(12.34, 45.6, 3.45);
    cv::Point3f l2(11.34, 5.6, 1.45);
    cv::Point3f l3(12.4, 4.6, 54.0);
    cv::Point3f l4(2.34, 8.2, 23.1);
    cv::Point3f l5(5.46, 23.7, 6.8);
    cv::Point3f l6(12.0, 6.54, 354.6);
    cv::Point3f l7(11.67, 34.78, 3.45);
    cv::Point3f l8(14.34, 5.67, 89.5);
    std::vector<cv::Point3f> landmarks = {l1, l2, l3, l4, l5, l6, l7, l8};

    // Create measurement 1
    std::vector<cv::KeyPoint> points1 = {};
    SLucAM::Measurement new_meas1(points1);

    // Create measurement 2
    cv::KeyPoint p1(cv::Point2f(54, 4), 1);
    cv::KeyPoint p2(cv::Point2f(59, 45), 1);
    cv::KeyPoint p3(cv::Point2f(4, 487), 1);
    cv::KeyPoint p4(cv::Point2f(1, 1), 1);
    cv::KeyPoint p5(cv::Point2f(4, 4), 1);
    cv::KeyPoint p6(cv::Point2f(45, 64), 1);
    cv::KeyPoint p7(cv::Point2f(8, 89), 1);
    cv::KeyPoint p8(cv::Point2f(98, 89), 1);
    cv::KeyPoint p9(cv::Point2f(2, 32), 1);
    std::vector<cv::KeyPoint> points2 = {p1, p2, p3, p4, p5, p6, p7, p8, p9};
    SLucAM::Measurement new_meas2(points2);

    // Create measurements
    std::vector<SLucAM::Measurement> new_measurements = {new_meas1, new_meas2};

    // Create landmark_observations
    std::vector<SLucAM::LandmarkObservation> landmark_observations;
    landmark_observations.emplace_back(0, 0, 0, 0);
    landmark_observations.emplace_back(0, 0, 1, 0);

    landmark_observations.emplace_back(0, 1, 0, 0);
    landmark_observations.emplace_back(0, 1, 1, 3);

    landmark_observations.emplace_back(0, 2, 0, 0);
    landmark_observations.emplace_back(0, 2, 1, 7);

    landmark_observations.emplace_back(0, 3, 0, 0);
    landmark_observations.emplace_back(0, 3, 1, 8);

    landmark_observations.emplace_back(0, 4, 0, 0);
    landmark_observations.emplace_back(0, 4, 1, 6);

    landmark_observations.emplace_back(0, 5, 0, 0);
    landmark_observations.emplace_back(0, 5, 1, 2);

    start = high_resolution_clock::now();
    SLucAM::perform_Posit(guessed_pose, \
                  new_measurements, \
                  1, \
                  landmark_observations, \
                  landmarks, \
                  state.getCameraMatrix(), \
                  1000, \
                  20000000, \
                  20000000, \
                  1);
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
    cout << "duration: " << duration2.count() << " microseconds" << endl;

    cout << endl << "Optimized Pose: " << endl << guessed_pose << endl;
    */
   
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