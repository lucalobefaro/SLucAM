//
// SLucAM_dataset.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_dataset.h>
#include <SLucAM_image.h>
#include <filesystem>
#include <fstream>


// TODO: delete this
#include <iostream>
using namespace std;



// -----------------------------------------------------------------------------
// Implementation of functions to deal with my personal dataset format
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function to load my personal dataset.
    * Inputs:
    *   dataset_folder: folder where to find all the images and
    *           the specification of the camera matrix in a file
    *           with name "camera.dat"
    *   state: state object where to store loaded infos
    */
    bool load_my_dataset(const std::string& dataset_folder, State& state) {

        // Initialization
        std::string current_filename;
        cv::Mat current_img, K;
        bool K_loaded = false;
        cv::Ptr<cv::Feature2D> orb_detector = cv::ORB::create();
        std::vector<Measurement> measurements;
        
        // For each element in the folder
        measurements.reserve(100);  // TODO: reserve better the space
        for (const auto& entry : \
                    std::filesystem::directory_iterator(dataset_folder)) {
            
            // Take the current filename
            current_filename = entry.path();
            
            // If it is the camera matrix file, load K
            if(current_filename == dataset_folder+"camera.dat") {
                if(load_camera_matrix(current_filename, K))
                    K_loaded = true;
                continue;
            }

            // Otherwise load the measurement
            if(!load_image(current_filename, current_img))
                return false;
            
            // Detect keypoints
            std::vector<cv::KeyPoint> points;
            cv::Mat descriptors;
            orb_detector->detectAndCompute(current_img, cv::Mat(), \
                                            points, descriptors);

            // Create new measurement
            measurements.emplace_back(Measurement(points, descriptors));

        }
        measurements.shrink_to_fit();

        // In case no camera.dat file is found return error
        if(!K_loaded)
            return false;

        // Initialize the state
        // TODO: correctly determine the number of landmarks to reserve
        state = State(K, measurements, measurements.size(), 5000);
        
        return true;
            
    }


    /*
    * Load the camera matrix from the "camera.dat" file in my dataset.
    * It returns false in case of errors.
    */
    bool load_camera_matrix(const std::string& filename, cv::Mat& K) {

        K = cv::Mat::zeros(3,3,CV_32F);

        std::fstream camera_file;
        camera_file.open(filename);
        if(camera_file.fail()) return false;
        camera_file >> \
            K.at<float>(0,0) >> K.at<float>(0,1) >> K.at<float>(0,2) >> \
            K.at<float>(1,0) >> K.at<float>(1,1) >> K.at<float>(1,2) >> \
            K.at<float>(2,0) >> K.at<float>(2,1) >> K.at<float>(2,2);
        camera_file.close();

        return true;
    }

} // namespace SLucAM