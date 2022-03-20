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
#include <iostream>


// TODO: delete this
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



// -----------------------------------------------------------------------------
// Implementation of functions to deal with the Pering Laboratory Dataset
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    /*
    * This function allows to load the PRD dataset. If it is the first time we
    * load it, we use the images to load the keypoints and we save them in
    * a set of files, otherwise we use directly the data in the files.
    * IMPORTANT: in order to understand if the data are already extracted
    * previously, this function verify if the folder ./cam0/extracted_data/
    * already exsists inside the specified dataset_folder, so in case of
    * errors, please delete that folder before to start this function.
    */  
    bool load_PRD_dataset(const std::string& dataset_folder, State& state) {

        // Initialization
        const std::string csv_filename = dataset_folder+"cam0/data.csv";
        const std::string imgs_folder = dataset_folder+"cam0/data/";
        const std::string extracted_data_folder = dataset_folder+"cam0/extracted_data/";
        const std::string camera_filename = dataset_folder+"cameraInfo.txt";
        std::string current_line, current_el_filename;
        unsigned int start_str_pose;
        unsigned int tot_n_points = 0;
        cv::Mat current_img, K;
        cv::Ptr<cv::Feature2D> orb_detector = cv::ORB::create();
        std::vector<Measurement> measurements;

        // Check if the data from images are already extracted
        // (if it is not the first time we use this dataset)
        bool already_extracted = false;
        if(std::filesystem::is_directory(extracted_data_folder))
            already_extracted = true;

        // If the data from images are not already extracted, create
        // the directory where to save the extracted data
        if(!already_extracted)
            std::filesystem::create_directory(extracted_data_folder);

        // Load the camera matrix
        if(!load_PRD_camera_matrix(camera_filename, K)) 
            return false;

        // Open the csv file
        std::fstream csv_file;
        csv_file.open(csv_filename);
        if(csv_file.fail()) return false;

        // Ignore the first line
        std::getline(csv_file, current_line);

        // Load all images (if already extracted load infos from file)
        measurements.reserve(1600);
        while(std::getline(csv_file, current_line)) {
            
            // Get the img filename
            std::stringstream ss(current_line);
            std::getline(ss, current_line, ',');
            std::getline(ss, current_el_filename, ',');

            // If the keypoints from the images are not already extracted then 
            // load the image, detect keypoints and save results on a file, 
            // otherwise load them from the files
            std::vector<cv::KeyPoint> points;
            cv::Mat descriptors;
            if(!already_extracted) {
                if(!load_image(imgs_folder+current_el_filename, current_img)) {
                    csv_file.close();
                    return false;
                }
                orb_detector->detectAndCompute(current_img, cv::Mat(), \
                                                points, descriptors);
                current_el_filename = current_el_filename.substr(0, current_el_filename.size()-3) \
                                            + "yml";
                if(!save_keypoints_on_file(extracted_data_folder+current_el_filename, points, descriptors))
                    return false;
            } else {
                current_el_filename = current_el_filename.substr(0, current_el_filename.size()-3) \
                                            + "yml";
                if(!load_keypoints_from_file(extracted_data_folder+current_el_filename, points, descriptors))
                    return false;
            }
            
            // Create new measurement
            measurements.emplace_back(Measurement(points, descriptors));

            // Count the number of points in the current measurement
            tot_n_points += points.size();

        }
        measurements.shrink_to_fit();

        // Close the csv file
        csv_file.close();

        // Initialize the state
        state = State(K, measurements, measurements.size(), tot_n_points);

        return true;
    }

    /*
    * This function allows to load the camera matrix formatted for PRD 
    * dataset.
    */
    bool load_PRD_camera_matrix(const std::string& filename, cv::Mat& K) {

        // Initialization 
        K = cv::Mat::eye(3,3,CV_32F);
        std::string current_line;

        // Open the file
        std::fstream file;
        file.open(filename);
        if(file.fail()) return false;

        // Ignore the first 3 lines
        std::getline(file, current_line);
        std::getline(file, current_line);
        std::getline(file, current_line);

        // Read the focal length
        std::getline(file, current_line);
        std::stringstream ss_focal_length(current_line);
        ss_focal_length >> K.at<float>(0,0) >> K.at<float>(1,1); 

        // Ignore next line
        std::getline(file, current_line);

        // Read the principal point
        std::getline(file, current_line);
        std::stringstream ss_principal_point(current_line);
        ss_principal_point >> K.at<float>(0,2) >> K.at<float>(1,2);

        // Close the file
        file.close();

        return true;
    }
    
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Functions to deal with my Synthetic Dataset
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function to load my synthetic dataset. 
    * IMPORTANT: this dataset contains the associations not done on images
    * but on 3D points ids, so we need to deal with this in the system.
    * Indeed in the vector associations it contains in position i, the list
    * of the ids for each point in the measurement i (ordered in the same
    * way of the points).
    * (This dataset is useful for test, because it contains no noise)
    */
    bool load_synthetic_dataset(const std::string& dataset_folder, State& state, \
                                std::vector<std::vector<unsigned int>>& associations) {

        // Initialization
        std::string camera_matrix_filename = dataset_folder + \
                        "camera_parameters.dat";
        std::string csv_filename = dataset_folder + "data.csv";
        std::string current_filename, current_line;
        cv::Mat K;
        std::vector<Measurement> measurements;
        std::fstream current_file;
        
        // Load the camera matrix
        if(!load_synthetic_camera_matrix(camera_matrix_filename, K))
            return false;
        
        // Open the csv file
        std::fstream csv_file;
        csv_file.open(csv_filename);
        if(csv_file.fail()) return false;

        // Load all measurements
        measurements.reserve(12);
        associations.reserve(12);
        while(std::getline(csv_file, current_line)) {
            
            // Get the current filename
            std::stringstream ss_current_line_csv_file(current_line);
            ss_current_line_csv_file >> current_filename; 
            ss_current_line_csv_file >> current_filename;
            current_filename = dataset_folder+current_filename;

            // Open the current file
            current_file.open(current_filename);
            if(current_file.fail()) return false;

            // Ignore the first 7 lines
            std::getline(current_file, current_line);
            std::getline(current_file, current_line);
            std::getline(current_file, current_line);
            std::getline(current_file, current_line);
            std::getline(current_file, current_line);
            std::getline(current_file, current_line);
            std::getline(current_file, current_line);

            // Read all the points
            std::vector<cv::KeyPoint> points;
            std::vector<unsigned int> ground_truth_ids;
            points.reserve(2000); ground_truth_ids.reserve(2000);
            while(std::getline(current_file, current_line)) {

                unsigned int gt_id;
                float x, y;                
                std::stringstream ss_current_line_points(current_line);
                
                ss_current_line_points >> gt_id >> x >> y;

                points.emplace_back(cv::KeyPoint(cv::Point2f(x,y), 1));
                ground_truth_ids.emplace_back(gt_id);
                
            }
            points.shrink_to_fit(); ground_truth_ids.shrink_to_fit();

            // Create new measurements
            // IMPORTANT: we have no descriptors in this dataset
            cv::Mat descriptors;
            measurements.emplace_back(Measurement(points, descriptors));
            associations.emplace_back(ground_truth_ids);

            // Close the current file
            current_file.close();

        }
        measurements.shrink_to_fit();
        associations.shrink_to_fit();

        // Close the csv file
        csv_file.close();

        // Initialize the state
        state = State(K, measurements, measurements.size(), 10000);

        return true;

    }


    /*
    * Function to load the K matrix for my synthetic dataset.
    */
    bool load_synthetic_camera_matrix(const std::string& filename, cv::Mat& K) {

        // Initialization
        K = cv::Mat::zeros(3,3,CV_32F);
        std::string current_line;

        // Open the file
        std::fstream file;
        file.open(filename);
        if(file.fail()) return false;

        // Ignore the first line;
        std::getline(file, current_line);

        // Load the first row
        std::getline(file, current_line);
        std::stringstream ss1(current_line);
        ss1 >> K.at<float>(0,0) >> K.at<float>(0,1) >> K.at<float>(0,2);

        // Load the second row
        std::getline(file, current_line);
        std::stringstream ss2(current_line);
        ss2 >> K.at<float>(1,0) >> K.at<float>(1,1) >> K.at<float>(1,2);

        // Load the third row
        std::getline(file, current_line);
        std::stringstream ss3(current_line);
        ss3 >> K.at<float>(2,0) >> K.at<float>(2,1) >> K.at<float>(2,2);

        // Close the file
        file.close();

        return true;

    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions to save and load general infos on files
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This function save a set of keypoints and corresponding descriptors
    * in a .yml file.
    */
    bool save_keypoints_on_file(const std::string& filename, \
                                const std::vector<cv::KeyPoint>& points, \
                                const cv::Mat& descriptors) {

        // If the file already exists, delete it
        if(std::filesystem::exists(filename))
            std::filesystem::remove(filename);

        // Open the file
        cv::FileStorage file(filename, cv::FileStorage::WRITE);

        // Save the infos
        file << "keypoints" << points;
        file << "descriptors" << descriptors;

        // Close the file
        file.release();
        
        return true;
    }
    

    /*
    * This function load a set of keypoints and corresponding descriptors
    * from a .yml file
    */
    bool load_keypoints_from_file(const std::string& filename, \
                                std::vector<cv::KeyPoint>& points, \
                                cv::Mat& descriptors) {
        
        // Open the file
        cv::FileStorage file(filename, cv::FileStorage::READ);

        // Check validity
        if(file["keypoints"].empty() || file["descriptors"].empty()) 
            return false;

        // Load the infos
        file["keypoints"] >> points;
        file["descriptors"] >> descriptors;

        // Close the file
        file.release();

        return true;
    }
} // namespace SLucAM