//
// SLucAM_dataset.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_dataset.h>
#include <SLucAM_image.h>
#include <SLucAM_geometry.h>
#include <filesystem>
#include <fstream>
#include <iostream>



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
    bool load_my_dataset(const std::string& dataset_folder, State& state, \
                        const cv::Ptr<cv::Feature2D>& detector, \
                        const bool verbose) {

        // Initialization
        std::string current_filename, current_line;
        std::string camera_matrix_filename = dataset_folder + "camera.dat";
        std::string csv_filename = dataset_folder + "data.csv";
        cv::Mat current_img, K, distorsion_coefficients;
        bool K_loaded = false;
        std::vector<Measurement> measurements;

        // Load the camera matrix
        if(!load_camera_matrix(camera_matrix_filename, K, distorsion_coefficients))
            return false;

        // Open the csv file
        std::fstream csv_file;
        csv_file.open(csv_filename);
        if(csv_file.fail()) return false;
        
        // Load all measurements
        measurements.reserve(35);
        while(std::getline(csv_file, current_line)) {

            // Get the current filename
            std::stringstream ss_current_line_csv_file(current_line);
            ss_current_line_csv_file >> current_filename; 
            ss_current_line_csv_file >> current_filename;
            current_filename = dataset_folder+current_filename;

            // Load the measurement
            if(!load_image(current_filename, current_img))
                return false;
            
            // Detect keypoints
            std::vector<cv::KeyPoint> points;
            cv::Mat descriptors;
            detector->detectAndCompute(current_img, cv::Mat(), \
                                            points, descriptors);

            // Undistort keypoints
            std::vector<cv::KeyPoint> undistorted_points;
            undistort_keypoints(points, undistorted_points, \
                                distorsion_coefficients, K);

            // Create new measurement
            measurements.emplace_back(Measurement(undistorted_points, \
                                        descriptors));

            // Memorize the name of the image
            measurements.back().setImgName(current_filename);

        }
        measurements.shrink_to_fit();

        // Initialize the state
        state = State(K, distorsion_coefficients, measurements, \
                        measurements.size(), 5000);

        if(verbose) {
            std::cout << "Loaded " << measurements.size() << " measurements" \
                << " with camera matrix:" << std::endl << K << std::endl \
                << "and distorsion parameters: " << std::endl \
                << distorsion_coefficients << std::endl;
        }
        
        return true;
            
    }


    /*
    * Load the camera matrix from the "camera.dat" file in my dataset.
    * It returns false in case of errors.
    */
    bool load_camera_matrix(const std::string& filename, cv::Mat& K, \
                            cv::Mat& distorsion_coefficients) {

        K = cv::Mat::zeros(3,3,CV_32F);
        distorsion_coefficients = cv::Mat::zeros(1,5,CV_32F);

        std::fstream camera_file;
        camera_file.open(filename);
        if(camera_file.fail()) return false;
        camera_file >> \
            K.at<float>(0,0) >> K.at<float>(0,1) >> K.at<float>(0,2) >> \
            K.at<float>(1,0) >> K.at<float>(1,1) >> K.at<float>(1,2) >> \
            K.at<float>(2,0) >> K.at<float>(2,1) >> K.at<float>(2,2) >> \
            distorsion_coefficients.at<float>(0,0) >> \
            distorsion_coefficients.at<float>(0,1) >> \
            distorsion_coefficients.at<float>(0,2) >> \
            distorsion_coefficients.at<float>(0,3) >> \
            distorsion_coefficients.at<float>(0,4);
        camera_file.close();

        return true;
    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions to deal with the TUM Dataset
// -----------------------------------------------------------------------------
namespace SLucAM {

    bool load_TUM_dataset(const std::string& dataset_folder, State& state, \
                            const cv::Ptr<cv::Feature2D>& detector, \
                            const bool verbose) {

        // Initialization
        std::string camera_matrix_filename = dataset_folder + "camera_parameters.txt";
        std::string imgs_names_filename = dataset_folder + "rgb.txt";
        std::string current_line, current_img_filename;
        cv::Mat K, distorsion_coefficients, current_img;
        std::vector<Measurement> measurements;

        // Load the camera matrix
        if(!load_TUM_camera_matrix(camera_matrix_filename, K, distorsion_coefficients))
            return false;

        // Open the file containing the ordered names of the images
        std::fstream imgs_names_file;
        imgs_names_file.open(imgs_names_filename);
        if(imgs_names_file.fail()) return false;

        // Ignore the first thre lines
        std::getline(imgs_names_file, current_line);
        std::getline(imgs_names_file, current_line);
        std::getline(imgs_names_file, current_line);

        // Load all measurements (only the first 200 for now)
        int i = 0;
        measurements.reserve(798);
        while(std::getline(imgs_names_file, current_line)) {

            if(i==200) break;
            
            // Get the current filename
            std::stringstream ss_current_line_csv_file(current_line);
            ss_current_line_csv_file >> current_img_filename; 
            ss_current_line_csv_file >> current_img_filename;
            current_img_filename = dataset_folder+current_img_filename;

            // Load the measurement (undistorted)
            if(!load_image(current_img_filename, current_img))
                return false;
            
            // Detect keypoints
            std::vector<cv::KeyPoint> points;
            cv::Mat descriptors;
            detector->detectAndCompute(current_img, cv::Mat(), \
                                            points, descriptors);
            
            // Undistort keypoints
            std::vector<cv::KeyPoint> undistorted_points;
            undistort_keypoints(points, undistorted_points, \
                                distorsion_coefficients, K);

            // Create new measurement
            measurements.emplace_back(Measurement(undistorted_points, 
                                        descriptors));

            // Memorize the name of the image
            measurements.back().setImgName(current_img_filename);

            ++i;

        }

        // Initialize the state
        state = State(K, distorsion_coefficients, measurements, \
                        measurements.size(), 50000);

        if(verbose) {
            std::cout << "Loaded " << measurements.size() << " measurements" \
                << " with camera matrix:" << std::endl << K << std::endl \
                << "and distorsion parameters: " << std::endl \
                << distorsion_coefficients << std::endl;
        }

        return true;

    }


    bool load_TUM_camera_matrix(const std::string& filename, cv::Mat& K, \
                                cv::Mat& distorsion_coefficients) {
        
        K = cv::Mat::zeros(3,3,CV_32F);
        distorsion_coefficients = cv::Mat::zeros(1,5,CV_32F);

        std::fstream camera_file;
        camera_file.open(filename);
        if(camera_file.fail()) return false;
        camera_file >> \
            K.at<float>(0,0) >> K.at<float>(0,1) >> K.at<float>(0,2) >> \
            K.at<float>(1,0) >> K.at<float>(1,1) >> K.at<float>(1,2) >> \
            K.at<float>(2,0) >> K.at<float>(2,1) >> K.at<float>(2,2) >> \
            distorsion_coefficients.at<float>(0,0) >> \
            distorsion_coefficients.at<float>(0,1) >> \
            distorsion_coefficients.at<float>(0,2) >> \
            distorsion_coefficients.at<float>(0,3) >> \
            distorsion_coefficients.at<float>(0,4);
        camera_file.close();

        return true;

    }



    bool save_TUM_results(const std::string& dataset_folder, const State& state) {

        // Initialization
        std::string results_filename = dataset_folder + "SLucAM_results.txt";
        std::string current_line;

        // Open the file where to save the results
        std::ofstream results_file;
        results_file.open(results_filename);
        if(results_file.fail()) return false;

        // Save each pose in the state (pose wrt world), in the TUM format
        // TODO: save also timestamps
        const unsigned int n_poses = state.getPoses().size();
        for(unsigned int i=0; i<n_poses; ++i) {
            const cv::Mat current_pose = invert_transformation_matrix(state.getPoses()[i]);
            cv::Mat current_quat;
            matrix_to_quaternion(current_pose.rowRange(0,3).colRange(0,3), \
                    current_quat);
            results_file << std::setprecision(4) \
                << current_pose.at<float>(0,3) << "\t" \
                << current_pose.at<float>(1,3) << "\t" \
                << current_pose.at<float>(2,3) << "\t" \
                << current_quat.at<float>(0,0) << "\t" \
                << current_quat.at<float>(1,0) << "\t" \
                << current_quat.at<float>(2,0) << "\t" \
                << current_quat.at<float>(3,0) << std::endl;
        }
        results_file.close();

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
                if(!save_keypoints_PRD(extracted_data_folder+current_el_filename, points, descriptors))
                    return false;
            } else {
                current_el_filename = current_el_filename.substr(0, current_el_filename.size()-3) \
                                            + "yml";
                if(!load_keypoints_PRD(extracted_data_folder+current_el_filename, points, descriptors))
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


     /*
    * This function save a set of keypoints and corresponding descriptors
    * in a .yml file.
    */
    bool save_keypoints_PRD(const std::string& filename, \
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
    bool load_keypoints_PRD(const std::string& filename, \
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
    * (This dataset is useful for test)
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


    /*
    * This function load the real position of the 3D points of the synthetic
    * dataset. In particular in position i of the vector we have the position
    * of the landmark with the idx i.
    */
    bool load_3dpoints_ground_truth(const std::string& filename, \
                                    std::vector<cv::Point3f>& gt_points) {
        
        // Initialization
        std::string current_line;
        unsigned int currend_idx;
        gt_points.reserve(50000);

        // Open the file
        std::fstream file;
        file.open(filename);
        if(file.fail()) return false;

        // Ignore the first line
        std::getline(file, current_line);

        // Load each line
        while(std::getline(file, current_line)) {
            float x, y, z;
            std::stringstream ss_current_line(current_line);
            ss_current_line >> currend_idx >> x >> y >> z;
            gt_points.emplace_back(cv::Point3f(x,y,z));
        }
        gt_points.shrink_to_fit();

        // Close the file
        file.close();

        return true;

    }



    /*
    * Function that, given the predicted 3D points, returns the error of the prediction
    * for the synthetic dataset.
    */
    float test_predicted_points(const std::string& dataset_folder, \
                                const std::vector<Keyframe>& keyframes, \
                                const std::vector<cv::Point3f>& predicted_points, \
                                const std::vector<std::vector<unsigned int>>& associations) {
        
        // Initialization
        std::vector<cv::Point3f> gt_points;
        const unsigned int n_keyframes = keyframes.size();
        unsigned int current_n_observations, current_meas_idx, current_gt_idx;
        float error = 0.0;
        float ratio_x, ratio_y, ratio_z, d1, d2, d3;
        std::map<unsigned int, cv::Point3f> id_to_prediction;

        // Load the ground truth
        SLucAM::load_3dpoints_ground_truth(dataset_folder+"3d_points.dat", \
                                            gt_points);
        
        // Setting up a map in which in key i we have the prediction
        // for the landmark with id i
        for(unsigned int keyframe_idx=0; keyframe_idx<n_keyframes; ++keyframe_idx) {

            // Get the current keypoint measurement idx
            current_meas_idx = keyframes[keyframe_idx].getMeasIdx();

            // Get the current list of observations point<->landmark
            const std::vector<std::pair<unsigned int, unsigned int>> current_observations = \
                    keyframes[keyframe_idx].getPointsAssociations();
            
            current_n_observations = current_observations.size();
            
            // For each observation
            for (unsigned int obs_idx=0; obs_idx<current_n_observations; ++obs_idx) {

                // Get the current observation
                const std::pair<unsigned int, unsigned int> current_obs = \
                        current_observations[obs_idx];

                // Take the idx of the ground_truth of the current point
                current_gt_idx = associations[current_meas_idx][current_obs.first];

                // If we do not have yet inserted the prediction for this landmark
                // insert it
                if(!id_to_prediction.count(current_gt_idx)) {
                    id_to_prediction[current_gt_idx] = predicted_points[current_obs.second];
                }

            }

        }

        // For each prediction compute the error
        for(const auto& idx_prediction : id_to_prediction) {

            // Take the real value
            const cv::Point3f& gt_point = gt_points[idx_prediction.first];

            // Take the prediction
            const cv::Point3f& predicted_point = idx_prediction.second;

            /* Compute ratios, avoiding zeros division
            if(predicted_point.x != 0) {
                ratio_x = gt_point.x/predicted_point.x;
            } else {
                ratio_x = gt_point.x;
            }
            if(predicted_point.y != 0) {
                ratio_y = gt_point.y/predicted_point.y;
            } else {
                ratio_y = gt_point.y;
            }
            if(predicted_point.z != 0) {
                ratio_z = gt_point.z/predicted_point.z;
            } else {
                ratio_z = gt_point.z;
            }

            // Compute the error as the distance between the three ratios
            d1 = ratio_x-ratio_y;
            d2 = ratio_x-ratio_z;
            d3 = ratio_y-ratio_z;
            error += (sqrt(d1*d1) + sqrt(d2*d2) + sqrt(d3*d3))/3.0;
            */
            
            // Compute the distance between the predicted point and the gt
            const float dx = predicted_point.x-gt_point.x;
            const float dy = predicted_point.y-gt_point.y;
            const float dz = predicted_point.z-gt_point.z;
            error += sqrt((dx*dx)+(dy*dy)+(dz*dz));
            
        }

        return error/id_to_prediction.size();
    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions to save and load general infos on files
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function that save the current state in a folder by saving all
    * the poses and landmarks, the filename of the last seen image
    * and the list of landmarks seen from the last keyframe/pose.
    */
    bool save_current_state(const std::string& folder, \
                            const State& state) {
        
        // Initialization
        const Keyframe& last_keyframe = state.getKeyframes().back();
        const Measurement& last_measure = state.getMeasurements()[last_keyframe.getMeasIdx()];
        const std::string& last_image = last_measure.getImgName();
        const std::vector<cv::KeyPoint>& last_measure_points = last_measure.getPoints();
        const std::string last_image_filename = folder + "SLucAM_image_name.dat";

        // Save the filename of the last seen image
        std::ofstream f_img;
        f_img.open(last_image_filename);
        if(f_img.fail()) return false;
        f_img << last_image;
        f_img.close();
        
        // Save all the poses
        if(!save_poses(folder, state.getPoses())) return false;

        // Save all landmarks
        if(!save_landmarks(folder, state.getLandmarks())) return false;

        // Save the edges last_keyframe <-> landmarks
        if(!save_edges(folder, last_keyframe)) return false;

        // Save the points on the last image
        if(!save_keypoints(folder, last_measure_points)) return false;

        return true;

    }


    /*
    * Function that save in a file all the predicted poses with the 
    * format: tx ty tz r11 r12 r13 ... r33 (where rij is the element
    * in the position <i,j> of the rotation part of the pose)
    */  
    bool save_poses(const std::string& folder, \
                    const std::vector<cv::Mat>& poses) {
        
        // Initialization
        const std::string filename = folder + "SLucAM_poses.dat";
        const unsigned int n_poses = poses.size();

        // Open the file
        std::ofstream f;
        f.open(filename);
        if(f.fail()) return false;

        // Write the header
        f << "PREDICTED POSES" << std::endl;
        f << "tx\tty\ttz\tr11\tr12\tr13\tr21\tr22\tr23\tr31\tr32\tr33\t";

        // Write all the landmarks
        for(unsigned int i=0; i<n_poses; ++i) {
            const cv::Mat& p = poses[i];
            f << std::endl \
                << p.at<float>(0,3) << "\t" << p.at<float>(1,3) << "\t" << p.at<float>(2,3) << "\t" \
                << p.at<float>(0,0) << "\t" << p.at<float>(0,1) << "\t" << p.at<float>(0,2) << "\t" \
                << p.at<float>(1,0) << "\t" << p.at<float>(1,1) << "\t" << p.at<float>(1,2) << "\t" \
                << p.at<float>(2,0) << "\t" << p.at<float>(2,1) << "\t" << p.at<float>(2,2);
                
        }

        // Close the file
        f.close();

        return true;

    }


    /*
    * Function that save in a file all the predicted 3D points.
    */  
    bool save_landmarks(const std::string& folder, \
                        const std::vector<cv::Point3f>& landmarks) {
        
        // Initialization
        const std::string filename = folder + "SLucAM_landmarks.dat";
        unsigned int n_landmars = landmarks.size();

        // Open the file
        std::ofstream f;
        f.open(filename);
        if(f.fail()) return false;

        // Write the header
        f << "3D PREDICTED POINTS" << std::endl;
        f << "x\ty\tz";

        // Write all the landmarks
        for(unsigned int i=0; i<n_landmars; ++i) {
            const cv::Point3f& l = landmarks[i];
            f << std::endl << l.x << "\t" << l.y << "\t" << l.z;
        }

        // Close the file
        f.close();

        return true;

    }


    /*
    * Function that saves in a file the list of landmarks indices seen
    * from the given keyframe.
    */
    bool save_edges(const std::string& folder, \
                    const Keyframe& keyframe) {
        
        // Initialization
        const std::string filename = folder + "SLucAM_edges.dat";
        const std::vector<std::pair<unsigned int, unsigned int>>& points_associations = \
                keyframe.getPointsAssociations();
        const unsigned int n_edges = points_associations.size();

        // Open the file
        std::ofstream f;
        f.open(filename);
        if(f.fail()) return false;

        // Write the header
        f << "LIST OF SEEN LANDMARKS" << std::endl;
        f << "landmark_idx";

        // Write all the landmarks
        for(unsigned int i=0; i<n_edges; ++i) {
            f << std::endl << points_associations[i].second;
        }

        // Close the file
        f.close();

        return true;

    }


    /*
    * Function that saves in a file the list of points positions seen
    * on a measurement.
    */
    bool save_keypoints(const std::string& folder, \
                        const std::vector<cv::KeyPoint>& points) {
        
        // Initialization
        const std::string filename = folder + "SLucAM_img_points.dat";
        unsigned int n_points = points.size();

        // Open the file
        std::ofstream f;
        f.open(filename);
        if(f.fail()) return false;

        // Write the header
        f << "2D POINTS ON IMAGE" << std::endl;
        f << "x\ty";

        // Write all the landmarks
        for(unsigned int i=0; i<n_points; ++i) {
            const cv::KeyPoint& p = points[i];
            f << std::endl << p.pt.x << "\t" << p.pt.y;
        }

        // Close the file
        f.close();

        return true;

    }

} // namespace SLucAM