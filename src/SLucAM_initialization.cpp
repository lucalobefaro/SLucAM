//
// SLucAM_initialization.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_initialization.h>
#include <SLucAM_geometry.h>
#include <numeric>
#include <algorithm>
#include <iostream>



// -----------------------------------------------------------------------------
// Implementation of main initialization functions
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function that, given two measurements, try to perform initialization, by
    * computing the Foundamental Matrix and the Homography between the two 
    * measurements. If a valid (not rotation only) transform between the two
    * is found, it computes the pose of the second pose w.r.t. the first one
    * and triangulates the points in common.
    * Inputs:
    *   meas1/meas2: the measurements to use to perform initialization
    *   matcher: tool to use to match the points between the measurements
    *   K: camera matrix of the camera used to take the measurements
    *   predicted_pose: the pose of the meas2 (world wrt camera)
    *   triangulated_points: the initial map
    *   parallax_threshold: if the parallax between the two measurement
    *       is under this threshold we consider the initialization 
    *       invalid (because we cannot triangulate a good initial map)
    *   verbose
    * Outputs:
    *   false if the measures are not good to perform initialization
    */
    bool initialize(const Measurement& meas1, \
                    const Measurement& meas2, \
                    Matcher& matcher, \
                    const cv::Mat& K, cv::Mat& predicted_pose, \
                    std::vector<cv::DMatch>& matches, \
                    std::vector<unsigned int>& matches_filter, \
                    std::vector<cv::Point3f>& triangulated_points, \
                    const unsigned int n_iters_ransac, \
                    const float& parallax_threshold, \
                    const bool verbose) {
        
        // Initialization
        matches.clear();
        matches_filter.clear();
        triangulated_points.clear();
        const std::vector<cv::KeyPoint>& p_img1 = meas1.getPoints();
        const std::vector<cv::KeyPoint>& p_img2 = meas2.getPoints();
        const float F_inliers_threshold = 3.84;
        const float H_inliers_threshold = 5.99;
        const float score_dump = std::max(F_inliers_threshold, H_inliers_threshold);
        cv::Mat first_pose = cv::Mat::eye(4,4,CV_32F);

        // Match the two measurements
        matcher.match_measurements(meas1, meas2, matches);

        // Compute the Fundamental matrix and the score, in the meanwhile fill
        // the matches_filter vector with the list of inliers in matches
        cv::Mat F;
        const float F_score = compute_fundamental(p_img1, p_img2, \
                                                matches, matches_filter, F, \
                                                n_iters_ransac, \
                                                F_inliers_threshold, \
                                                score_dump);
                
        // Compute the score obtained from the Homography
        const float H_score = compute_homography(p_img1, p_img2, \
                                                matches, n_iters_ransac, \
                                                H_inliers_threshold, \
                                                score_dump);

        // According to [Raúl Mur-Artal, J. M. M. Montiel and Juan D. Tardós. 
        // ORB-SLAM: A Versatile and Accurate Monocular SLAM System. 
        // IEEE Transactions on Robotics, vol. 31, no. 5, pp. 1147-1163, 2015. 
        // (2015 IEEE Transactions on Robotics Best Paper Award).]
        // we trust of F only if RH <= 0.45, where RH = (H_score)/(H_score+F_score)
        float RH = (H_score)/(H_score+F_score);
        if(RH > 0.45) {
            if(verbose) {
                std::cout << "-> ROTATION ONLY" << std::endl;
            }
            return false;
        }

        // Extract the best possible pose from the Fundamental Matrix (if possible)
        // and update the inliers filter (matches_filter) (it also triangulate initial map)
        if(!initialize_map(p_img1, p_img2, matches, matches_filter, \
                            F, K, predicted_pose, triangulated_points)) {
            if(verbose) {
                std::cout << "-> CANNOT COMPUTE A GOOD ENOUGH POSE" << std::endl;
            }
            return false;
        }
                
        // Compute the parallax between the two poses  
        std::vector<unsigned int> common_landmarks(triangulated_points.size());
        std::iota(common_landmarks.begin(), common_landmarks.end(), 0);
        float parallax = computeParallax(first_pose, predicted_pose, \
                                        triangulated_points, common_landmarks);

        // If we do not have enough parallax, refuse initialization
        if(parallax <= parallax_threshold) {
            if(verbose) {
                std::cout << "-> NOT ENOUGH PARALLAX " << "(" << parallax << ")" << std::endl;
            }
            //return false;
        }

        // Initialization performed
        if(verbose) {
            std::cout << "Initialization performed with " << matches_filter.size() << " inliers." << std::endl;
        }        
        return true;

    }



    /*
    * This function allows us to obtain the rotation matrix R and the
    * translation vector t between two images, for which we have the 
    * Fundamental Matrix F, composed in a transformation matrix X = [R|t]
    * and the initial triangulated points
    * Inputs:
    *   p_img1/p_img2: points to use in order to understand wich X computed
    *                   is "better"
    *   matches: all matches between p_img1 and p_img2, with outliers
    *   matches_filter: we will consider only those points contained in 
    *           the matches vector, for wich we have indices in this vector
    *           We also update this vector with only inliers detected from
    *           the best solution.
    *   F: Fundamental Matrix from which to extract X
    *   K: camera matrix of the two cameras
    *   X: output Transformation Matrix extracted from E [R|t]
    *   triangulated_points: the initial map
    */
    bool initialize_map(const std::vector<cv::KeyPoint>& p_img1, \
                        const std::vector<cv::KeyPoint>& p_img2, \
                        const std::vector<cv::DMatch>& matches, \
                        const std::vector<unsigned int>& matches_filter, \
                        const cv::Mat& F, const cv::Mat& K, \
                        cv::Mat& X, \
                        std::vector<cv::Point3f>& triangulated_points) {

        // Initialization
        cv::Mat best_R, best_t;
        cv::Mat W_mat = cv::Mat::zeros(3,3,CV_32F);
        W_mat.at<float>(0,1) = -1;
        W_mat.at<float>(1,0) = 1;
        W_mat.at<float>(2,2) = 1;
        unsigned int best_score = 0;
        unsigned int second_best_score = 0;
        unsigned int current_score = 0;
        std::vector<cv::Point3f> current_triangulated_points;

        // Compute the Essential Matrix and decompose it
        cv::Mat u,w,vt;
        cv::Mat E = K.t()*F*K;
        cv::SVD::compute(E,w,u,vt,cv::SVD::FULL_UV);

        // Extract the R matrix (2 solutions)
        cv::Mat R1 = u*W_mat*vt;
        if(cv::determinant(R1) < 0) {   // right handed condition
            R1 = -R1;
        }
        cv::Mat R2 = u*W_mat.t()*vt;
        if(cv::determinant(R2) < 0) {    // right handed condition
            R2 = -R2;
        }

        // Extract the t vector and "normalize" it (2 solutions)
        cv::Mat t1;
        u.col(2).copyTo(t1);
        t1=t1/cv::norm(t1);
        cv::Mat t2 = -t1;

        // Evaluate all solutions
        const std::vector<cv::Mat> rotations = {R1, R1, R2, R2};
        const std::vector<cv::Mat> translations = {t1, t2, t1, t2};
        for(unsigned int i=0; i<4; ++i) {
            
            // Evaluate the current solution
            current_triangulated_points.clear();
            current_score = compute_transformation_inliers(p_img1, p_img2, matches, matches_filter, \
                                                            rotations[i], translations[i], K, \
                                                            current_triangulated_points);
            
            // DEBUG:
            std::cout << std::endl << "SCORE " << i << ": " << current_score << std::endl;
            std::cout << rotations[i] << std::endl;
            std::cout << translations[i] << std::endl << std::endl;

            // If it is the best one, save it
            if(current_score > best_score) {
                second_best_score = best_score;
                best_score = current_score;
                best_R = rotations[i].clone();
                best_t = translations[i].clone();
                triangulated_points.swap(current_triangulated_points);
            } 

        }

        // Check if we have enough inliers and a clear winner
        if((best_score < 0.6*matches_filter.size()) \
            || second_best_score > 0.8*best_score) {
            return false;
        }

        // Save best solution
        X = cv::Mat::eye(4,4,CV_32F);
        best_R.copyTo(X.rowRange(0,3).colRange(0,3));
        best_t.copyTo(X.rowRange(0,3).col(3));

        std::cout << "BEST X: " << std::endl << X \
                << std::endl << std::endl;
                
        // Pose computed correctly
        return true;
    }


    /*
    * This function evaluates a solution (R and t) extracted from the Essential
    * matrix by computing the number of inliers that such pose has. It also
    * triangulated all the valid matches.
    * Inputs:
    *   p_img1/p_im2: measured points in the two cameras
    *   matches: matches between p_img1 and p_img2 with outliers
    *   matches_filter: list of valid matches ids
    *   matches_inliers: list of matches ids that are considered inliers
    *   R/t: pose to evaluate
    *   K: camera matrix of the two cameras
    *   triangulated_points
    * Outputs:
    *   #inliers (size of the matches_inliers vector)
    */
    unsigned int compute_transformation_inliers(const std::vector<cv::KeyPoint>& p_img1, \
                                                const std::vector<cv::KeyPoint>& p_img2, \
                                                const std::vector<cv::DMatch>& matches, \
                                                const std::vector<unsigned int>& matches_filter, \
                                                const cv::Mat& R, const cv::Mat& t, \
                                                const cv::Mat& K, \
                                                std::vector<cv::Point3f>& triangulated_points) {
        
        // Build the pose
        cv::Mat X = cv::Mat::eye(4,4,CV_32F);
        R.copyTo(X.rowRange(0,3).colRange(0,3));
        t.copyTo(X.rowRange(0,3).col(3));

        // TODO: add some other check if needed

        // Return the number of valid triangulation
        return triangulate_points(p_img1, p_img2, matches, matches_filter, \
                                    cv::Mat::eye(4,4,CV_32F), X, K, triangulated_points);
        
    }


} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions to compute Essential and Homography
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This function, given two images (represented by two sets of points)
    * and matches between them, computes the Fundamental matrix filtering
    * inliers, and compute a score (it uses the RANSAC mechanism)
    * Inputs:
    *   p_img1/p_img2: set of points for the two images
    *   K: camera matrix of the two images
    *   matches: a set of correspondances between the two sets of points
    *   matches_filter: it will contain a list of indices of inliers for matches
    *   F: the predicted fundamental matrix
    * Outputs:
    *   F_score: the score of the F matrix
    */
    const float compute_fundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                        const std::vector<cv::KeyPoint>& p_img2, \
                                        const std::vector<cv::DMatch>& matches, \
                                        std::vector<unsigned int>& matches_filter, \
                                        cv::Mat& F, \
                                        const unsigned int n_iters_ransac, \
                                        const float& inliers_threshold, \
                                        const float& score_dump) {
        
        // Initialization
        matches_filter.clear();
        const unsigned int n_matches = matches.size();
        float current_score = 0;
        float best_score = 0;
        std::vector<unsigned int> current_matches_filter, best_matches_filter;

        // Normalize points
        std::vector<cv::KeyPoint> normalized_p_img1, normalized_p_img2;
        cv::Mat T1, T2;
        normalize_points(p_img1, normalized_p_img1, T1);
        normalize_points(p_img2, normalized_p_img2, T2);

        // Generate random sets of matches indices, one for iteration of RANSAC
        std::vector<std::vector<unsigned int>> random_idxs(n_iters_ransac, \
                                                    std::vector<unsigned int>(8));
        std::vector<unsigned int> indices(n_matches);
        std::iota(indices.begin(), indices.end(), 0);
        srand(time(NULL));
        for(unsigned int i=0; i<n_iters_ransac; ++i) {
            std::random_shuffle(indices.begin(), indices.end(), [](int i) {return rand() % i;});
            random_idxs[i].assign(indices.begin(), indices.begin()+8);
        }

        // Perform RANSAC for given iterations
        for(unsigned int iter=0; iter<n_iters_ransac; ++iter) {

            // Compute the Foundamental Matrix on the current minimal set
            estimate_foundamental(normalized_p_img1, normalized_p_img2, matches, \
                                    random_idxs[iter], F);

            // Denormalize F
            F = T2.t()*F*T1;
            
            // Compute the score with the current F
            current_score = evaluate_fundamental(p_img1, p_img2, \
                                                matches, current_matches_filter, F, \
                                                inliers_threshold, score_dump);

            // If it is the best encountered so far, save it as the best one
            if(current_score > best_score) {
                best_score = current_score;
                best_matches_filter.swap(current_matches_filter);
            }
        }

        // Compute the best version of F using only inliers of the best iteration
        // (contained in best_matches_filter)
        estimate_foundamental(normalized_p_img1, normalized_p_img2, matches, \
                                best_matches_filter, F);
        F = T2.t()*F*T1;

        // Return the score and fill matches_filter during evaluation
        return evaluate_fundamental(p_img1, p_img2, \
                                    matches, matches_filter, F, \
                                    inliers_threshold, score_dump);

    }



    /*
    * Function that, given two images (represented by two sets of points)
    * and matches between them, computes the Homography and returns the 
    * score (it uses the RANSAC mechanism).
    * Inputs:
    *   p_img1/p_img2: set of points for the two images
    *   K: camera matrix of the two images
    *   matches: a set of correspondances between the two sets of points
    *   inliers_threshold: value to determine inliers
    *  Outputs:
    *   H_score
    */
    const float compute_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const unsigned int n_iters_ransac, \
                                    const float& inliers_threshold, \
                                    const float& score_dump) {
        
        // Initialization
        const unsigned int n_matches = matches.size();
        float current_score = 0;
        float best_score = 0;
        cv::Mat H;
        std::vector<unsigned int> current_matches_filter, best_matches_filter;

        // Normalize points
        std::vector<cv::KeyPoint> normalize_p_img1, normalize_p_img2;
        cv::Mat T1, T2;
        normalize_points(p_img1, normalize_p_img1, T1);
        normalize_points(p_img2, normalize_p_img2, T2);

        // Generate random sets of matches indices, one for iteration of RANSAC
        std::vector<std::vector<unsigned int>> random_idxs(n_iters_ransac, \
                                                    std::vector<unsigned int>(8));
        std::vector<unsigned int> indices(n_matches);
        std::iota(indices.begin(), indices.end(), 0);
        srand(time(NULL));
        for(unsigned int i=0; i<n_iters_ransac; ++i) {
            std::random_shuffle(indices.begin(), indices.end(), [](int i) {return rand() % i;});
            random_idxs[i].assign(indices.begin(), indices.begin()+8);
        }

        // Perform RANSAC for given iterations
        for(unsigned int iter=0; iter<n_iters_ransac; ++iter) {

            // Compute the Homography Matrix on the current minimal set
            estimate_homography(normalize_p_img1, normalize_p_img2, matches, \
                                    random_idxs[iter], H);

            // Denormalize H
            H = T2.inv()*H*T1;
            
            // Compute the score with the current H
            current_score = evaluate_homography(p_img1, p_img2, \
                                                matches, current_matches_filter, H, \
                                                inliers_threshold, score_dump);

            // If it is the best encountered so far, save it as the best one
            if(current_score > best_score) {
                best_score = current_score;
                best_matches_filter.swap(current_matches_filter);
            }
        }

        // Compute the best version of H using only inliers of the best iteration
        // (contained in best_matches_filter)
        estimate_homography(normalize_p_img1, normalize_p_img2, matches, \
                                best_matches_filter, H);
        H = T2.inv()*H*T1;

        // Return the score of the best H
        return evaluate_homography(p_img1, p_img2, \
                                    matches, best_matches_filter, H, \
                                    inliers_threshold, score_dump);
    }
    

    
    float evaluate_fundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                std::vector<unsigned int>& matches_filter, \
                                const cv::Mat& F, \
                                const float& inliers_threshold, \
                                const float& score_dump) {
        
        // Initialization
        unsigned int n_matches = matches.size();
        float d1, d2, Fp1, Fp2, Fp3;
        float score = 0.0;
        matches_filter.clear();

        // Some reference to save time
        const float& F11 = F.at<float>(0,0);
        const float& F12 = F.at<float>(0,1);
        const float& F13 = F.at<float>(0,2);
        const float& F21 = F.at<float>(1,0);
        const float& F22 = F.at<float>(1,1);
        const float& F23 = F.at<float>(1,2);
        const float& F31 = F.at<float>(2,0);
        const float& F32 = F.at<float>(2,1);
        const float& F33 = F.at<float>(2,2);

        // For each correspondance
        matches_filter.reserve(n_matches);
        for(unsigned int i=0; i<n_matches; ++i) {

            float is_inlier = true;

            // Take the current points
            const float& p1_x = p_img1[matches[i].queryIdx].pt.x;
            const float& p1_y = p_img1[matches[i].queryIdx].pt.y;
            const float& p2_x = p_img2[matches[i].trainIdx].pt.x;
            const float& p2_y = p_img2[matches[i].trainIdx].pt.y;

            // Compute the square distance between point 2 and point 1 
            // projected in the image 2
            Fp1 = F11*p1_x + F12*p1_y + F13;
            Fp2 = F21*p1_x + F22*p1_y + F23;
            Fp3 = F31*p1_x + F32*p1_y + F33;
            d1 = pow(Fp1*p2_x+Fp2*p2_y+Fp3, 2)/(pow(Fp1, 2) + pow(Fp2, 2));

            // If the distance of this reprojection is over the threshold
            // then discard it as outlier, otherwise compute the score
            if(d1 > inliers_threshold) {
                is_inlier = false;
            } else {
                score += score_dump-d1;
            }

            // Compute the square distance between point 1 and point 2 
            // projected in the image 1
            Fp1 = F11*p2_x + F21*p2_y + F31;
            Fp2 = F12*p2_x + F22*p2_y + F32;
            Fp3 = F13*p2_x + F23*p2_y + F33;
            d2 = pow(Fp1*p1_x+Fp2*p1_y+Fp3, 2)/(pow(Fp1, 2) + pow(Fp2, 2));

            // If the distance of this reprojection is over the threshold
            // then discard it as outlier, otherwise compute the score
            if(d2 > inliers_threshold) {
                is_inlier = false;
            } else {
                score += score_dump-d2;
            }

            // If the current match is evaluated as inlier save it
            if(is_inlier) {
                matches_filter.emplace_back(i);
            }

        }
        matches_filter.shrink_to_fit();

        // Return the score
        return score;
    }



    float evaluate_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                        const std::vector<cv::KeyPoint>& p_img2, \
                                        const std::vector<cv::DMatch>& matches, \
                                        std::vector<unsigned int>& matches_filter, \
                                        const cv::Mat& H, \
                                        const float& inliers_threshold, \
                                        const float& score_dump) {
        
        // Initialization
        matches_filter.clear();
        unsigned int n_matches = matches.size();
        float d1, d2, Hp1, Hp2, Hp3;
        float score = 0.0;

        // Precompute inverse of H to save time
        cv::Mat inv_H = H.inv();
        const float& inv_H11 = inv_H.at<float>(0,0);
        const float& inv_H12 = inv_H.at<float>(0,1);
        const float& inv_H13 = inv_H.at<float>(0,2);
        const float& inv_H21 = inv_H.at<float>(1,0);
        const float& inv_H22 = inv_H.at<float>(1,1);
        const float& inv_H23 = inv_H.at<float>(1,2);
        const float& inv_H31 = inv_H.at<float>(2,0);
        const float& inv_H32 = inv_H.at<float>(2,1);
        const float& inv_H33 = inv_H.at<float>(2,2);

        // Some other reference to save time
        const float& H11 = H.at<float>(0,0);
        const float& H12 = H.at<float>(0,1);
        const float& H13 = H.at<float>(0,2);
        const float& H21 = H.at<float>(1,0);
        const float& H22 = H.at<float>(1,1);
        const float& H23 = H.at<float>(1,2);
        const float& H31 = H.at<float>(2,0);
        const float& H32 = H.at<float>(2,1);
        const float& H33 = H.at<float>(2,2);

        // For each correspondance
        matches_filter.reserve(n_matches);
        for(unsigned int i=0; i<n_matches; ++i) {

            float is_inlier = true;

            // Take the current points
            const float& p1_x = p_img1[matches[i].queryIdx].pt.x;
            const float& p1_y = p_img1[matches[i].queryIdx].pt.y;
            const float& p2_x = p_img2[matches[i].trainIdx].pt.x;
            const float& p2_y = p_img2[matches[i].trainIdx].pt.y;

            // Compute the square distance d1 = d(Hx1, x2)
            Hp3 = H31*p1_x + H32*p1_y + H33;
            Hp1 = (H11*p1_x + H12*p1_y + H13)/Hp3;
            Hp2 = (H21*p1_x + H22*p1_y + H23)/Hp3;
            d1 = pow(p2_x-Hp1, 2) + pow(p2_y-Hp2, 2);

            // If the distance of this reprojection is over the threshold
            // then discard it as outlier, otherwise compute the score
            if(d1 > inliers_threshold) {
                is_inlier = false;
            } else {
                score += score_dump-d1;
            }

            // Compute the square distance d2 = d(inv(H)x2, x1)
            Hp3 = inv_H31*p2_x + inv_H32*p2_y + inv_H33;
            Hp1 = (inv_H11*p2_x + inv_H12*p2_y + inv_H13)/Hp3;
            Hp2 = (inv_H21*p2_x + inv_H22*p2_y + inv_H23)/Hp3;
            d2 = pow(p1_x-Hp1, 2) + pow(p1_y-Hp2, 2);

            // If the distance of this reprojection is over the threshold
            // then discard it as outlier, otherwise compute the score
            if(d2 > inliers_threshold) {
                is_inlier = false;
            } else {
                score += score_dump-d2;
            }

            // If the current match is evaluated as inlier save it
            if(is_inlier) {
                matches_filter.emplace_back(i);
            }

        }
        matches_filter.shrink_to_fit();

        // Return the score
        return score;
    }

} // namespace SLucAM