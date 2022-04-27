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
#include <opencv2/calib3d.hpp>



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
                    const float& parallax_threshold, \
                    const bool verbose) {
        
        // Initialization
        matches.clear();
        matches_filter.clear();
        triangulated_points.clear();
        const std::vector<cv::KeyPoint>& p_img1 = meas1.getPoints();
        const std::vector<cv::KeyPoint>& p_img2 = meas2.getPoints();
        const float F_inliers_threshold = 4;
        const float H_inliers_threshold = 6;
        cv::Mat first_pose = cv::Mat::eye(4,4,CV_32F);

        // Match the two measurements
        matcher.match_measurements(meas1, meas2, matches);

        // Compute the Essential matrix and the 
        // score obtained from the corresponding Fundamental Matrix
        cv::Mat E;
        const float F_score = compute_essential(p_img1, p_img2, K, \
                                                matches, matches_filter, E, \
                                                F_inliers_threshold);

        // Compute the score obtained from the Homography
        const float H_score = compute_homography(p_img1, p_img2, K, \
                                                    matches, H_inliers_threshold);

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

        // Extract the best possible pose from the Essential Matrix (if possible)
        // and update the inliers filter (matches_filter)
        if(!extract_X_from_E(p_img1, p_img2, matches, matches_filter, \
                                E, K, predicted_pose)) {
            if(verbose) {
                std::cout << "-> CANNOT COMPUTE A GOOD ENOUGH POSE" << std::endl;
            }
            return false;
        }
        
        // Triangulate the initial map
        linear_triangulation(p_img1, p_img2, matches, matches_filter, \
                            first_pose, predicted_pose, K, \
                            triangulated_points);
        //triangulate_points(p_img1, p_img2, matches, matches_filter, \
                            first_pose, predicted_pose, K, \
                            triangulated_points);
                
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
            return false;
        }

        // Initialization performed
        if(verbose) {
            std::cout << "Initialization performed with " << matches_filter.size() << " inliers." << std::endl;
        }        
        return true;

    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions to compute Essential and Homography
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This function, given two images (represented by two sets of points)
    * and matches between them, computes the Essential matrix filtering
    * inliers, and compute a score for the corresponding Fundamental Matrix
    * Inputs:
    *   p_img1/p_img2: set of points for the two images
    *   K: camera matrix of the two images
    *   matches: a set of correspondances between the two sets of points
    *   matches_filter: it will contain a list of indices of inliers for matches
    *   E: the predicted essential matrix
    * Outputs:
    *   F_score: the score of the F matrix
    */
    const float compute_essential(const std::vector<cv::KeyPoint>& p_img1, \
                                        const std::vector<cv::KeyPoint>& p_img2, \
                                        const cv::Mat K, \
                                        const std::vector<cv::DMatch>& matches, \
                                        std::vector<unsigned int>& matches_filter, \
                                        cv::Mat& E, \
                                        const float& inliers_threshold) {
        
        // Initialization
        matches_filter.clear();
        const unsigned int n_matches = matches.size();
        const cv::Mat K_inv = K.inv();
        
        // Compute the list of matched points, ordered
        std::vector<cv::Point2f> matched_p_img1; matched_p_img1.reserve(n_matches);
        std::vector<cv::Point2f> matched_p_img2; matched_p_img2.reserve(n_matches);
        for(unsigned int i=0; i<n_matches; ++i) {
            matched_p_img1.emplace_back(p_img1[matches[i].queryIdx].pt.x,
                                        p_img1[matches[i].queryIdx].pt.y);
            matched_p_img2.emplace_back(p_img2[matches[i].trainIdx].pt.x,
                                        p_img2[matches[i].trainIdx].pt.y);
        }
        matched_p_img1.shrink_to_fit();
        matched_p_img2.shrink_to_fit();

        // Compute the Essential matrix and inliers
        cv::Mat inliers_mask, R, t;
        E = cv::findEssentialMat(matched_p_img1, matched_p_img2, K, \
                                cv::RANSAC, 0.999, inliers_threshold, \
                                inliers_mask);
        E.convertTo(E, CV_32F);

        // Fill the filter of inliers
        matches_filter.reserve(n_matches);
        for(unsigned int i=0; i<n_matches; ++i) {
            if(inliers_mask.at<unsigned char>(i) == 1) {
                matches_filter.emplace_back(i);
            }
        }

        // Compute the score of the corresponding Fundamental Matrix
        cv::Mat F = K_inv.t()*E*K_inv;
        return evaluate_fundamental(p_img1, p_img2, matches, F, inliers_threshold);
    }



    /*
    * Function that, given two images (represented by two sets of points)
    * and matches between them, computes the Homography and returns the 
    * score.
    * Inputs:
    *   p_img1/p_img2: set of points for the two images
    *   K: camera matrix of the two images
    *   matches: a set of correspondances between the two sets of points
    *  Outputs:
    *   H_score
    */
    const float compute_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const cv::Mat K, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const float& inliers_threshold) {
        
        // Initialization
        const unsigned int n_matches = matches.size();

        // Compute the list of matched points, ordered
        std::vector<cv::Point2f> matched_p_img1; matched_p_img1.reserve(n_matches);
        std::vector<cv::Point2f> matched_p_img2; matched_p_img2.reserve(n_matches);
        for(unsigned int i=0; i<n_matches; ++i) {
            matched_p_img1.emplace_back(p_img1[matches[i].queryIdx].pt.x,
                                        p_img1[matches[i].queryIdx].pt.y);
            matched_p_img2.emplace_back(p_img2[matches[i].trainIdx].pt.x,
                                        p_img2[matches[i].trainIdx].pt.y);
        }
        matched_p_img1.shrink_to_fit();
        matched_p_img2.shrink_to_fit();

        // Compute the Homography
        cv::Mat H = cv::findHomography(matched_p_img1, matched_p_img2, \
                                        cv::RANSAC, inliers_threshold);
        H.convertTo(H, CV_32F);

        // Compute the score
        return evaluate_homography(p_img1, p_img2, matches, H, inliers_threshold);
    }
    
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions that evaluates the "goodness" of a matrix F or H
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    float evaluate_fundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const cv::Mat& F, \
                                const float& inliers_threshold) {
        
        // Initialization
        unsigned int n_matches = matches.size();
        float d1, d2, Fp1, Fp2, Fp3;
        float score = 0.0;

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

            // If the distance of this reprojection is under the threshold
            // increment the score
            if(d1 <= inliers_threshold) {
                score += 5.991-d1;
            }

            // Compute the square distance between point 1 and point 2 
            // projected in the image 1
            Fp1 = F11*p2_x + F21*p2_y + F31;
            Fp2 = F12*p2_x + F22*p2_y + F32;
            Fp3 = F13*p2_x + F23*p2_y + F33;
            d2 = pow(Fp1*p1_x+Fp2*p1_y+Fp3, 2)/(pow(Fp1, 2) + pow(Fp2, 2));

            // If the distance of this reprojection is under the threshold
            // increment the score
            if(d2 <= inliers_threshold) {
                score += 5.991-d2;
            }

        }

        // Return the score
        return score;
    }



    float evaluate_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                        const std::vector<cv::KeyPoint>& p_img2, \
                                        const std::vector<cv::DMatch>& matches, \
                                        const cv::Mat& H, \
                                        const float& inliers_threshold) {
        
        // Initialization
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

            // If the distance of this reprojection is under the threshold
            // increment the score
            if(d1 <= inliers_threshold) {
                score += 5.991-d1;
            }

            // Compute the square distance d2 = d(inv(H)x2, x1)
            Hp3 = inv_H31*p2_x + inv_H32*p2_y + inv_H33;
            Hp1 = (inv_H11*p2_x + inv_H12*p2_y + inv_H13)/Hp3;
            Hp2 = (inv_H21*p2_x + inv_H22*p2_y + inv_H23)/Hp3;
            d2 = pow(p1_x-Hp1, 2) + pow(p1_y-Hp2, 2);

            // If the distance of this reprojection is under the threshold
            // increment the score
            if(d2 <= inliers_threshold) {
                score += 5.991-d2;
            }

        }

        // Return the score
        return score;
    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions to compute the pose from the Essential Matrix
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This function allows us to obtain the rotation matrix R and the
    * translation vector t between two images, for which we have the 
    * Essential Matrix E, composed in a transformation matrix X = [R|t].
    * Inputs:
    *   p_img1/p_img2: points to use in order to understand wich X computed
    *                   is "better"
    *   matches: all matches between p_img1 and p_img2, with outliers
    *   matches_filter: we will consider only those points contained in 
    *           the matches vector, for wich we have indices in this vector
    *           We also update this vector with only inliers detected from
    *           the best solution.
    *   E: Essential Matrix from which to extract X
    *   K: camera matrix of the two cameras
    *   X: output Transformation Matrix extracted from E [R|t]
    */
    bool extract_X_from_E(const std::vector<cv::KeyPoint>& p_img1, \
                            const std::vector<cv::KeyPoint>& p_img2, \
                            const std::vector<cv::DMatch>& matches, \
                            std::vector<unsigned int>& matches_filter, \
                            const cv::Mat& E, const cv::Mat& K, \
                            cv::Mat& X) {

        // Initialization
        cv::Mat best_R, best_t;
        cv::Mat W_mat = cv::Mat::zeros(3,3,CV_32F);
        W_mat.at<float>(0,1) = -1;
        W_mat.at<float>(1,0) = 1;
        W_mat.at<float>(2,2) = 1;
        unsigned int best_score = 0;
        unsigned int second_best_score = 0;
        unsigned int current_score = 0;
        std::vector<unsigned int> current_matches_filter, best_matches_filter;

        // Decompose the Essential Matrix
        cv::Mat u,w,vt;
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
            current_score = compute_transformation_inliers(p_img1, p_img2, matches, matches_filter, \
                                                            current_matches_filter, \
                                                            rotations[i], translations[i], K);
            std::cout << std::endl << "SCORE " << i << " " << current_score << std::endl;
            std::cout << R1 << std::endl;
            std::cout << t1 << std::endl << std::endl;
            if(current_score > best_score) {
                second_best_score = best_score;
                best_score = current_score;
                best_R = rotations[i].clone();
                best_t = translations[i].clone();
                best_matches_filter.swap(current_matches_filter);
            } 
        }

        // Check if we have enough inliers and a clear winner
        if((best_score < 0.8*matches.size()) \
            || second_best_score > 0.7*best_score) {
            return false;
        }

        // Scale the t vector to avoid too large translations
        best_t /= 10.0;

        // Save best solution
        X = cv::Mat::eye(4,4,CV_32F);
        best_R.copyTo(X.rowRange(0,3).colRange(0,3));
        best_t.copyTo(X.rowRange(0,3).col(3));
        matches_filter.swap(best_matches_filter);

        std::cout << "BEST X: " << std::endl << X \
                << std::endl << std::endl;
                
        // Pose computed correctly
        return true;
    }



    /*
    * This function evaluates a solution (R and t) extracted from the Essential
    * matrix by computing the number of inliers that such pose has.
    * Inputs:
    *   p_img1/p_im2: measured points in the two cameras
    *   matches: matches between p_img1 and p_img2 with outliers
    *   matches_filter: list of valid matches ids
    *   matches_inliers: list of matches ids that are considered inliers
    *   R/t: pose to evaluate
    *   K: camera matrix of the two cameras
    * Outputs:
    *   #inliers (size of the matches_inliers vector)
    */
    unsigned int compute_transformation_inliers(const std::vector<cv::KeyPoint>& p_img1, \
                                                const std::vector<cv::KeyPoint>& p_img2, \
                                                const std::vector<cv::DMatch>& matches, \
                                                const std::vector<unsigned int>& matches_filter, \
                                                std::vector<unsigned int>& matches_inliers, \
                                                const cv::Mat& R, const cv::Mat& t, \
                                                const cv::Mat& K) {
        
        // Initialization
        matches_inliers.clear();
        const unsigned int n_points = matches_filter.size();
        const float reprojection_threshold = 4.0;
        cv::Mat u,w,vt;
        cv::Mat A = cv::Mat::zeros(4,4,CV_32F);
        cv::Mat current_3D_point, current_3D_point_wrt2, d1, d2;
        bool is_current_match_valid;
        float current_cos_parallax, imx, imy, invz;
        const float& fx = K.at<float>(0,0);
        const float& fy = K.at<float>(1,1);
        const float& cx = K.at<float>(0,2);
        const float& cy = K.at<float>(1,2);

        // Compute the projection matrices of the two cameras
        cv::Mat P1 = cv::Mat::zeros(3,4,CV_32F);
        cv::Mat P2 = cv::Mat::zeros(3,4,CV_32F);
        K.copyTo(P1.rowRange(0,3).colRange(0,3));   // first pose is at the origin
        R.copyTo(P2.rowRange(0,3).colRange(0,3));
        t.copyTo(P2.rowRange(0,3).col(3));
        P2 = K*P2;

        // Compute the origins of the two cameras
        cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);
        cv::Mat O2 = -R.t()*t;

        // For each valid match
        matches_inliers.reserve(n_points);
        for(unsigned int i=0; i<n_points; ++i) {

            // Take references to the current matched points
            const float& p1_x = p_img1[matches[matches_filter[i]].queryIdx].pt.x;
            const float& p1_y = p_img1[matches[matches_filter[i]].queryIdx].pt.y;
            const float& p2_x = p_img2[matches[matches_filter[i]].trainIdx].pt.x;
            const float& p2_y = p_img2[matches[matches_filter[i]].trainIdx].pt.y;

            // Triangulate them with the linear triangulation method
            A.row(0) = p1_x*P1.row(2)-P1.row(0);
            A.row(1) = p1_y*P1.row(2)-P1.row(1);
            A.row(2) = p2_x*P2.row(2)-P2.row(0);
            A.row(3) = p2_y*P2.row(2)-P2.row(1);
            cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
            current_3D_point = vt.row(3).t();
            current_3D_point = current_3D_point.rowRange(0,3)/current_3D_point.at<float>(3);
            const float& current_3D_point_x = current_3D_point.at<float>(0);
            const float& current_3D_point_y = current_3D_point.at<float>(1);
            const float& current_3D_point_z = current_3D_point.at<float>(2);

            // Check that the point is not at infinity
            if(!std::isfinite(current_3D_point_x) || \
                !std::isfinite(current_3D_point_y) || \
                !std::isfinite(current_3D_point_z)) {
                continue;
            }

            // Compute the parallax
            d1 = current_3D_point - O1;
            d2 = current_3D_point - O2;
            current_cos_parallax = d1.dot(d2)/(cv::norm(d1)*cv::norm(d2));

            // Check if the point is in front of the first camera
            if(current_3D_point_z<=0 && current_cos_parallax<0.99998)
                continue;
            
            // Check if the point is in front of the second camera
            current_3D_point_wrt2 = R*current_3D_point+t;
            const float& current_3D_point_wrt2_x = current_3D_point_wrt2.at<float>(0);
            const float& current_3D_point_wrt2_y = current_3D_point_wrt2.at<float>(1);
            const float& current_3D_point_wrt2_z = current_3D_point_wrt2.at<float>(2);
            if(current_3D_point_wrt2_z<=0 && current_cos_parallax<0.99998)
                continue;
            
            // Check reprojection error in first image
            invz = 1.0/current_3D_point.at<float>(2);
            imx = fx*current_3D_point_x*invz+cx;
            imy = fy*current_3D_point_y*invz+cy;
            if(((imx-p1_x)*(imx-p1_x)+(imy-p1_y)*(imy-p1_y)) > \
                    reprojection_threshold)
                continue;

            // Check reprojection error in second image
            invz = 1.0/current_3D_point_wrt2.at<float>(2);
            imx = fx*current_3D_point_wrt2_x*invz+cx;
            imy = fy*current_3D_point_wrt2_y*invz+cy;
            if(((imx-p2_x)*(imx-p2_x)+(imy-p2_y)*(imy-p2_y)) > \
                    reprojection_threshold)
                continue;
            
            // All checks passed, it is an inlier
            matches_inliers.emplace_back(matches_filter[i]);

        }

        return matches_inliers.size();

    }

} // namespace SLucAM