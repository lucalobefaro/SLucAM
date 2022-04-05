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

// TODO: delete this
#include <iostream>
using namespace std;



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
    *   predicted_pose: the pose of the meas2 w.r.t. pose1
    *   triangulated_points: all the points triangulated from the matches
    *   ransac_iter
    *   rotation_only_threshold_rate: threshold to understand if we have a 
    *       rotation only
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
                    const unsigned int& ransac_iter, \
                    const float& rotation_only_threshold_rate) {
        
        // Match the two measurements
        matcher.match_measurements(meas1, meas2, matches);
        unsigned int n_matches = matches.size();

        // Generate random sets of matches indices, one for iteration
        std::vector<std::vector<unsigned int>> random_idxs(ransac_iter, std::vector<unsigned int>(8));
        std::vector<unsigned int> indices(n_matches);
        std::iota(indices.begin(), indices.end(), 0);
        srand(time(NULL));
        for(unsigned int i=0; i<ransac_iter; ++i) {
            std::random_shuffle(indices.begin(), indices.end(), [](int i) {return rand() % i;});
            random_idxs[i].assign(indices.begin(), indices.begin()+8);
        }

        // Perform RANSAC
        const std::vector<cv::KeyPoint>& p_img1 = meas1.getPoints();
        const std::vector<cv::KeyPoint>& p_img1_normalized = meas1.getNormalizedPoints();
        const std::vector<cv::KeyPoint>& p_img2 = meas2.getPoints();
        const std::vector<cv::KeyPoint>& p_img2_normalized = meas2.getNormalizedPoints();
        const cv::Mat& T1 = meas1.getTNorm();
        const cv::Mat& T2 = meas2.getTNorm();
        std::vector<bool> inliers_mask_F(n_matches);
        std::vector<bool> inliers_mask_H(n_matches);

        unsigned int n_inliers_F;
        float score_F = SLucAM::ransac_foundamental(p_img1_normalized, \
                                                    p_img2_normalized, \
                                                    p_img1, \
                                                    p_img2, \
                                                    matches, \
                                                    T1, T2, \
                                                    random_idxs, \
                                                    n_inliers_F, \
                                                    inliers_mask_F, \
                                                    ransac_iter);
        unsigned int n_inliers_H;
        float score_H = SLucAM::ransac_homography(p_img1_normalized, \
                                                    p_img2_normalized, \
                                                    p_img1, \
                                                    p_img2, \
                                                    matches, \
                                                    T1, T2, \
                                                    random_idxs, \
                                                    n_inliers_H, \
                                                    inliers_mask_H, \
                                                    ransac_iter);

        // According to [Raúl Mur-Artal, J. M. M. Montiel and Juan D. Tardós. 
        // ORB-SLAM: A Versatile and Accurate Monocular SLAM System. 
        // IEEE Transactions on Robotics, vol. 31, no. 5, pp. 1147-1163, 2015. 
        // (2015 IEEE Transactions on Robotics Best Paper Award).]
        // we trust of F only if RH <= 0.45, where RH = (score_H)/(score_H+score_F)
        float RH = (score_H)/(score_H+score_F);
        if(RH > 0.45) {
            cout << "ROTATION ONLY" << endl;    // TODO: delete this
            return false;
        }

        // From the inliers mask compute a vector of indices for matches
        matches_filter.reserve(n_matches);
        for(unsigned int i=0; i<n_matches; ++i) {
            if(inliers_mask_F[i]) {
                matches_filter.emplace_back(i);
            }
        }
        matches_filter.shrink_to_fit();

        // Compute F on all inliers (its best version)
        cv::Mat F;
        estimate_foundamental(p_img1_normalized, p_img2_normalized, matches, matches_filter, F);
        F = T1.t()*F*T2;    // Denormalization

        // Compute pose of image2 w.r.t. image1 from F
        extract_X_from_F(p_img1, p_img2, matches, matches_filter, \
                            F, K, predicted_pose, triangulated_points);
        
        return true;

    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions that evaluates the "goodness" of a matrix F or H
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    float evaluate_foundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                        const std::vector<cv::KeyPoint>& p_img2, \
                                        const std::vector<cv::DMatch>& matches, \
                                        const cv::Mat& F, \
                                        std::vector<bool>& inliers_mask, \
                                        const float& inliers_threshold, \
                                        unsigned int& n_inliers) {
        
        // Initialization
        unsigned int n_matches = matches.size();
        float d1, d2, Fp1, Fp2, Fp3;
        float score = 0.0;
        n_inliers = 0;

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
            Fp1 = F11*p1_x + F21*p1_y + F31;
            Fp2 = F12*p1_x + F22*p1_y + F32;
            Fp3 = F13*p1_x + F23*p1_y + F33;
            d1 = pow(Fp1*p2_x+Fp2*p2_y+Fp3, 2)/(pow(Fp1, 2) + pow(Fp2, 2));

            // If the distance of this reprojection is over the threshold
            // then discard it as outlier, otherwise compute the score
            if(isnan(d1) || d1 >= inliers_threshold) {
                is_inlier = false;
            } else {
                score += 5.991-d1;
            }

            // Compute the square distance between point 1 and point 2 
            // projected in the image 1
            Fp1 = F11*p2_x + F12*p2_y + F13;
            Fp2 = F21*p2_x + F22*p2_y + F23;
            Fp3 = F31*p2_x + F32*p2_y + F33;
            d2 = pow(Fp1*p1_x+Fp2*p1_y+Fp3, 2)/(pow(Fp1, 2) + pow(Fp2, 2));

            // If the distance of this reprojection is over the threshold
            // then discard it as outlier, otherwise compute the score
            if(isnan(d2) || d2 >= inliers_threshold) {
                is_inlier = false;
            } else {
                score += 5.991-d2;
            }

            // If both the computed distances are under the threshold
            // consider the currend correspondance as inlier
            if(is_inlier) {
                ++n_inliers;
                inliers_mask[i] = true;
            } else {
                inliers_mask[i] = false;
            }

        }

        // Return the score
        return score;
    }



    float evaluate_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                        const std::vector<cv::KeyPoint>& p_img2, \
                                        const std::vector<cv::DMatch>& matches, \
                                        const cv::Mat& H, \
                                        std::vector<bool>& inliers_mask, \
                                        const float& inliers_threshold, \
                                        unsigned int& n_inliers) {
        
        // Initialization
        unsigned int n_matches = matches.size();
        float d1, d2, Hp1, Hp2, Hp3;
        float score = 0.0;
        n_inliers = 0;

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

            // If the distance of this reprojection is over the threshold
            // then discard it as outlier, otherwise compute the score
            if(isnan(d1) || d1 >= inliers_threshold) {
                is_inlier = false;
            } else {
                score += 5.991-d1;
            }

            // Compute the square distance d2 = d(inv(H)x2, x1)
            Hp3 = inv_H31*p2_x + inv_H32*p2_y + inv_H33;
            Hp1 = (inv_H11*p2_x + inv_H12*p2_y + inv_H13)/Hp3;
            Hp2 = (inv_H21*p2_x + inv_H22*p2_y + inv_H23)/Hp3;
            d2 = pow(p1_x-Hp1, 2) + pow(p1_y-Hp2, 2);

            // If the distance of this reprojection is over the threshold
            // then discard it as outlier, otherwise compute the score
            if(isnan(d2) || d2 >= inliers_threshold) {
                is_inlier = false;
            } else {
                score += 5.991-d2;
            }

            // If both the computed distances are under the threshold
            // consider the currend correspondance as inlier
            if(is_inlier) {
                ++n_inliers;
                inliers_mask[i] = true;
            } else {
                inliers_mask[i] = false;
            }

        }

        // Return the score
        return score;
    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of functions that uses RANSAC to compute F and H 
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function that performs the RANSAC method on a set of correspondances
    * of points in order to find the minimal set of points in such correspondances
    * that allows to obtain the Foundamental Matrix that allows to obtain the
    * higher number of inliers. IMPORTANT: we assume that the points 
    * p_img1_normalized/p_img2_normalized are normalized using, respectively, 
    * the matrices T1 and T2. We need them to denormalize the produced F matrix.
    * Then it is evaluated on p_img1/p_img2, the original points, not normalized.
    * Inputs:
    *   p_img1_normalized/p_img2_normalized: points to compute F, normalized with
    *               T1 and T2 matrices.
    *   p_img1/p_img2: the same points, but not normalized
    *   matches: the correspondances between the two set of points, with 
    *               outliers
    *   T1/T2: normalization matrices for p_img1/p_img2
    *   rand_idxs: n_iter sets of 8 random indices, generated in order to take
    *               random matches at each iteration
    *   n_inliers: the number of inliers finded (output)
    *   inliers_mask: vector that contains, for each correspondance in the matches
    *                   vector, "true" if such match is an inlier, "false" otherwise
    *                   (output)
    *   n_iter: #iterations to perform on RANSAC
    *   inliers_threshold: minimum distance that two points must have, after
    *                       "projected" with a F matrix, in order to be considered
    *                       inliers
    *  Outputs:
    *   score: score obtained with the best F
    */
    float ransac_foundamental(const std::vector<cv::KeyPoint>& p_img1_normalized, \
                            const std::vector<cv::KeyPoint>& p_img2_normalized, \
                            const std::vector<cv::KeyPoint>& p_img1, \
                            const std::vector<cv::KeyPoint>& p_img2, \
                            const std::vector<cv::DMatch>& matches, \
                            const cv::Mat& T1, const cv::Mat& T2, \
                            const std::vector<std::vector<unsigned int>>& rand_idxs, \
                            unsigned int& n_inliers, \
                            std::vector<bool>& inliers_mask, \
                            const unsigned int n_iter, \
                            const float inliers_threshold) {
        
        // Initialization
        unsigned int n_matches = matches.size();
        unsigned int current_num_inliers = 0;
        float best_score = 0.0;
        float current_score = 0.0;
        std::vector<bool> current_inliers_mask(n_matches);
        inliers_mask.resize(n_matches);
        cv::Mat F;
        n_inliers = 0;

        // Perform RANSAC for n_iter iterations
        for(unsigned int iter=0; iter<n_iter; ++iter) {

            // Compute the Foundamental Matrix on the current minimal set
            estimate_foundamental(p_img1_normalized, p_img2_normalized, matches, rand_idxs[iter], F);

            // Denormalize F
            F = T1.t()*F*T2;
            
            // Compute the score with the current F
            current_score = evaluate_foundamental(p_img1, p_img2, \
                                        matches, F, current_inliers_mask, \
                                        inliers_threshold, current_num_inliers);

            // If it is the best encountered so far, save it as the best one
            if(current_score > best_score) {
                best_score = current_score;
                n_inliers = current_num_inliers;
                inliers_mask = current_inliers_mask;
            }
        }

        // Return the score
        return best_score;
    
    }


    /*
    * Function that performs the RANSAC method on a set of correspondances
    * of points in order to find the minimal set of points in such correspondances
    * that allows to obtain the Homgraphy Matrix that allows to obtain the
    * higher number of inliers. IMPORTANT: we assume that the points 
    * p_img1_normalized/p_img2_normalized are normalized using, respectively, 
    * the matrices T1 and T2. We need them to denormalize the produced H matrix.
    * Then it is evaluated on p_img1/p_img2, the original points, not normalized.
    * Inputs:
    *   p_img1_normalized/p_img2_normalized: points to compute H, normalized with
    *               T1 and T2 matrices.
    *   p_img1/p_img2: the same points, but not normalized
    *   matches: the correspondances between the two set of points, with 
    *               outliers
    *   T1/T2: normalization matrices for p_img1/p_img2
    *   rand_idxs: n_iter sets of 4 random indices, generated in order to take
    *               random matches at each iteration
    *   n_inliers: the number of inliers finded (output)
    *   inliers_mask: vector that contains, for each correspondance in the matches
    *                   vector, "true" if such match is an inlier, "false" otherwise
    *                   (output)
    *   n_iter: #iterations to perform on RANSAC
    *   inliers_threshold: minimum distance that two points must have, after
    *                       "projected" with a H matrix, in order to be considered
    *                       inliers
    *  Outputs:
    *   score: score obtained with the best H
    */
    float ransac_homography(const std::vector<cv::KeyPoint>& p_img1_normalized, \
                            const std::vector<cv::KeyPoint>& p_img2_normalized, \
                            const std::vector<cv::KeyPoint>& p_img1, \
                            const std::vector<cv::KeyPoint>& p_img2, \
                            const std::vector<cv::DMatch>& matches, \
                            const cv::Mat& T1, const cv::Mat& T2, \
                            const std::vector<std::vector<unsigned int>>& rand_idxs, \
                            unsigned int& n_inliers, \
                            std::vector<bool>& inliers_mask, \
                            const unsigned int n_iter, \
                            const float inliers_threshold) {
        
        // Initialization
        unsigned int n_matches = matches.size();
        unsigned int current_num_inliers = 0;
        float best_score = 0.0;
        float current_score = 0.0;
        std::vector<bool> current_inliers_mask(n_matches);
        inliers_mask.resize(n_matches);
        cv::Mat H;
        n_inliers = 0;

        // Perform RANSAC for n_iter iterations
        for(unsigned int iter=0; iter<n_iter; ++iter) {

            // Compute the Homography Matrix on the current minimal set
            estimate_homography(p_img1_normalized, p_img2_normalized, matches, rand_idxs[iter], H);

            // Denormalize H
            H = T2.inv()*H*T1;
            
            // Compute the score with the current F
            current_score = evaluate_homography(p_img1, p_img2, \
                                        matches, H, current_inliers_mask, \
                                        inliers_threshold, current_num_inliers);
            
            // If it is the best encountered so far, save it as the best one
            if(current_score > best_score) {
                best_score = current_score;
                n_inliers = current_num_inliers;
                inliers_mask = current_inliers_mask;
            }
        }

        // Return the score
        return best_score;
    
    }
    
} // namespace SLucAM