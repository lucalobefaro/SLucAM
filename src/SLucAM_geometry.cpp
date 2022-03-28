//
// SLucAM_geometry.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_geometry.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <limits>

// TODO delete this
#include <iostream>
using namespace std;



// -----------------------------------------------------------------------------
// Implementation of basic geometric functions
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * A simple normalization function for points. It shifts all points and bring
    * their centroid to the origin and then scale them in such a way their 
    * average distance from the origin is sqrt(2).
    * Inputs:
    *   points: points to normalize
    *   normalized_points: output points, normalized
    *   T: transformation matrix to normalize the point (useful
    *       for denormalization)
    */
    void normalize_points(const std::vector<cv::KeyPoint>& points, \
                            std::vector<cv::KeyPoint>& normalized_points, \
                            cv::Mat& T) {
        
        // Initialization
        unsigned int n_points = points.size();
        normalized_points.resize(n_points);
        T = cv::Mat::eye(3, 3, CV_32F);
        
        // Compute the centroid of the points
        float mu_x = 0;
        float mu_y = 0;
        for(const cv::KeyPoint& p: points) {
            mu_x += p.pt.x;
            mu_y += p.pt.y;
        }
        mu_x /= n_points;
        mu_y /= n_points;

        // Shift the points such that the centroid will be the origin
        // and in the meantime compute the average distance from the
        // origin
        float average_distance = 0;
        for(unsigned int i=0; i<n_points; i++) {
            normalized_points[i].pt.x = points[i].pt.x-mu_x;
            normalized_points[i].pt.y = points[i].pt.y-mu_y;
            average_distance += sqrt((normalized_points[i].pt.x*\
                                        normalized_points[i].pt.x)+\
                                     (normalized_points[i].pt.y*\
                                        normalized_points[i].pt.y));
        }
        average_distance /= n_points;
        
        // Scale the points such that the average distance from 
        // the origin is sqrt(2)
        float scale = sqrt(2)/average_distance;
        for(unsigned int i=0; i<n_points; i++) {
            normalized_points[i].pt.x *= scale;
            normalized_points[i].pt.y *= scale;
        }

        // Ensemble the T matrix
        T = cv::Mat::eye(3, 3, CV_32F);
        T.at<float>(0,0) = scale;
        T.at<float>(1,1) = scale;
        T.at<float>(0,2) = -mu_x*scale;
        T.at<float>(1,2) = -mu_y*scale; 
    }


    /*
    * Function that triangulates a bunch of points.
    * Inputs:
    *   p_img1/p_img2: points seen from two cameras
    *   matches: all matches between p_img1 and p_img2, with outliers
    *   idxs: we will consider only those points contained in 
    *           the matches vector, for wich we have indices in this vector idxs
    *   X: pose of the camera 2 w.r.t. camera 1
    *   K: camera matrix of the two cameras
    *   triangulated_points: vector where to store the triangulated points
    * Outputs:
    *   n_triangulated_points: number of triangulated points
    */
    unsigned int triangulate_points(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const std::vector<unsigned int>& idxs, \
                                    const cv::Mat& X, const cv::Mat& K, \
                                    std::vector<cv::Point3f>& triangulated_points) {
        
        // Initialization
        const unsigned int n_points = idxs.size();
        unsigned int n_triangulated_points = 0;
        triangulated_points.reserve(n_points);

        // Pre-computation to save time
        const cv::Mat inv_K = K.inv();
        const float& inv_K11 = inv_K.at<float>(0,0);
        const float& inv_K12 = inv_K.at<float>(0,1);
        const float& inv_K13 = inv_K.at<float>(0,2);
        const float& inv_K21 = inv_K.at<float>(1,0);
        const float& inv_K22 = inv_K.at<float>(1,1);
        const float& inv_K23 = inv_K.at<float>(1,2);
        const float& inv_K31 = inv_K.at<float>(2,0);
        const float& inv_K32 = inv_K.at<float>(2,1);
        const float& inv_K33 = inv_K.at<float>(2,2);

        const cv::Mat inv_X = X.inv();
        const cv::Mat inv_R_inv_K = inv_X(cv::Rect(0,0,3,3))*inv_K;
        const float& inv_R_inv_K11 = inv_R_inv_K.at<float>(0,0);
        const float& inv_R_inv_K12 = inv_R_inv_K.at<float>(0,1);
        const float& inv_R_inv_K13 = inv_R_inv_K.at<float>(0,2);
        const float& inv_R_inv_K21 = inv_R_inv_K.at<float>(1,0);
        const float& inv_R_inv_K22 = inv_R_inv_K.at<float>(1,1);
        const float& inv_R_inv_K23 = inv_R_inv_K.at<float>(1,2);
        const float& inv_R_inv_K31 = inv_R_inv_K.at<float>(2,0);
        const float& inv_R_inv_K32 = inv_R_inv_K.at<float>(2,1);
        const float& inv_R_inv_K33 = inv_R_inv_K.at<float>(2,2);

        cv::Mat O2 = inv_X(cv::Rect(3,0,1,3));
        const float O2_x = O2.at<float>(0,0);
        const float O2_y = O2.at<float>(1,0);
        const float O2_z = O2.at<float>(2,0);
        
        // Triangluate each couple of points
        for(unsigned int i=0; i<n_points; ++i) {

            // Take references to the current couple of points
            const float& p1_x = p_img1[matches[idxs[i]].queryIdx].pt.x;
            const float& p1_y = p_img1[matches[idxs[i]].queryIdx].pt.y;
            const float& p2_x = p_img2[matches[idxs[i]].trainIdx].pt.x;
            const float& p2_y = p_img2[matches[idxs[i]].trainIdx].pt.y;

            // Ensemble the matrix D as [-d1,d2] where d1 and d2 are
            // the directions starting from p1 and p2 respectively, 
            // computed as:
            //  d1 = inv_K*p1
            //  d2 = inv_R_inv_K*p2 
            cv::Mat D = cv::Mat::eye(3,2,CV_32F);
            D.at<float>(0,0) = -(inv_K11*p1_x + inv_K12*p1_y + inv_K13);
            D.at<float>(1,0) = -(inv_K21*p1_x + inv_K22*p1_y + inv_K23);
            D.at<float>(2,0) = -(inv_K31*p1_x + inv_K32*p1_y + inv_K33);
            D.at<float>(0,1) = (inv_R_inv_K11*p2_x + inv_R_inv_K12*p2_y + inv_R_inv_K13);
            D.at<float>(1,1) = (inv_R_inv_K21*p2_x + inv_R_inv_K22*p2_y + inv_R_inv_K23);;
            D.at<float>(2,1) = (inv_R_inv_K31*p2_x + inv_R_inv_K32*p2_y + inv_R_inv_K33);;

            // Triangulate
            cv::Mat s = (-(D.t()*D)).inv()*(D.t()*O2);
            const float& s1 = s.at<float>(0,0);
            const float& s2 = s.at<float>(0,1);
            if(s1<0 || s2<0) {    // The point is behind the camera
                triangulated_points.emplace_back(cv::Point3f(0,0,0));
            } else {
                triangulated_points.emplace_back( 
                    cv::Point3f(0.5*(-D.at<float>(0,0)*s1 + (D.at<float>(0,1)*s2+O2_x)), \
                                0.5*(-D.at<float>(1,0)*s1 + (D.at<float>(1,1)*s2+O2_y)), \
                                0.5*(-D.at<float>(2,0)*s1 + (D.at<float>(2,1)*s2+O2_z))) 
                );
                ++n_triangulated_points;
            }
        }

        triangulated_points.shrink_to_fit();

        return n_triangulated_points;
    }


    /*
    * Function that, given a transformation matrix and a perturbation vector
    * where from position "starting_idx" to "starting_idx+6" contains the
    * [tx, ty, tz, x-angle, y-angle, z-angle] perturbation to apply.
    * Inputs:
    *   perturbation: the perturbation vector
    *   T_matrix: the transformation matrix to "perturb"
    *   starting_idx: the position from which we have the updates for T_matrix 
    */
    void apply_perturbation_Tmatrix(const cv::Mat& perturbation, \
                                    cv::Mat& T_matrix, const unsigned int& starting_idx) {
        
        // Some reference to save time
        const float& tx = perturbation.at<float>(starting_idx);
        const float& ty = perturbation.at<float>(starting_idx+1);
        const float& tz = perturbation.at<float>(starting_idx+2);
        const float& x_angle = perturbation.at<float>(starting_idx+3);
        const float cx = cos(x_angle);
        const float sx = sin(x_angle);
        const float& y_angle = perturbation.at<float>(starting_idx+4);
        const float cy = cos(y_angle);
        const float sy = sin(y_angle);
        const float& z_angle = perturbation.at<float>(starting_idx+5);
        const float cz = cos(z_angle);
        const float sz = sin(z_angle);
        const float T11 = T_matrix.at<float>(0,0);
        const float T12 = T_matrix.at<float>(0,1);
        const float T13 = T_matrix.at<float>(0,2);
        const float T14 = T_matrix.at<float>(0,3);
        const float T21 = T_matrix.at<float>(1,0);
        const float T22 = T_matrix.at<float>(1,1);
        const float T23 = T_matrix.at<float>(1,2);
        const float T24 = T_matrix.at<float>(1,3);
        const float T31 = T_matrix.at<float>(2,0);
        const float T32 = T_matrix.at<float>(2,1);
        const float T33 = T_matrix.at<float>(2,2);
        const float T34 = T_matrix.at<float>(2,3);
        const float T41 = T_matrix.at<float>(3,0);
        const float T42 = T_matrix.at<float>(3,1);
        const float T43 = T_matrix.at<float>(3,2);
        const float T44 = T_matrix.at<float>(3,3);

        // Apply the perturbation
        T_matrix.at<float>(0,0) = T31*sy + T41*tx + T11*cy*cz - T21*cy*sz;
        T_matrix.at<float>(0,1) = T32*sy + T42*tx + T12*cy*cz - T22*cy*sz;
        T_matrix.at<float>(0,2) = T33*sy + T43*tx + T13*cy*cz - T23*cy*sz;
        T_matrix.at<float>(0,3) = T34*sy + T44*tx + T14*cy*cz - T24*cy*sz;
        T_matrix.at<float>(1,0) = T41*ty + T11*(cx*sz + cz*sx*sy) + \
                                    T21*(cx*cz - sx*sy*sz) - T31*cy*sx;
        T_matrix.at<float>(1,1) = T42*ty + T12*(cx*sz + cz*sx*sy) + \
                                    T22*(cx*cz - sx*sy*sz) - T32*cy*sx;
        T_matrix.at<float>(1,2) = T43*ty + T13*(cx*sz + cz*sx*sy) + \
                                    T23*(cx*cz - sx*sy*sz) - T33*cy*sx;
        T_matrix.at<float>(1,3) = T44*ty + T14*(cx*sz + cz*sx*sy) + \
                                    T24*(cx*cz - sx*sy*sz) - T34*cy*sx;
        T_matrix.at<float>(2,0) = T41*tz + T11*(sx*sz - cx*cz*sy) + \
                                    T21*(cz*sx + cx*sy*sz) + T31*cx*cy;
        T_matrix.at<float>(2,1) = T42*tz + T12*(sx*sz - cx*cz*sy) + \
                                    T22*(cz*sx + cx*sy*sz) + T32*cx*cy;
        T_matrix.at<float>(2,2) = T43*tz + T13*(sx*sz - cx*cz*sy) + \
                                    T23*(cz*sx + cx*sy*sz) + T33*cx*cy;
        T_matrix.at<float>(2,3) = T44*tz + T14*(sx*sz - cx*cz*sy) + \
                                    T24*(cz*sx + cx*sy*sz) + T34*cx*cy;

    }


    // Invert a transformation matrix in a fast way by transposing the
    // rotational part and computing the translational part as
    // -R't
    cv::Mat invert_transformation_matrix(const cv::Mat& T_matrix) {

        cv::Mat T_matrix_inv = cv::Mat::eye(4,4,CV_32F);
        
        // Reference to rotational part (in transpose way)
        const float& R_11 = T_matrix.at<float>(0,0);
        const float& R_12 = T_matrix.at<float>(1,0);
        const float& R_13 = T_matrix.at<float>(2,0);
        const float& R_21 = T_matrix.at<float>(0,1);
        const float& R_22 = T_matrix.at<float>(1,1);
        const float& R_23 = T_matrix.at<float>(2,1);
        const float& R_31 = T_matrix.at<float>(0,2);
        const float& R_32 = T_matrix.at<float>(1,2);
        const float& R_33 = T_matrix.at<float>(2,2);

        // Reference to the translational part
        const float& t_x = T_matrix.at<float>(0,3);
        const float& t_y = T_matrix.at<float>(1,3);
        const float& t_z = T_matrix.at<float>(2,3);

        // Transpose the rotational part
        T_matrix_inv.at<float>(0,0) = R_11;
        T_matrix_inv.at<float>(0,1) = R_12;
        T_matrix_inv.at<float>(0,2) = R_13;
        T_matrix_inv.at<float>(1,0) = R_21;
        T_matrix_inv.at<float>(1,1) = R_22;
        T_matrix_inv.at<float>(1,2) = R_23;
        T_matrix_inv.at<float>(2,0) = R_31;
        T_matrix_inv.at<float>(2,1) = R_32;
        T_matrix_inv.at<float>(2,2) = R_33;

        // Compute the translational part
        T_matrix_inv.at<float>(0,3) = -(R_11*t_x + R_12*t_y + R_13*t_z);
        T_matrix_inv.at<float>(1,3) = -(R_21*t_x + R_22*t_y + R_23*t_z);
        T_matrix_inv.at<float>(2,3) = -(R_31*t_x + R_32*t_y + R_33*t_z);

        return T_matrix_inv;

    }



    /*
    * This function, given a quaternion represented as a cv::Mat with
    * dim 4x1 and with this structure: [w; x; y; z], returns the
    * corresponding rotation matrix R.
    */
    void quaternion_to_matrix(const cv::Mat& quaternion, cv::Mat& R) {

        // Initialization
        R = cv::Mat::zeros(3,3,CV_32F);

        // Create the Eigen quaternion
        Eigen::Quaternionf Eigen_quaternion;
        Eigen_quaternion.w() = quaternion.at<float>(0,0);
        Eigen_quaternion.x() = quaternion.at<float>(1,0);
        Eigen_quaternion.y() = quaternion.at<float>(2,0);
        Eigen_quaternion.z() = quaternion.at<float>(3,0);

        // Convert it to R
        Eigen::Matrix3f Eigen_R = Eigen_quaternion.normalized().toRotationMatrix();

        // Back to cv representation
        R.at<float>(0,0) = Eigen_R(0,0);
        R.at<float>(0,1) = Eigen_R(0,1);
        R.at<float>(0,2) = Eigen_R(0,2);
        R.at<float>(1,0) = Eigen_R(1,0);
        R.at<float>(1,1) = Eigen_R(1,1);
        R.at<float>(1,2) = Eigen_R(1,2);
        R.at<float>(2,0) = Eigen_R(2,0);
        R.at<float>(2,1) = Eigen_R(2,1);
        R.at<float>(2,2) = Eigen_R(2,2);
    }



    /*
    * This function, given a rotation matrix represented as a cv::Mat, 
    * returns the corresponding quaternion represented as a cv::Mat with 
    * dim 4x1 and with this structure: [w; x; y; z].
    */
    void matrix_to_quaternion(cv::Mat& R, cv::Mat& quaternion) {

        // Initialization
        quaternion = cv::Mat::zeros(4,1,CV_32F);

        // Create the Eigen matrix R
        Eigen::Matrix3f Eigen_R(3,3);
        Eigen_R(0,0) = R.at<float>(0,0);
        Eigen_R(0,1) = R.at<float>(0,1);
        Eigen_R(0,2) = R.at<float>(0,2);
        Eigen_R(1,0) = R.at<float>(1,0);
        Eigen_R(1,1) = R.at<float>(1,1);
        Eigen_R(1,2) = R.at<float>(1,2);
        Eigen_R(2,0) = R.at<float>(2,0);
        Eigen_R(2,1) = R.at<float>(2,1);
        Eigen_R(2,2) = R.at<float>(2,2);
        
        // Convert it to quaternion
        Eigen::Quaternionf Eigen_quaternion(Eigen_R);

        // Back to cv representation
        quaternion.at<float>(0,0) = Eigen_quaternion.w();
        quaternion.at<float>(1,0) = Eigen_quaternion.x();
        quaternion.at<float>(2,0) = Eigen_quaternion.y();
        quaternion.at<float>(3,0) = Eigen_quaternion.z();
    }



    /* 
    * This function takes a 3D point p and a costellation c of points
    * and return a pair:
    *   <idx of the nearest point to p in c, distance>
    */
    std::pair<int, float> nearest_3d_point(\
            const cv::Point3f& p, const std::vector<cv::Point3f>& c) {

        // Initialization
        const unsigned int& n_points = c.size();
        float current_distance;
        std::pair<int, float> result(-1, std::numeric_limits<float>::max());

        // For each point in the costellation c
        for(unsigned int i=0; i<n_points; ++i) {
            
            // Take the current point
            const cv::Point3f& p2 = c[i];

            // Compute distance
            current_distance = cv::norm(p-p2);

            // If it is the nearest one so far, save it
            if(current_distance < result.second) {
                result.first = i;
                result.second = current_distance;
            }
        }

        return result;

    }
    
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of multi-view geometry functions
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function that, given a set of, at least, 8 couples of points, estimate
    * the Foundamental matrix (implementation of the 8-point algorithm).
    * Better if the points p_img1 and p_img2 are normalized, in such case
    * please de-normalize F after this function.
    * Inputs:
    *   p_img1/p_img2: input points
    *   matches: the correspondances between the two set of points
    *   idxs: we will compute the F matrix only on those points contained in 
    *           the matches vector, for wich we have indices in this vector idxs
    *   F: estimated Foundamental Matrix
    */
    void estimate_foundamental(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const std::vector<unsigned int>& idxs, \
                                cv::Mat& F) {
        
        // Initialization
        unsigned int n_points = idxs.size();

        // Ensemble the H matrix
        cv::Mat H = cv::Mat::zeros(9,9,CV_32F);
        cv::Mat A = cv::Mat::ones(1,9,CV_32F);
        for(unsigned int i = 0; i<n_points; ++i) {
            const float& p1_x = p_img1[matches[idxs[i]].queryIdx].pt.x;
            const float& p1_y = p_img1[matches[idxs[i]].queryIdx].pt.y;
            const float& p2_x = p_img2[matches[idxs[i]].trainIdx].pt.x;
            const float& p2_y = p_img2[matches[idxs[i]].trainIdx].pt.y;
            A.at<float>(0,0) = p1_x * p2_x;
            A.at<float>(0,1) = p1_y * p2_x;
            A.at<float>(0,2) = p2_x;
            A.at<float>(0,3) = p1_x * p2_y;
            A.at<float>(0,4) = p1_y * p2_y;
            A.at<float>(0,5) = p2_y;
            A.at<float>(0,6) = p1_x;
            A.at<float>(0,7) = p1_y;
            H += A.t()*A;
        }

        // Extract the right eigenvalues from H and build the F matrix
        cv::Mat u,w,vt;
        cv::SVDecomp(H,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        F = vt.row(8).reshape(0,3).t();
    }


    /*
    * Function that, given a set of, at least, 4 couples of points, estimate
    * the Homography (implementation of the DLT algorithm).
    * We assume that the points p_img1 and p_img2 are normalized and,
    * please, de-normalize H after this function.
    * Inputs:
    *   p_img1/p_img2: input points
    *   H: estimated Homography
    *   matches: the correspondances between the two set of points
    *   idxs: we will compute the H matrix only on those points contained in 
    *           the matches vector, for wich we have indices in this vector idxs
    *           In particular we will consider only the first 4 elements of it
    *           (the minimal set to compute H)
    */
    void estimate_homography(const std::vector<cv::KeyPoint>& p_img1, \
                                const std::vector<cv::KeyPoint>& p_img2, \
                                const std::vector<cv::DMatch>& matches, \
                                const std::vector<unsigned int>& idxs, \
                                cv::Mat& H) {
        
        // Initialization
        // TODO: decide if this is good (it is no good if we want to re-estimate
        // the H matrix after RANSAC, for now I want H only to understand if we have
        // a rotation only movement, so I don't need a "perfect" H...)
        unsigned int n_points = 4; // We use only the first 4 points in idxs

        // Ensemble the A matrix
        cv::Mat A = cv::Mat::zeros(2*n_points,9,CV_32F);
        for(int i=0; i<n_points; i++)
        {
            const float& p1_x = p_img1[matches[idxs[i]].queryIdx].pt.x;
            const float& p1_y = p_img1[matches[idxs[i]].queryIdx].pt.y;
            const float& p2_x = p_img2[matches[idxs[i]].trainIdx].pt.x;
            const float& p2_y = p_img2[matches[idxs[i]].trainIdx].pt.y;

            A.at<float>(2*i,3) = -p1_x;
            A.at<float>(2*i,4) = -p1_y;
            A.at<float>(2*i,5) = -1;
            A.at<float>(2*i,6) = p2_y*p1_x;
            A.at<float>(2*i,7) = p2_y*p1_y;
            A.at<float>(2*i,8) = p2_y;
            A.at<float>(2*i+1,0) = p1_x;
            A.at<float>(2*i+1,1) = p1_y;
            A.at<float>(2*i+1,2) = 1;
            A.at<float>(2*i+1,6) = -p2_x*p1_x;
            A.at<float>(2*i+1,7) = -p2_x*p1_y;
            A.at<float>(2*i+1,8) = -p2_x;
        }

        // compute SVD of A
        cv::Mat u,w,vt;
        cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        // Use the last column of V to build the h vector and, reshaping it
        // we obtain the homography matrix
        H = vt.row(8).reshape(0, 3);
    }


    /*
    * This function allows us to obtain the rotation matrix R and the
    * translation vector t between two images, for which we have the 
    * Foundamental Matrix F, composed in a transformation matrix 
    * X = [R|t]. 
    * Inputs:
    *   p_img1/p_img2: points to use in order to understand wich X computed
    *                   triangulates "better"
    *   matches: all matches between p_img1 and p_img2, with outliers
    *   idxs: we will consider only those points contained in 
    *           the matches vector, for wich we have indices in this vector idxs
    *   F: Foundamental Matrix from which to extract X
    *   K: camera matrix
    *   X: output Transformation Matrix extracted from F [R|t]
    *   triangulated_points: the point triangulated using p_img1/p_img2 and X
    */
    void extract_X_from_F(const std::vector<cv::KeyPoint>& p_img1, \
                            const std::vector<cv::KeyPoint>& p_img2, \
                            const std::vector<cv::DMatch>& matches, \
                            const std::vector<unsigned int>& idxs, \
                            const cv::Mat& F, const cv::Mat& K, \
                            cv::Mat& X, \
                            std::vector<cv::Point3f>& triangulated_points) {
        
        // Initialization
        cv::Mat W_mat = cv::Mat::zeros(3,3,CV_32F);
        W_mat.at<float>(0,1) = -1;
        W_mat.at<float>(1,0) = 1;
        W_mat.at<float>(2,2) = 1;
        float best_score = 0;
        float current_score = 0;
        std::vector<cv::Point3f> current_triangulated_points;
        cv::Mat X_pred = cv::Mat::eye(4,4,CV_32F);

        // Extract the essential matrix and decompose it
        cv::Mat u,w,vt,v, ut;
        cv::Mat E = K.t()*F*K;
        cv::SVD::compute(E,w,u,vt);
        v = vt.t();
        ut = u.t();

        // Extract the R matrix (2 solutions)
        cv::Mat R1 = v*W_mat*ut;
        if(cv::determinant(R1) < 0) {   // right handed condition
            R1=-R1;
        }
        cv::Mat R2 = v*W_mat.t()*ut;
        if(cv::determinant(R2)<0) {     // right handed condition
            R2=-R2;
        }

        // Extract the t vector and "normalize" it
        cv::Mat t;
        u.col(2).copyTo(t);
        t=t/cv::norm(t);

        // Evaluate first solution
        X_pred.at<float>(0,0) = R1.at<float>(0,0);
        X_pred.at<float>(0,1) = R1.at<float>(0,1);
        X_pred.at<float>(0,2) = R1.at<float>(0,2);
        X_pred.at<float>(1,0) = R1.at<float>(1,0);
        X_pred.at<float>(1,1) = R1.at<float>(1,1);
        X_pred.at<float>(1,2) = R1.at<float>(1,2);
        X_pred.at<float>(2,0) = R1.at<float>(2,0);
        X_pred.at<float>(2,1) = R1.at<float>(2,1);
        X_pred.at<float>(2,2) = R1.at<float>(2,2);
        X_pred.at<float>(0,3) = t.at<float>(0,0);
        X_pred.at<float>(1,3) = t.at<float>(0,1);
        X_pred.at<float>(2,3) = t.at<float>(0,2);
        current_score = triangulate_points(p_img1, p_img2, matches, idxs, X_pred, K, \
                                            current_triangulated_points);
        if(current_score > best_score) {
            best_score = current_score;
            X = X_pred.clone();
            triangulated_points.swap(current_triangulated_points);
        } 

        // Evaluate second solution
        X_pred.at<float>(0,3) = -X_pred.at<float>(0,3);
        X_pred.at<float>(1,3) = -X_pred.at<float>(1,3);
        X_pred.at<float>(2,3) = -X_pred.at<float>(2,3);
        current_triangulated_points.clear();
        current_score = triangulate_points(p_img1, p_img2, matches, idxs, X_pred, K, \
                                            current_triangulated_points);
        if(current_score > best_score) {
            best_score = current_score;
            X = X_pred.clone();
            triangulated_points.swap(current_triangulated_points);
        } 

        // Evaluate third solution
        X_pred.at<float>(0,0) = R2.at<float>(0,0);
        X_pred.at<float>(0,1) = R2.at<float>(0,1);
        X_pred.at<float>(0,2) = R2.at<float>(0,2);
        X_pred.at<float>(1,0) = R2.at<float>(1,0);
        X_pred.at<float>(1,1) = R2.at<float>(1,1);
        X_pred.at<float>(1,2) = R2.at<float>(1,2);
        X_pred.at<float>(2,0) = R2.at<float>(2,0);
        X_pred.at<float>(2,1) = R2.at<float>(2,1);
        X_pred.at<float>(2,2) = R2.at<float>(2,2);
        X_pred.at<float>(0,3) = t.at<float>(0,0);
        X_pred.at<float>(1,3) = t.at<float>(0,1);
        X_pred.at<float>(2,3) = t.at<float>(0,2);
        current_triangulated_points.clear();
        current_score = triangulate_points(p_img1, p_img2, matches, idxs, X_pred, K, \
                                            current_triangulated_points);
        if(current_score > best_score) {
            best_score = current_score;
            X = X_pred.clone();
            triangulated_points.swap(current_triangulated_points);
        }

        // Evaluate fourth solution
        X_pred.at<float>(0,3) = -X_pred.at<float>(0,3);
        X_pred.at<float>(1,3) = -X_pred.at<float>(1,3);
        X_pred.at<float>(2,3) = -X_pred.at<float>(2,3);
        current_triangulated_points.clear();
        current_score = triangulate_points(p_img1, p_img2, matches, idxs, X_pred, K, \
                                            current_triangulated_points);
        if(current_score > best_score) {
            best_score = current_score;
            X = X_pred.clone();
            triangulated_points.swap(current_triangulated_points);
        } 
    }
    
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Projective ICP functions implementation
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function that computes the error and jacobian for Projective ICP
    * Inputs:
    *   guessed_pose: guessed pose of the world w.r.t. camera
    *   guessed_landmark: point in the space where we think is located the 
    *           measured landmark
    *   measured_point: measured point on the projective plane
    *   K: camera matrix
    *   img_rows: #rows in the image plane pixels matrix
    *   img_cols: #cols in the image plane pixels matrix
    *   error: (output) we assume it is already initialized as a 2x1 
    *       matrix
    *   J: jacobian matrix of the error w.r.t. guessed_pose (output), 
    *       we assume it is already initialized as a 2x6 matrix
    * Outputs:
    *   true if the projection is valid, false otherwise
    */
    bool error_and_jacobian_Posit(const cv::Mat& guessed_pose, \
                                const cv::Point3f& guessed_landmark, \
                                const cv::KeyPoint& measured_point, \
                                const cv::Mat& K, \
                                const float& img_rows, \
                                const float& img_cols, \
                                cv::Mat& error, cv::Mat& J) {
        
        // Some reference to save time+
        const float& K_11 = K.at<float>(0,0);
        const float& K_12 = K.at<float>(0,1);
        const float& K_13 = K.at<float>(0,2);
        const float& K_21 = K.at<float>(1,0);
        const float& K_22 = K.at<float>(1,1);
        const float& K_23 = K.at<float>(1,2);
        const float& K_31 = K.at<float>(2,0);
        const float& K_32 = K.at<float>(2,1);
        const float& K_33 = K.at<float>(2,2);
        const float& X_11 = guessed_pose.at<float>(0,0);
        const float& X_12 = guessed_pose.at<float>(0,1);
        const float& X_13 = guessed_pose.at<float>(0,2);
        const float& X_14 = guessed_pose.at<float>(0,3);
        const float& X_21 = guessed_pose.at<float>(1,0);
        const float& X_22 = guessed_pose.at<float>(1,1);
        const float& X_23 = guessed_pose.at<float>(1,2);
        const float& X_24 = guessed_pose.at<float>(1,3);
        const float& X_31 = guessed_pose.at<float>(2,0);
        const float& X_32 = guessed_pose.at<float>(2,1);
        const float& X_33 = guessed_pose.at<float>(2,2);
        const float& X_34 = guessed_pose.at<float>(2,3);
        const float& X_41 = guessed_pose.at<float>(3,0);
        const float& X_42 = guessed_pose.at<float>(3,1);
        const float& X_43 = guessed_pose.at<float>(3,2);
        const float& X_44 = guessed_pose.at<float>(3,3);
        const float& P_x = guessed_landmark.x;
        const float& P_y = guessed_landmark.y;
        const float& P_z = guessed_landmark.z;

        // Compute the position of the point w.r.t. camera frame
        const float p_cam_x = (X_11*P_x + X_12*P_y + X_13*P_z) + X_14;
        const float p_cam_y = (X_21*P_x + X_22*P_y + X_23*P_z) + X_24;
        const float p_cam_z = (X_31*P_x + X_32*P_y + X_33*P_z) + X_34; 

        // Check if the prediction is in front of the camera
        if(p_cam_z < 0) return false;

        // Compute the prediction (projection)
        const float p_camK_x = K_11*p_cam_x + K_12*p_cam_y + K_13*p_cam_z;
        const float p_camK_y = K_21*p_cam_x + K_22*p_cam_y + K_23*p_cam_z;
        const float p_camK_z = K_31*p_cam_x + K_32*p_cam_y + K_33*p_cam_z;
        const float iz = 1/(p_camK_z);
        const float z_hat_x = p_camK_x*iz;
        const float z_hat_y = p_camK_y*iz;

        // Check if the point prediction on projection plane is inside 
        // the camera frustum
        // TODO: assicurati che img_cols e img_rows siano corretti
        if (z_hat_x < 0 || 
            z_hat_x > img_cols ||
            z_hat_y < 0 || 
            z_hat_y > img_rows)
            return false;
                
        // Compute the error
        error.at<float>(0,0) = z_hat_x - measured_point.pt.x;
        error.at<float>(1,0) = z_hat_y - measured_point.pt.y;

        // Compute the Jacobian
        const float iz2 = iz*iz;
        const float p_cam_iz2_x = -p_camK_x*iz2;
        const float p_cam_iz2_y = -p_camK_y*iz2;

        J.at<float>(0,0) = K_11*iz + K_31*p_cam_iz2_x;
        J.at<float>(0,1) = K_12*iz + K_32*p_cam_iz2_x;
        J.at<float>(0,2) = K_13*iz + K_33*p_cam_iz2_x;
        J.at<float>(0,3) = p_cam_y*(K_13*iz + K_33*p_cam_iz2_x) - \
                            p_cam_z*(K_12*iz + K_32*p_cam_iz2_x);
        J.at<float>(0,4) = p_cam_z*(K_11*iz + K_31*p_cam_iz2_x) - \
                            p_cam_x*(K_13*iz + K_33*p_cam_iz2_x);
        J.at<float>(0,5) = p_cam_x*(K_12*iz + K_32*p_cam_iz2_x) - \
                            p_cam_y*(K_11*iz + K_31*p_cam_iz2_x);
        J.at<float>(1,0) = K_21*iz + K_31*p_cam_iz2_y;
        J.at<float>(1,1) = K_22*iz + K_32*p_cam_iz2_y;
        J.at<float>(1,2) = K_23*iz + K_33*p_cam_iz2_y;
        J.at<float>(1,3) = p_cam_y*(K_23*iz + K_33*p_cam_iz2_y) - \
                            p_cam_z*(K_22*iz + K_32*p_cam_iz2_y);
        J.at<float>(1,4) = p_cam_z*(K_21*iz + K_31*p_cam_iz2_y) - \
                            p_cam_x*(K_23*iz + K_33*p_cam_iz2_y);
        J.at<float>(1,5) = p_cam_x*(K_22*iz + K_32*p_cam_iz2_y) - \
                            p_cam_y*(K_21*iz + K_31*p_cam_iz2_y);

        return true;

    }


    /*
    * Function that perform, given a measurement taken from a camera,
    * the projective ICP to get the pose of the camera w.r.t. the world
    * from which such measurement are taken. There will be taken in consideration
    * only such measurements for which the pose of the landmark is already
    * triangulated (so guessed)
    * Inputs:
    *   guessed_pose: initial guess
    *   measurement: the measurement for which we need to predict the pose
    *   points_associations: list of associations 2D point <-> 3D point
    *   landmarks: set of triangulated landmarks
    *   K: camera matrix
    *   n_iterations: #iterations to perform for Posit
    *   kernel_threshold: threshold for the outliers
    *   threshold_to_ignore: error threshold that determine if an outlier 
    *           is too outlier to be considered
    */
    void perform_Posit(cv::Mat& guessed_pose, \
                        const Measurement& measurement, \
                        const std::vector<std::pair<unsigned int, \
                                unsigned int>>& points_associations, \
                        const std::vector<cv::Point3f>& landmarks, \
                        const cv::Mat& K, \
                        const unsigned int& n_iterations, \
                        const float& kernel_threshold, \
                        const float& threshold_to_ignore, \
                        const float& damping_factor) {
        
        // Initialization
        const unsigned int n_observations = points_associations.size();
        const float img_rows = 2*K.at<float>(1, 2);
        const float img_cols = 2*K.at<float>(0, 2);
        float current_chi = 0.0;
        std::vector<unsigned int> n_inliers(n_iterations, 0);
        std::vector<float> chi_stats(n_iterations, 0.0);
        cv::Mat H, b;
        cv::Mat error = cv::Mat::zeros(2,1,CV_32F);
        cv::Mat J = cv::Mat::zeros(2,6,CV_32F);
        const cv::Mat DampingMatrix = \
                    cv::Mat::eye(6, 6, CV_32F)*damping_factor;

        // For each iteration
        for(unsigned int iter=0; iter<n_iterations; ++iter) {
            
            // Reset H and b
            H = cv::Mat::zeros(6,6,CV_32F);
            b = cv::Mat::zeros(6,1,CV_32F);

            // For each observation
            for(unsigned int obs_idx=0; obs_idx<n_observations; ++obs_idx) {

                // Take the measured 2D point for the current observation
                const cv::KeyPoint& measured_point = \
                        measurement.getPoints()[points_associations[obs_idx].first];

                // Take the guessed landmark position of the current observation
                const cv::Point3f& guessed_landmark = \
                        landmarks[points_associations[obs_idx].second];
                
                // Compute error and jacobian
                if(!error_and_jacobian_Posit(guessed_pose, guessed_landmark, \
                                                measured_point, K, img_rows, \
                                                img_cols, error, J))
                    continue;   // Discard not valid projections

                // Compute chi error
                const float& e_1 = error.at<float>(0,0);
                const float& e_2 = error.at<float>(1,0);
                current_chi = (e_1*e_1) + (e_2*e_2);

                // Deal with outliers
                if(current_chi > threshold_to_ignore)
                    continue;

                // Robust kernel
                if(current_chi > kernel_threshold) {
                    error *= sqrt(kernel_threshold/current_chi);
                    current_chi = kernel_threshold;
                } else {
                    ++n_inliers[iter];
                }

                // Update chi stats
                chi_stats[iter] += current_chi;

                // Update H and b
                H += J.t()*J;
                b += J.t()*error;

            }

            // Damping the H matrix
            H += DampingMatrix;

            // Solve linear system to get the perturbation
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> \
                        H_Eigen(H.ptr<float>(), H.rows, H.cols);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> \
                        b_Eigen(b.ptr<float>(), b.rows, b.cols);
            Eigen::VectorXf dx_Eigen = H_Eigen.ldlt().solve(-b_Eigen);
            cv::Mat dx(dx_Eigen.rows(), dx_Eigen.cols(), CV_32F, dx_Eigen.data());

            // Apply the perturbation
            apply_perturbation_Tmatrix(dx, guessed_pose, 0);
        }

    }

} // namespace SLucAM