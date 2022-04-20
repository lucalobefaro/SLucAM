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
    * Function that triangulates a bunch of points seen from two cameras.
    * Inputs:
    *   p_img1/p_img2: points seen from two cameras
    *   matches: all matches between p_img1 and p_img2, with outliers
    *   idxs: we will consider only those points contained in 
    *           the matches vector, for wich we have indices in this vector idxs
    *   pose1/pose2: pose of the two cameras
    *   K: camera matrix of the two cameras
    *   triangulated_points: vector where to store the triangulated points
    *       (expressed w.r.t. world)
    * Outputs:
    *   n_triangulated_points: number of triangulated points
    */
    unsigned int triangulate_points(const std::vector<cv::KeyPoint>& p_img1, \
                                    const std::vector<cv::KeyPoint>& p_img2, \
                                    const std::vector<cv::DMatch>& matches, \
                                    const std::vector<unsigned int>& idxs, \
                                    const cv::Mat& pose1, const cv::Mat& pose2, \
                                    const cv::Mat& K, \
                                    std::vector<cv::Point3f>& triangulated_points) {
        
        // Initialization
        const unsigned int n_points = idxs.size();
        unsigned int n_triangulated_points = 0;
        triangulated_points.reserve(n_points);

        // Compute the inverse of the camera matrix
        // TODO: this computation can be avoided to be done each time
        const cv::Mat inv_K = K.inv();

        // Compute the projection matrix for camera 1 (with references)
        //  P1 = R1*K.inv()
        const cv::Mat P1 = pose1.rowRange(0,3).colRange(0,3)*inv_K;
        const float& P1_11 = P1.at<float>(0,0);
        const float& P1_12 = P1.at<float>(0,1);
        const float& P1_13 = P1.at<float>(0,2);
        const float& P1_21 = P1.at<float>(1,0);
        const float& P1_22 = P1.at<float>(1,1);
        const float& P1_23 = P1.at<float>(1,2);
        const float& P1_31 = P1.at<float>(2,0);
        const float& P1_32 = P1.at<float>(2,1);
        const float& P1_33 = P1.at<float>(2,2);

        // Compute the projection matrix for camera 2 (with references)
        //  P2 = R2*K.inv()
        const cv::Mat P2 = pose2.rowRange(0,3).colRange(0,3)*inv_K;
        const float& P2_11 = P2.at<float>(0,0);
        const float& P2_12 = P2.at<float>(0,1);
        const float& P2_13 = P2.at<float>(0,2);
        const float& P2_21 = P2.at<float>(1,0);
        const float& P2_22 = P2.at<float>(1,1);
        const float& P2_23 = P2.at<float>(1,2);
        const float& P2_31 = P2.at<float>(2,0);
        const float& P2_32 = P2.at<float>(2,1);
        const float& P2_33 = P2.at<float>(2,2);

        // Take references to the position of the origin of camera 1
        cv::Mat O1 = pose1.col(3).rowRange(0,3);
        const float& O1_x = O1.at<float>(0);
        const float& O1_y = O1.at<float>(1);
        const float& O1_z = O1.at<float>(2);

        // Take references to the position of the origin of camera 2
        cv::Mat O2 = pose2.col(3).rowRange(0,3);
        const float& O2_x = O2.at<float>(0);
        const float& O2_y = O2.at<float>(1);
        const float& O2_z = O2.at<float>(2);
        
        // Triangulate each couple of points
        for(unsigned int i=0; i<n_points; ++i) {

            // Take references to the current couple of points
            const float& p1_x = p_img1[matches[idxs[i]].queryIdx].pt.x;
            const float& p1_y = p_img1[matches[idxs[i]].queryIdx].pt.y;
            const float& p2_x = p_img2[matches[idxs[i]].trainIdx].pt.x;
            const float& p2_y = p_img2[matches[idxs[i]].trainIdx].pt.y;

            // Ensemble the matrix D as [-d1,d2] where d1 and d2 are
            // the rays starting from p1 and p2 respectively
            cv::Mat D = cv::Mat::eye(3,2,CV_32F);
            D.at<float>(0,0) = -(P1_11*p1_x + P1_12*p1_y + P1_13);
            D.at<float>(1,0) = -(P1_21*p1_x + P1_22*p1_y + P1_23);
            D.at<float>(2,0) = -(P1_31*p1_x + P1_32*p1_y + P1_33);
            D.at<float>(0,1) = (P2_11*p2_x + P2_12*p2_y + P2_13);
            D.at<float>(1,1) = (P2_21*p2_x + P2_22*p2_y + P2_23);;
            D.at<float>(2,1) = (P2_31*p2_x + P2_32*p2_y + P2_33);;

            // Compute the closest points on the two rays
            cv::Mat s = (-(D.t()*D)).inv()*(D.t()*(O2-O1));
            const float& s1 = s.at<float>(0,0);
            const float& s2 = s.at<float>(0,1);

            // Check if one of the closest is behind the camera 
            if(s1<0 || s2<0) {

                // It is invalid: an invalid point is assumed at position (0,0,0)
                // TODO: this assumption can be avoided
                triangulated_points.emplace_back(cv::Point3f(0,0,0));

                // Do not count it as good
                continue;
            }
            
            // Compute the 3D point as the middlepoint between the two closest points
            // on the rays (-D1 to re-do the minus computed before)
            triangulated_points.emplace_back( 
                cv::Point3f(0.5*((-D.at<float>(0,0)*s1+O1_x) + (D.at<float>(0,1)*s2+O2_x)), \
                            0.5*((-D.at<float>(1,0)*s1+O1_y) + (D.at<float>(1,1)*s2+O2_y)), \
                            0.5*((-D.at<float>(2,0)*s1+O1_z) + (D.at<float>(2,1)*s2+O2_z))) 
            );

            // Count it as good
            ++n_triangulated_points;
        
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



    /*
    * Given two poses and a list of 3D points seen in common between them, this function 
    * computes the parallax between the two poses.
    * Inputs:
    *   pose1/pose2
    *   landmarks: the list of 3D points contained in the state
    *   common_landmarks_idx: the list of predicted 3D points ids that are seen in common between
    *       the two poses
    * Outputs:
    *   parallax
    */
    float computeParallax(const cv::Mat& pose1, const cv::Mat& pose2, \
                                const std::vector<cv::Point3f>& landmarks, \
                                const std::vector<unsigned int>& common_landmarks_ids) {
        
        // Initialization
        const unsigned int n_points = common_landmarks_ids.size();
        std::vector<float> parallaxesCos;
        cv::Mat normal1 = cv::Mat::zeros(3,1,CV_32F);
        cv::Mat normal2 = cv::Mat::zeros(3,1,CV_32F);
        float dist1, dist2;

        // Compute the origin of the pose 1
        cv::Mat O1 = -pose1.rowRange(0,3).colRange(0,3).t() * \
                            pose1.rowRange(0,3).col(3);

        // Compute the origin of the pose2
        cv::Mat O2 = -pose2.rowRange(0,3).colRange(0,3).t() * \
                            pose2.rowRange(0,3).col(3);

        // For each point
        parallaxesCos.reserve(n_points);
        for(unsigned int i=0; i<n_points; ++i) {

            // Take the current 3D point
            const cv::Point3f& current_point = landmarks[common_landmarks_ids[i]];

            // Compute the normal origin-point for pose1
            normal1.at<float>(0,0) = current_point.x - O1.at<float>(0);
            normal1.at<float>(1,0) = current_point.y - O1.at<float>(1);
            normal1.at<float>(2,0) = current_point.z - O1.at<float>(2);

            // Compute the normal origin-point for pose2
            normal2.at<float>(0,0) = current_point.x - O2.at<float>(0);
            normal2.at<float>(1,0) = current_point.y - O2.at<float>(1);
            normal2.at<float>(2,0) = current_point.z - O2.at<float>(2);

            // Compute the distances pose-point
            dist1 = cv::norm(normal1);
            dist2 = cv::norm(normal2);

            // Compute the parallax cosine
            parallaxesCos.emplace_back( normal1.dot(normal2)/(dist1*dist2) );

        }
        parallaxesCos.shrink_to_fit();

        // Get the max parallax cosine and use it to compute the parallax
        std::sort(parallaxesCos.begin(), parallaxesCos.end());
        if(parallaxesCos.back() < 1)
            return std::acos(parallaxesCos.back())*180 / CV_PI;
        return 0;   // we cannot compute acos

    }
    
} // namespace SLucAM



// -----------------------------------------------------------------------------
// Representation conversion functions
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    /*
    * This function, given a quaternion represented as a cv::Mat with
    * dim 4x1 and with this structure: [x; y; z; w], returns the
    * corresponding rotation matrix R.
    */
    void quaternion_to_matrix(const cv::Mat& quaternion, cv::Mat& R) {

        // Initialization
        R = cv::Mat::zeros(3,3,CV_32F);

        // Create the Eigen quaternion
        Eigen::Quaternionf Eigen_quaternion;
        Eigen_quaternion.x() = quaternion.at<float>(0,0);
        Eigen_quaternion.y() = quaternion.at<float>(1,0);
        Eigen_quaternion.z() = quaternion.at<float>(2,0);
        Eigen_quaternion.w() = quaternion.at<float>(3,0);

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
    * dim 4x1 and with this structure: [x; y; z; w].
    */
    void matrix_to_quaternion(const cv::Mat& R, cv::Mat& quaternion) {

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
        quaternion.at<float>(0,0) = Eigen_quaternion.x();
        quaternion.at<float>(1,0) = Eigen_quaternion.y();
        quaternion.at<float>(2,0) = Eigen_quaternion.z();
        quaternion.at<float>(3,0) = Eigen_quaternion.w();
    }



    /*
    * This function, given a transformation matrix, returns the same matrix
    * represented as a SE3Quat for g2o.
    */
    g2o::SE3Quat transformation_matrix_to_SE3Quat(const cv::Mat& T_matrix) {
        Eigen::Matrix<double,3,3> R;
        R << T_matrix.at<float>(0,0), T_matrix.at<float>(0,1), T_matrix.at<float>(0,2),
            T_matrix.at<float>(1,0), T_matrix.at<float>(1,1), T_matrix.at<float>(1,2),
            T_matrix.at<float>(2,0), T_matrix.at<float>(2,1), T_matrix.at<float>(2,2);

        Eigen::Matrix<double,3,1> t(T_matrix.at<float>(0,3), \
                                    T_matrix.at<float>(1,3), \
                                    T_matrix.at<float>(2,3));

        return g2o::SE3Quat(R,t);
    }



    /*
    * This function, given a SE3Quat pose, returns the same matrix represented
    * as a OpenCV transformation matrix.
    */
    cv::Mat SE3Quat_to_transformation_matrix(const g2o::SE3Quat& se3quat) {
        cv::Mat T_matrix = cv::Mat::zeros(4,4,CV_32F);
        Eigen::Matrix<double,4,4> eigen_T_matrix = se3quat.to_homogeneous_matrix();
        for(int i=0;i<4;i++)
            for(int j=0; j<4; j++)
                T_matrix.at<float>(i,j) = eigen_T_matrix(i,j);
        return T_matrix;
    }



    /*
    * This function, given a Point3f (float) in OpenCV representation, returns the same
    * point in Eigen Matrix (double) representation: useful for g2o.
    */
    Eigen::Matrix<double,3,1> point_3d_to_vector_3d(const cv::Point3f& point) {
        Eigen::Matrix<double,3,1> v;
        v << point.x, point.y, point.z;
        return v;
    }



    /*
    * This function, given a Eigen 3x1 vector (double), returns the same vector
    * as a 3d point in OpenCV representation (Point3f).
    */
    cv::Point3f vector_3d_to_point_3d(const Eigen::Matrix<double,3,1>& vector) {
        return cv::Point3f(vector(0), vector(1), vector(2));
    }




    /*
    * This function, given a Keypoint (float) in OpenCV representation, returns the same
    * point in Eigen Matrix (double) representation: useful for g2o.
    */
    Eigen::Matrix<double,2,1> point_2d_to_vector_2d(const cv::KeyPoint& point) {
        Eigen::Matrix<double,2,1> v;
        v << point.pt.x, point.pt.y;
        return v;
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
    *                   is "better"
    *   matches: all matches between p_img1 and p_img2, with outliers
    *   idxs: we will consider only those points contained in 
    *           the matches vector, for wich we have indices in this vector idxs
    *   F: Foundamental Matrix from which to extract X
    *   K: camera matrix
    *   X: output Transformation Matrix extracted from F [R|t]
    *   triangulated_points: points triangulated between the first pose (assumed
    *       at the origin) and the pose X extracted from F
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
        const cv::Mat pose1 = cv::Mat::eye(4,4,CV_32F);

        // Extract the essential matrix and decompose it
        cv::Mat E = K.t()*F*K;
        cv::Mat u,w,vt;
        cv::SVD::compute(E,w,u,vt);

        // Extract the R matrix (2 solutions)
        cv::Mat R1 = u*W_mat*vt;
        if(cv::determinant(R1) < 0) {   // right handed condition
            R1 = -R1;
        }
        cv::Mat R2 = u*W_mat.t()*vt;
        if(cv::determinant(R2) < 0) {
            R2=-R2;
        }

        // Extract the t vector and "normalize" it
        cv::Mat t;
        u.col(2).copyTo(t);
        t=t/cv::norm(t);

        // Scale t vector
        t /= 4.4;

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
        current_score = triangulate_points(p_img1, p_img2, matches, idxs, \
                                    pose1, X_pred, K, \
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
        current_score = triangulate_points(p_img1, p_img2, matches, idxs, \
                                    pose1, X_pred, K, \
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
        current_score = triangulate_points(p_img1, p_img2, matches, idxs, \
                                    pose1, X_pred, K, \
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
        current_score = triangulate_points(p_img1, p_img2, matches, idxs, \
                                    pose1, X_pred, K, \
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
    *   guessed_pose: initial guess (and output)
    *   measurement: the measurement for which we need to predict the pose
    *   points_associations_filter: it will contain in position i true if
    *           the element in position i of the points_associations vector
    *           is an inlier, false otherwise
    *   points_associations: list of associations 2D point <-> 3D point
    *   landmarks: set of triangulated landmarks
    *   K: camera matrix
    *   n_iterations: #iterations to perform for Posit
    *   kernel_threshold: threshold for the outliers
    *   threshold_to_ignore: error threshold that determine if an outlier 
    *           is too outlier to be considered
    * Outputs:
    *   n_inliers of the last iteration
    */
    unsigned int perform_Posit(cv::Mat& guessed_pose, \
                                const Measurement& measurement, \
                                std::vector<bool>& points_associations_filter, \
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

        // Consider the pose of the world w.r.t. camera
        guessed_pose = invert_transformation_matrix(guessed_pose);

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
                                                img_cols, error, J)) {
                    points_associations_filter[obs_idx] = false;
                    continue;   // Discard not valid projections
                }
                    

                // Compute chi error
                const float& e_1 = error.at<float>(0,0);
                const float& e_2 = error.at<float>(1,0);
                current_chi = (e_1*e_1) + (e_2*e_2);

                // Deal with outliers
                if(current_chi > threshold_to_ignore){
                    points_associations_filter[obs_idx] = false;
                    continue;
                }

                // Robust kernel
                if(current_chi > kernel_threshold) {
                    error *= sqrt(kernel_threshold/current_chi);
                    current_chi = kernel_threshold;
                    points_associations_filter[obs_idx] = false;
                } else {
                    ++n_inliers[iter];
                    points_associations_filter[obs_idx] = true;
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

        // Go back in the representation of pose w.r.t. world
        guessed_pose = invert_transformation_matrix(guessed_pose);

        return n_inliers.back();

    }

} // namespace SLucAM