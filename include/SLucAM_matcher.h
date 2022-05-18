//
// SLucAM_matcher.h
//
// This module describe the class useful to match points between two
// measurements in a flexible and general way.
//


#ifndef SLUCAM_MATCHER_H
#define SLUCAM_MATCHER_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>
#include <SLucAM_measurement.h>
#include <SLucAM_keypoint.h>



// -----------------------------------------------------------------------------
// Matcher class
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This class represent a matcher that we need to match points between
    * two images. 
    */
    class Matcher {

    public:

        Matcher(const std::string& feat_types);

        void match_measurements(const Measurement& meas1, \
                                const Measurement& meas2,  
                                std::vector<cv::DMatch>& matches, \
                                const float& match_threshold=30);   // match_threshold=15 for ANMS, 30 for ORB

        static int compute_descriptors_distance(const cv::Mat& d1, \
                                                const cv::Mat& d2);
        
        static int compute_descriptors_distance(const cv::Mat& d1, \
                                                const std::vector<cv::Mat>& d2_set);

    private:

        cv::Ptr<cv::BFMatcher> _bf_matcher;

    }; // class Matcher

} // namespace SLucAM



#endif // SLUCAM_MATCHER_H