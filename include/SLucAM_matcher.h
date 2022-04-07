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



// -----------------------------------------------------------------------------
// Matcher class
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This class represent a matcher that we need to match points between
    * two images. By default it is initialized with a BFMatcher (in this case
    * we will use it when we need to match two measurements) or with
    * a list of ids.
    * In the last case we have in position i, the list of ids for each point
    * in the measurement i. In this way we can use this ids to understand
    * which measured points referes to the same 3D world point of another 
    * measured point.
    */
    class Matcher {

    public:

        Matcher();

        Matcher(std::vector<std::vector<unsigned int>>& points_ids) :\
            _points_ids(points_ids),
            _use_default_matcher(false)
        {};

        void match_measurements(const Measurement& meas1, \
                                const Measurement& meas2,  
                                std::vector<cv::DMatch>& matches, \
                                const float& match_threshold=50);
    
    private:

        bool _use_default_matcher = true;

        cv::BFMatcher _bf_matcher;

        std::vector<std::vector<unsigned int>> _points_ids;

    }; // class Matcher

} // namespace SLucAM



#endif // SLUCAM_MATCHER_H