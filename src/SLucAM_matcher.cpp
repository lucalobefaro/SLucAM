//
// SLucAM_matcher.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_matcher.h>

// TODO: delete this
#include <iostream>
using namespace std;




// -----------------------------------------------------------------------------
// Implementation of the Matcher class methods
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Default constructor of the Matcher class.
    */
    Matcher::Matcher() {
        this->_bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    }


    /*
    * Function that, given two measurement, determime which keypoints are
    * the same between the two.
    * Inputs:
    *   meas1/meas2: the two measurements to match
    *   matches: vector where to store the matches
    */
   void Matcher::match_measurements(const Measurement& meas1, \
                                const Measurement& meas2,  
                                std::vector<cv::DMatch>& matches, \
                                const float& match_threshold) {
                                    
        // --- DEFAULT MATCHER ---

        if(this->_use_default_matcher) {
            std::vector<cv::DMatch> unfiltered_matches;
            this->_bf_matcher->match(meas1.getDescriptors(), \
                                    meas2.getDescriptors(), \
                                    unfiltered_matches);
            
            // Filter matches
            matches.reserve(unfiltered_matches.size());
            for(auto& m : unfiltered_matches) {
                if(m.distance <= match_threshold) 
                    matches.emplace_back(m);
            }
            matches.shrink_to_fit();

            return;
        }

        // --- POINTS IDS MATCHER ---
        // Initialization
        unsigned int meas1_idx = meas1.getId(); 
        unsigned int meas2_idx = meas2.getId();
        const std::vector<unsigned int>& gt_points_meas1 = \
                this->_points_ids[meas1_idx];
        const std::vector<unsigned int>& gt_points_meas2 = \
                this->_points_ids[meas2_idx];
        unsigned int n_points_meas1 = meas1.getPoints().size();
        unsigned int n_points_meas2 = meas2.getPoints().size();
        unsigned int current_gt_meas1, current_gt_meas2;

        matches.reserve(n_points_meas1);
        for(unsigned int current_p_meas1=0; \
                current_p_meas1<n_points_meas1; \
                ++current_p_meas1) {
    
            // Get the current point grounf truth id
            current_gt_meas1 = gt_points_meas1[current_p_meas1];

            for(unsigned int current_p_meas2=0; \
                current_p_meas2<n_points_meas2; \
                ++current_p_meas2) {
            
                // Get the current point grounf truth id
                current_gt_meas2 = gt_points_meas2[current_p_meas2];

                // If they match, save the match and go to the next 
                // point
                if(current_gt_meas1 == current_gt_meas2) {
                    matches.emplace_back();
                    matches.back().queryIdx = current_p_meas1;
                    matches.back().trainIdx = current_p_meas2;
                    break;
                }

            }
        }
        matches.shrink_to_fit();

    }

} // namespace SLucAM