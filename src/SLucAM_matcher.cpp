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
    * Default constructor of the Matcher class. If we use ORB features we use
    * NORM_HAMMING distance for matching, otherwise we use NORM_L2.
    */
    Matcher::Matcher(const std::string& feat_types) {
        if(feat_types == "orb")
            this->_bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        else
            this->_bf_matcher = cv::BFMatcher::create(cv::NORM_L2, true);
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



    /*
    * This function, given two ORB descriptors (d1, d2) computes the distance
    * between them using the bit set count operation from:
    * http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    */
    int Matcher::compute_descriptors_distance(const cv::Mat& d1, \
                                                const cv::Mat& d2) {
    
        const int *d1_ptr = d1.ptr<int32_t>();
        const int *d2_ptr = d2.ptr<int32_t>();

        int distance = 0;

        for(int i=0; i<8; i++, d1_ptr++, d2_ptr++)
        {
            unsigned  int v = *d1_ptr ^ *d2_ptr;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            distance += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return distance;

    }



    /*
    * This function computes the distance between a single descriptor and a set of
    * descriptors and returns the nearest distance computed.
    */
    int Matcher::compute_descriptors_distance(const cv::Mat& d1, \
                                                const std::vector<cv::Mat>& d2_set) {
    
        // Initialization
        unsigned int best_distance = 10000;
        unsigned int current_distance;
        const unsigned int n_descriptors = d2_set.size();

        // Search for the best distance
        for(unsigned int i=0; i<n_descriptors; ++i) {
            current_distance = compute_descriptors_distance(d1, d2_set[i]);
            if(current_distance < best_distance) 
                best_distance = current_distance;
        }

        return best_distance;
    }

} // namespace SLucAM