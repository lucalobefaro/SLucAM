//
// SLucAM_keypoint.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_keypoint.h>
#include <SLucAM_matcher.h>
#include <algorithm>
#include <iostream>



// -----------------------------------------------------------------------------
// Implementation of Keypoint class methods
// -----------------------------------------------------------------------------
namespace SLucAM {


    /*
    * This function find, among all the descriptors of the points that
    * observe the current keypoint, the descriptor that has the minor
    * median distance to the rest of keypoints.
    */
    void Keypoint::updateDescriptor(const std::vector<Keyframe>& keyframes, \
                                    const std::vector<Measurement>& measurements) {

        // Initialization
        const unsigned int n_observers = this->_observers.size();
        std::vector<std::vector<int>> distances(n_observers, \
                                            std::vector<int>(n_observers, 0));
        int best_distance = INT_MAX;
        int current_distance;
        unsigned int best_idx = 0;

        // Compute mutual distances
        for(unsigned int i=0; i<n_observers; ++i) {
            for(unsigned int j=0; j<n_observers; ++j) {
                distances[i][j] = Matcher::compute_descriptors_distance(\
                        this->getObserverDescriptor(keyframes, measurements, i), 
                        this->getObserverDescriptor(keyframes, measurements, j));
            }
        }
        
        // Save as representative descriptor the descriptor with the
        // lower median distance
        for(unsigned int i=0; i<n_observers; ++i){
            std::vector<int>& row = distances[i];
            std::sort(row.begin(), row.end());
            current_distance = row[n_observers/2];
            if(current_distance < best_distance) {
                best_distance = current_distance;
                best_idx = i;
            }
        }
        this->_descriptor = this->getObserverDescriptor(keyframes, measurements, \
                                                        best_idx);

    }


    /*
    * This function, given the idx of a keyframe, deletes from the current
    * keypoint all the observations from such keyframe.
    * Outputs:
    *   number of observers deleted
    */
    unsigned int Keypoint::deleteObservers(const unsigned int& keyframe_idx) {

        // Initialization
        const unsigned int n_observers = this->_observers.size();

        // Create a vector that will contain all the observers
        std::vector<std::pair<unsigned int, unsigned int>> old_observers;
        old_observers.swap(this->_observers);

        // Refill the original vector by ignoring the observations made
        // from the given keyframe
        this->_observers.reserve(n_observers);
        for(unsigned int i=0; i<n_observers; ++i) {
            if(old_observers[i].first != keyframe_idx) 
                this->_observers.emplace_back(old_observers[i]);
        }
        this->_observers.shrink_to_fit();

        // Count how many observations we loose
        return n_observers-this->_observers.size();
    } 



    /*
    * This function, given an idx of an observer, returns the associated
    * descriptor, that is the descriptor of the observer point in the 
    * measurement associated to the observer keyframe.
    */
    const cv::Mat Keypoint::getObserverDescriptor(const std::vector<Keyframe>& keyframes, \
                                                const std::vector<Measurement>& measurements, \
                                                const unsigned int& idx) {
        return measurements[keyframes[this->_observers[idx].first]\
                                .getMeasIdx()].getDescriptor(this->_observers[idx].second);
    }
    
} // namespace SLucAM