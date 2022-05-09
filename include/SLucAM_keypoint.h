//
// SLucAM_keyopint.h
//
// In this module we have all the function to deal with the concept of keypoint.
//


#ifndef SLUCAM_KEYPOINT_H
#define SLUCAM_KEYPOINT_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <vector>
#include <opencv2/features2d.hpp>
#include <SLucAM_keyframe.h>
#include <SLucAM_measurement.h>
#include <iostream>



// -----------------------------------------------------------------------------
// Keypoint class
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * This class contains a point in the world. It models the position in the 
    * world frame and mantain a list of references to the couples <keyframe,
    * 2D point> that indicates which point from which keyframe refers
    * to this keypoint. In this way we can obtain a list of descriptors, one
    * for each 2D point connected to it.
    */
    class Keypoint {

    public:

        Keypoint(const cv::Point3f& position) {
            this->_position = position;
        }

        const cv::Point3f& getPosition() const {return this->_position;}

        void setPosition(const cv::Point3f& new_position) {this->_position = new_position;}

        const std::vector<std::pair<unsigned int, unsigned int>>& getObservers() const {
            return this->_observers;
        }

        const cv::Mat& getDescriptor() const {return this->_descriptor;}

        void addObserver(const unsigned int& keyframe_idx, const unsigned int& point_idx) {
            this->_observers.emplace_back(keyframe_idx, point_idx);
        }

        unsigned int deleteObservers(const unsigned int& keyframe_idx);

        void updateDescriptor(const std::vector<Keyframe>& keyframes, \
                                const std::vector<Measurement>& measurements);

        const cv::Mat getObserverDescriptor(const std::vector<Keyframe>& keyframes, \
                                                const std::vector<Measurement>& measurements, \
                                                const unsigned int& idx);
    
    private:

        // The position in world frame of the keypoint
        cv::Point3f _position; 

        // The most representative descriptor
        cv::Mat _descriptor;

        // List of couples <keyframe idx, 2D point idx> that observes this keypoint
        std::vector<std::pair<unsigned int, unsigned int>> _observers;

    };

} // namespace SLucAM


#endif // SLUCAM_KEYPOINT_H