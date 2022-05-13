//
// SLucAM_keyframe.h
//
// In this module we have all the function to deal with the concept of keyframe.
//


#ifndef SLUCAM_KEYFRAME_H
#define SLUCAM_KEYFRAME_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <vector>
#include <iostream>
#include <set>



// -----------------------------------------------------------------------------
// Keyframe class
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    /*
    * This class take note of each KeyFrame in the state. Basically it contains
    * informations about:
    *   - which measurement in the state is the measurement from which
    *       this keyframe is taken
    *   - which pose in the state is the predicted pose for this keyframe
    *   - a vector of Map <point_idx, landmark_idx> that associates at each point
    *       in the measurement to which this keyframe refers, the 3D 
    *       predicted landmark in the state
    *   - a vector that contains all the keyframe(poses) that the current Keyframe 
    *       observes
    *   - a vector that contains all the keyframes(poses) that observes the current
    *       keyframe
    */
    class Keyframe {

    public:

        Keyframe(const unsigned int& meas_idx, \
                const unsigned int& pose_idx, \
                std::vector<std::pair<unsigned int, unsigned int>>& points_associations, \
                const unsigned int& n_points_meas);

        const int point2Landmark(const unsigned int& point_idx) const;

        void addPointAssociation(const unsigned int& p_idx, const unsigned int& l_idx) {
            this->_points_associations.emplace_back(p_idx, l_idx);
        }

        void addPointsAssociations(std::vector<std::pair<unsigned int, unsigned int>>& \
                                        new_points_associations){
            this->_points_associations.insert(this->_points_associations.end(), \
                new_points_associations.begin(), new_points_associations.end()); 
        }

        void addKeyframeObserved(const int& pose_idx) {
            this->_keyframes_observed.emplace_back(pose_idx);
        }

        void addObserverKeyframe(const int& pose_idx) {
            this->_observers_keyframes.emplace_back(pose_idx);
        }

        const unsigned int& getPoseIdx() const {return this->_pose_idx;}

        const unsigned int& getMeasIdx() const {return this->_meas_idx;}

        const std::vector<std::pair<unsigned int, unsigned int>>& \
                    getPointsAssociations() const {
            return this->_points_associations;
        }

        unsigned int getObservedPoints(std::vector<unsigned int>& ids) const;

        unsigned int addObservedPointsSet(std::set<unsigned int>& ids_set) const;

        const std::vector<unsigned int>& getKeyframesObserved() const {
            return this->_keyframes_observed;
        }

        const std::vector<unsigned int>& getObserversKeyframes() const {
            return this->_observers_keyframes;
        }

        friend std::ostream& operator<< (std::ostream& out, const Keyframe& data) {

            const unsigned int n_points_associations = data._points_associations.size();
            const unsigned int n_keyframes_associations = data._keyframes_observed.size();

            out << "MEASUREMENT IDX: " << data._meas_idx << std::endl;
            out << "POSE IDX: " << data._pose_idx << std::endl;

            out << "N. POINTS ASSOCIATED: " << n_points_associations << std::endl;
            out << "POINTS ASSOCIATION <2d point idx : 3d point idx>: " << std::endl;
            for(unsigned int i=0; i<n_points_associations; ++i) {
                out << "\t[" << data._points_associations[i].first << \
                    " : " << data._points_associations[i].second << "]" << std::endl;
            }
            out << "OBSERVED KEYFRAMES IDS: ";
            for(unsigned int i=0; i<n_keyframes_associations; ++i) {
                out << "[" << data._keyframes_observed[i] << "] ";
            }
            out << std::endl;

            return out;
        }

    private:

        // Idx of the measure of the keyframe (referred to the list of measurements
        // in the state)
        unsigned int _meas_idx;

        // Idx of the pose of the keyframe (referred to the list of poses
        // in the state)
        unsigned int _pose_idx;

        // List of associations 2D point <-> 3D point (referred to the 2D points
        // in the corresponding measurement and the 3D keypoint in the state)
        std::vector<std::pair<unsigned int, unsigned int>> _points_associations;
        
        // List of keyframes that this observes
        std::vector<unsigned int> _keyframes_observed;

        // List of keyframes that observes this
        std::vector<unsigned int> _observers_keyframes;

    };

} // namespace SLucAM



#endif // SLUCAM_KEYFRAME_H