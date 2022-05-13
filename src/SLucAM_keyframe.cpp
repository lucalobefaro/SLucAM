//
// SLucAM_keyframe.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_keyframe.h>



// -----------------------------------------------------------------------------
// Implementation of Keyframe class methods
// -----------------------------------------------------------------------------
namespace SLucAM {


    /*
    * Basic constructor
    */
    Keyframe::Keyframe(const unsigned int& meas_idx, \
                        const unsigned int& pose_idx, \
                        std::vector<std::pair<unsigned int, unsigned int>>& points_associations, \
                        const unsigned int& n_points_meas) {
        this->_meas_idx = meas_idx;
        this->_pose_idx = pose_idx;
        this->_points_associations = points_associations;
        this->_points_associations.reserve(n_points_meas);
    }



    /*
    * Given a point, returns the corresponding landmark. If no association
    * for such point is present, return -1.
    */
    const int Keyframe::point2Landmark(const unsigned int& point_idx) const {
        const unsigned int& n_associations = this->_points_associations.size();
        for(unsigned int i=0; i<n_associations; ++i) {
            const std::pair<unsigned int, unsigned int>& current_association = \
                    this->_points_associations[i];
            if(current_association.first == point_idx) {
                return current_association.second;
            }
        }
        return -1;
    }



    /* 
    * This function returns the list of ids of the points seen from the 
    * keyframe.
    */
    unsigned int Keyframe::getObservedPoints(std::vector<unsigned int>& ids) const {
        ids.clear();
        ids.reserve(this->_points_associations.size());
        for(const auto& el: this->_points_associations)
            ids.emplace_back(el.second);
        ids.shrink_to_fit();
        return ids.size();
    }



    /*
    * This function takes a set of ids and adds to it all the ids of the 3D points
    * observed by the keyframe. Then returns the number of seen 3D point;
    */
    unsigned int Keyframe::addObservedPointsSet(std::set<unsigned int>& ids_set) const {
        unsigned int n_added_ids = 0;
        for(const auto& el: this->_points_associations) {
            ids_set.insert(el.second);
            ++n_added_ids;
        }
        return n_added_ids;
    }

} // namespace SLucAM