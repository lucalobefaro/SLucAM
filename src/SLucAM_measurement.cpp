//
// SLucAM_measurement.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_measurement.h>
#include <SLucAM_image.h>
#include <SLucAM_geometry.h>



// -----------------------------------------------------------------------------
// Implementation of the measurement class methods
// -----------------------------------------------------------------------------
namespace SLucAM {

    // Initialize the static member
    unsigned int Measurement::_next_id = 0;

    /*
    * Constructor of the class Measurement. It takes the vector of points
    * for a given measurements and the relative image descriptors. It
    * normalizes the points also
    */
    Measurement::Measurement(std::vector<cv::KeyPoint>& points, \
                            cv::Mat& descriptors) {
        
        // Store the points
        this->_points = points;
        this->_descriptors = descriptors;

        // Assign the id to the measurement
        this->_meas_id = Measurement::_next_id;
        Measurement::_next_id++;
    }



    /*
    * Gives a descriptor of a point, determined by its idx.
    */
    cv::Mat Measurement::getDescriptor(const unsigned int& idx) const {
        return this->_descriptors.row(idx);
    }

} // namespace SLucAM