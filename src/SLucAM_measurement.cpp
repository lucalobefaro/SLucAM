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

        // Normalize points
        normalize_points(this->_points, this->_normalized_points, this->_T_norm);

        // Assign the id to the measurement
        this->_meas_id = Measurement::_next_id;
        Measurement::_next_id++;
    }

} // namespace SLucAM