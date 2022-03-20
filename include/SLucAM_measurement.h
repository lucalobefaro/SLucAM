//
// SLucAM_measurement.h
//
// In this module we have the class to deal with measurements.
//


#ifndef SLUCAM_MEASUREMENT_H
#define SLUCAM_MEASUREMENT_H

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <opencv2/features2d.hpp>



// -----------------------------------------------------------------------------
// Measurement class
// -----------------------------------------------------------------------------
namespace SLucAM {
    
    /*
    * This class represent a single measurement, so it mantains the features 
    * extracted from an image.
    */
    class Measurement {
    
    public:
        
        Measurement(std::vector<cv::KeyPoint>& points, \
                    cv::Mat& descriptors);

        const std::vector<cv::KeyPoint>& getPoints() const \
                {return this->_points;};

        const std::vector<cv::KeyPoint>& getNormalizedPoints() const \
                {return this->_normalized_points;};

        const cv::Mat& getDescriptors() const \
                {return this->_descriptors;};

        const cv::Mat& getTNorm() const \
                {return this->_T_norm;};

        const unsigned int getId() const \
                {return this->_meas_id;};

        // Take note of the next measurement id to use
        static unsigned int _next_id;

    private:

        // The set of points in the image and the same point normalized
        std::vector<cv::KeyPoint> _points;
        std::vector<cv::KeyPoint> _normalized_points;

        // The set of descriptors, one per point in the image
        cv::Mat _descriptors;

        // The matrix used to normalize the points
        cv::Mat _T_norm;

        // The id of the current measurement
        unsigned int _meas_id;

    }; // class Measurement

} // namespace SLucAM



#endif // SLUCAM_MEASUREMENT_H