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

        const cv::Mat& getDescriptors() const \
                {return this->_descriptors;};

        cv::Mat getDescriptor(const unsigned int& idx) const;

        const unsigned int getId() const \
                {return this->_meas_id;};
        
        const std::string getImgName() const \
                {return this->_img_filename;};

        const void setImgName(const std::string& img_filename) \
                {this->_img_filename = img_filename;};

        // Take note of the next measurement id to use
        static unsigned int _next_id;

    private:

        // The set of points in the image
        std::vector<cv::KeyPoint> _points;

        // The set of descriptors, one per point in the image
        cv::Mat _descriptors;

        // The id of the current measurement
        unsigned int _meas_id;

        // The name of the image which this measurement refers to,
        // if any
        std::string _img_filename;

    }; // class Measurement

} // namespace SLucAM



#endif // SLUCAM_MEASUREMENT_H