//
// SLucAM_measurement.h
//
// In this module we have all the function to deal with measurements, we also
// have the implementation of the Measurement class.
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

        Measurement(const std::string filename, \
                    const cv::Ptr<cv::Feature2D>& detector);

        const std::string& getName() const {return this->_img_name;};

        const std::vector<cv::KeyPoint>& getPoints() const {return this->_points;};

        const std::vector<cv::KeyPoint>& getNormalizedPoints() const {return this->_normalized_points;};

        const cv::Mat& getDescriptors() const {return this->_descriptors;};

        const cv::Mat& getTNorm() const {return this->_T_norm;};

    private:

        // The path of the image to wich this measurement refers to
        std::string _img_name;

        // The set of points in the image and the same point normalized
        std::vector<cv::KeyPoint> _points;
        std::vector<cv::KeyPoint> _normalized_points;

        // The set of descriptors, one per point in the image
        cv::Mat _descriptors;

        // The matrix used to normalize the points
        cv::Mat _T_norm;

    }; // class MEasurement

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Utilities for the measurements
// -----------------------------------------------------------------------------
namespace SLucAM {
    void match_measurements(const Measurement& meas1, const Measurement& meas2, \
                            std::vector<cv::DMatch>& matches, \
                            const cv::BFMatcher& matcher);
} // namespace SLucAM



#endif // SLUCAM_MEASUREMENT_H