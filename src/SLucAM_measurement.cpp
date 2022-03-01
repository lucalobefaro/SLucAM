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

    /*
    * Constructor of the class Measurement. It takes the image to which the 
    * measurement refers to and the detector to use to extract the keypoints
    * and generate all the information needed about the image.
    * Inputs:
    *   filename: path of the image
    *   detector: tool to extract the keypoints
    */
    Measurement::Measurement(const std::string filename, \
                            const cv::Ptr<cv::Feature2D>& detector) {
        
        // Load the image
        this->_img_name = filename;
        cv::Mat img;
        SLucAM::load_image(filename, img); // TODO: implement error in loading (false)
        
        // Detect keypoints
        detector->detectAndCompute(img, cv::Mat(), this->_points, this->_descriptors);

        // Normalize points
        SLucAM::normalize_points(this->_points, this->_normalized_points, this->_T_norm);
    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of utilities for the measurements
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Function that, given two measurement, determime which keypoints are
    * the same between the two.
    * Inputs:
    *   meas1/meas2: the two measurements to match
    *   matches: vector where to store the matches
    *   matcher: an instance of the BFMatcher that will perform the match 
    *             between keypoints
    */
    void match_measurements(const Measurement& meas1, const Measurement& meas2, \
                            std::vector<cv::DMatch>& matches, \
                            const cv::BFMatcher& matcher) {
        matcher.match(meas1.getDescriptors(), meas2.getDescriptors(), matches);
        // TODO: decide if we want this or not:
        //std::sort(matches.begin(), matches.end());
        //while (matches.size() > 1000) {
        //    matches.pop_back();
        //}
    }

} // namespace SLucAM