//
// SLucAM_image.h implementation
//


// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------
#include <SLucAM_image.h>
#include <opencv2/highgui.hpp>



// -----------------------------------------------------------------------------
// Implementation of loading and saving utilities
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Load an image, it also check for the consistency of the inputs,
    * in case of errors it returns "false".
    * Inputs:
    *   filename: relative path of the image to load
    *   img: OpenCV Matrix object where the image will be loaded
    */
    bool load_image(const std::string& filename, cv::Mat& img) {
        img = cv::imread(filename, cv::IMREAD_COLOR);
        if (img.empty()) {
            return false;
        }
        return true;
    }

} // namespace SLucAM



// -----------------------------------------------------------------------------
// Implementation of visualization utilities
// -----------------------------------------------------------------------------
namespace SLucAM {

    /*
    * Useful to visualize the matches between two images.
    */
    void visualize_matches(const cv::Mat& img1, const cv::Mat& img2, \
                        const std::vector<cv::KeyPoint>& keypoints1, \
                        const std::vector<cv::KeyPoint>& keypoints2, \
                        const std::vector<cv::DMatch>& matches, \
                        const std::string image_name) {
        cv::Mat img_matches;
        cv::drawMatches(img1, keypoints1, img2, keypoints2,
                    matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
        cv::imshow(image_name, img_matches);
        cv::waitKey(0);
    }

} // namespace SLucAM