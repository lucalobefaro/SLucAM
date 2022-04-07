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
    bool visualize_matches(const Measurement& meas1, const Measurement& meas2, \
                        const std::vector<cv::DMatch>& matches, \
                        const std::vector<unsigned int>& matches_filter) {
        
        // Get the names of the imgs to visualize
        const std::string& img1_filename = meas1.getImgName();
        const std::string& img2_filename = meas2.getImgName();

        // Load the images
        cv::Mat img1, img2;
        if(!load_image(img1_filename, img1)) return false;
        if(!load_image(img2_filename, img2)) return false;

        // Filter out the matches
        const unsigned int n_matches = matches_filter.size();
        std::vector<cv::DMatch> filtered_matches;
        filtered_matches.reserve(n_matches);
        for(unsigned int i=0; i<n_matches; ++i) {
            filtered_matches.emplace_back(matches[matches_filter[i]]);
        }
        filtered_matches.shrink_to_fit();

        cv::Mat img_matches;
        cv::drawMatches(img1, meas1.getPoints(), img2, meas2.getPoints(),
                    filtered_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::namedWindow("MATCHED IMGS", cv::WINDOW_AUTOSIZE);
        cv::imshow("MATCHED IMGS", img_matches);
        cv::waitKey(0);

        return true;
    }

} // namespace SLucAM