//
// Created by Loredana on 03-Apr-25.
//

#ifndef RED_EYE_EYE_DETECTION_H
#define RED_EYE_EYE_DETECTION_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

struct image_channels_bgr {
    Mat B, G, R;
};

struct image_channels_hsv {
    Mat H, S, V;
};

struct labels_ {
    Mat labels;
    int no_newlabels;
};

const int dx[8] = {-1,-1,-1,0,0,1,1,1};
const int dy[8] = {-1,0,1,-1,1,-1,0,1};

image_channels_bgr break_channels(Mat source);
image_channels_hsv bgr_to_hsv(image_channels_bgr bgr_channels);
Mat create_red_mask(image_channels_hsv hsv_channels);
Mat correct_red_eye(Mat mask, Mat source);
Mat detect_circular_components(Mat binary, double circularityThreshold);

#endif //RED_EYE_EYE_DETECTION_H
