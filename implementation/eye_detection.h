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

image_channels_bgr break_channels(Mat source);
image_channels_hsv bgr_to_hsv(image_channels_bgr bgr_channels);
Mat create_red_mask(image_channels_hsv hsv_channels);

#endif //RED_EYE_EYE_DETECTION_H
