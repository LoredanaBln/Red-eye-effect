//
// Created by Loredana on 03-Apr-25.
//

#ifndef RED_EYE_EYE_DETECTION_H
#define RED_EYE_EYE_DETECTION_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

struct labels_
{
    Mat labels;
    int no_newlabels;
};

struct HSV
{
    float h;
    float s;
    float v;
};

const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

Mat create_mask(Mat source);
Mat correct_red_eye(Mat mask, Mat source);
Mat detect_circular_components(Mat binary, Mat original, double circularityThreshold);
Mat dilate(Mat source, int no_iter);
Mat erode(Mat source, int no_iter);
HSV rgb_to_hsv_pixel(float r, float g, float b);

#endif // RED_EYE_EYE_DETECTION_H
