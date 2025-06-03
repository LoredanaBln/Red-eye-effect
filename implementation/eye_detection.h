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

struct gradients_structure
{
    Mat x;
    Mat y;
    Mat magnitude;
    Mat direction;
};

struct filter_structure
{
    int *filter_x;
    int *filter_y;
    int *di;
    int *dj;
};

const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

Mat create_mask(Mat source);
Mat correct_red_eye(Mat mask, Mat source);
Mat detect_circular_components(Mat binary, Mat original, double circularityThreshold);
Mat dilate(Mat source, int no_iter);
Mat erode(Mat source, int no_iter);
HSV rgb_to_hsv_pixel(float r, float g, float b);

vector<float> compute_kernel_1D(int kernel_size);
Mat apply_gaussian_filtering_1D(Mat source, int kernel_size);
gradients_structure compute_gradients(Mat source, const int *filter_x, const int *filter_y, const int *di, const int *dj);
int getArea(float direction);
Mat non_maxima_gradient_suppression(gradients_structure gradient);
filter_structure get_filter(string filter_type);
Mat normalize_supression(Mat supression, string filter_type);
int adaptive_threshold(Mat magnitude, float p, bool verbose);
Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose);
int *compute_histogram_naive(Mat source);

#endif // RED_EYE_EYE_DETECTION_H
