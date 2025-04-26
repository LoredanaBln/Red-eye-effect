#include "eye_detection.h"

using namespace std;
using namespace cv;

image_channels_bgr break_channels(Mat source) {
    int rows = source.rows;
    int cols = source.cols;
    Mat B = Mat(rows, cols, CV_8UC1);
    Mat G = Mat(rows, cols, CV_8UC1);
    Mat R = Mat(rows, cols, CV_8UC1);
    image_channels_bgr bgr_channels;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b pixel = source.at<Vec3b>(i, j);
            B.at<uchar>(i, j) = pixel[0];
            G.at<uchar>(i, j) = pixel[1];
            R.at<uchar>(i, j) = pixel[2];
        }
    }

    bgr_channels.B = B;
    bgr_channels.G = G;
    bgr_channels.R = R;

    return bgr_channels;
}

image_channels_hsv bgr_to_hsv(image_channels_bgr bgr_channels) {
    int rows = bgr_channels.B.rows;
    int cols = bgr_channels.B.cols;
    Mat H = Mat::zeros(rows, cols, CV_32FC1);
    Mat S = Mat::zeros(rows, cols, CV_32FC1);
    Mat V = Mat::zeros(rows, cols, CV_32FC1);
    image_channels_hsv hsv_channels;

    float M, m, C;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float r = (float) bgr_channels.R.at<uchar>(i, j) / 255;
            float g = (float) bgr_channels.G.at<uchar>(i, j) / 255;
            float b = (float) bgr_channels.B.at<uchar>(i, j) / 255;

            M = max(r, max(g, b));
            m = min(r, min(g, b));
            C = M - m;

            V.at<float>(i, j) = M;
            if (M != 0.0) {
                S.at<float>(i, j) = C / M;
            } else {
                S.at<float>(i, j) = 0.0;
            }

            if (C != 0.0) {
                if (M == r) {
                    H.at<float>(i, j) = 60 * (g - b) / C;
                }
                if (M == g) {
                    H.at<float>(i, j) = 120 + 60 * (b - r) / C;
                }
                if (M == b) {
                    H.at<float>(i, j) = 240 + 60 * (r - g) / C;
                }
            } else {
                H.at<float>(i, j) = 0.0;
            }
            if (H.at<float>(i, j) < 0.0) {
                H.at<float>(i, j) += 360;
            }
        }
    }

    hsv_channels.H = H;
    hsv_channels.S = S;
    hsv_channels.V = V;

    return hsv_channels;
}

Mat create_red_mask(image_channels_hsv hsv_channels) {
    Mat mask = Mat::zeros(hsv_channels.H.size(), CV_8UC1);

    for (int i = 0; i < hsv_channels.H.rows; i++) {
        for (int j = 0; j < hsv_channels.H.cols; j++) {
            float hue = hsv_channels.H.at<float>(i, j);
            float sat = hsv_channels.S.at<float>(i, j) * 255;
            float val = hsv_channels.V.at<float>(i, j) * 255;

            if ((hue < 10 || hue > 170) && sat > 100 && val > 50) {
                mask.at<uchar>(i, j) = 255;
            }
        }
    }
    return mask;
}
