#include <opencv2/opencv.hpp>
#include "implementation/eye_detection.h"
using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("E:\\AN3\\SEM2\\pi\\red_eye\\images\\red1.jpg",
                       IMREAD_COLOR);

    image_channels_bgr channelsBgr = break_channels(image);
    image_channels_hsv channelsHsv = bgr_to_hsv(channelsBgr);

    Mat redMask = create_red_mask(channelsHsv);

    imshow("Original Image", image);
    imshow("Red Mask", redMask);

    Mat extracted = detect_circular_components(redMask, 0.4);
    imshow("Circular", extracted);

    Mat corrected = correct_red_eye(extracted, image);
    imshow("Corrected", corrected);
    waitKey(0);
    return 0;
}