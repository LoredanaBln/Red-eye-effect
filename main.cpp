#include <opencv2/opencv.hpp>
#include "implementation/eye_detection.h"

using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("E:\\AN3\\SEM2\\pi\\red_eye\\images\\red8.jpg",
                       IMREAD_COLOR);

    Mat redMask = create_mask(image);

    imshow("Original Image", image);
    imshow("Red Mask", redMask);

    Mat circularMask = detect_circular_components(redMask, image, 0.6);
    imshow("Circular", circularMask);

    Mat corrected = correct_red_eye(circularMask, image);
    imshow("Corrected", corrected);
    waitKey(0);
    return 0;
}