#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

void houghCircle(InputArray image, OutputArray circles, int method, double dp,
                 double minDist, double param1, double param2,
                 int minRadius, int maxRadius)
{
    // Convert input image to grayscale if it's not already
    Mat gray;
    if (image.channels() > 1)
    {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = image.getMat();
    }

    // Apply Gaussian blur to reduce noise
    Mat blurred;
    GaussianBlur(gray, blurred, Size(9, 9), 2, 2);

    // Edge detection using Canny
    Mat edges;
    Canny(blurred, edges, param1, param2);

    // Initialize output vector
    vector<Vec3f> detectedCircles;

    // Accumulator array dimensions
    int rows = blurred.rows;
    int cols = blurred.cols;
    int maxRadiusPixels = maxRadius;
    int minRadiusPixels = minRadius;

    // Create accumulator array with higher precision
    vector<vector<vector<float>>> accumulator(rows, vector<vector<float>>(cols, vector<float>(maxRadiusPixels - minRadiusPixels + 1, 0.0f)));

    // Edge pixel detection with gradient information
    Mat dx, dy;
    Sobel(blurred, dx, CV_32F, 1, 0);
    Sobel(blurred, dy, CV_32F, 0, 1);

    // Calculate total edge pixels for normalization
    int totalEdgePixels = 0;
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            if (edges.at<uchar>(y, x) > 0)
            {
                totalEdgePixels++;
            }
        }
    }

    // Edge pixel detection with gradient voting
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            if (edges.at<uchar>(y, x) > 0)
            {
                float gx = dx.at<float>(y, x);
                float gy = dy.at<float>(y, x);
                float magnitude = sqrt(gx * gx + gy * gy);

                if (magnitude > 0)
                {
                    // Normalize gradient
                    gx /= magnitude;
                    gy /= magnitude;

                    // For each possible radius
                    for (int r = minRadiusPixels; r <= maxRadiusPixels; r++)
                    {
                        // Calculate center candidates using gradient direction
                        float a1 = x - r * gx;
                        float b1 = y - r * gy;
                        float a2 = x + r * gx;
                        float b2 = y + r * gy;

                        // Vote for both possible centers
                        if (a1 >= 0 && a1 < cols && b1 >= 0 && b1 < rows)
                        {
                            accumulator[static_cast<int>(b1)][static_cast<int>(a1)][r - minRadiusPixels] += magnitude;
                        }
                        if (a2 >= 0 && a2 < cols && b2 >= 0 && b2 < rows)
                        {
                            accumulator[static_cast<int>(b2)][static_cast<int>(a2)][r - minRadiusPixels] += magnitude;
                        }
                    }
                }
            }
        }
    }

    // Find local maxima in accumulator with adaptive threshold
    float maxAccumulator = 0;
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            for (int r = 0; r <= maxRadiusPixels - minRadiusPixels; r++)
            {
                maxAccumulator = max(maxAccumulator, accumulator[y][x][r]);
            }
        }
    }

    // Adaptive threshold based on maximum accumulator value and component size
    float baseThreshold = maxAccumulator * 0.4f; // Increased from 0.3 to 0.4

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            for (int r = 0; r <= maxRadiusPixels - minRadiusPixels; r++)
            {
                float threshold = baseThreshold;

                // Adjust threshold based on radius
                float radius = r + minRadiusPixels;
                float expectedPerimeter = 2 * CV_PI * radius;
                float minRequiredEdges = expectedPerimeter * 0.6f; // At least 60% of perimeter should be edges

                if (accumulator[y][x][r] > threshold)
                {
                    // Check if this is a local maximum
                    bool isLocalMax = true;
                    for (int dy = -1; dy <= 1 && isLocalMax; dy++)
                    {
                        for (int dx = -1; dx <= 1 && isLocalMax; dx++)
                        {
                            for (int dr = -1; dr <= 1 && isLocalMax; dr++)
                            {
                                int ny = y + dy;
                                int nx = x + dx;
                                int nr = r + dr;
                                if (ny >= 0 && ny < rows && nx >= 0 && nx < cols &&
                                    nr >= 0 && nr <= maxRadiusPixels - minRadiusPixels)
                                {
                                    if (accumulator[ny][nx][nr] > accumulator[y][x][r])
                                    {
                                        isLocalMax = false;
                                    }
                                }
                            }
                        }
                    }

                    if (isLocalMax)
                    {
                        // Additional circle validation
                        int votes = 0;
                        int total = 0;
                        float radius = r + minRadiusPixels;

                        // Sample points on the circle perimeter
                        for (int theta = 0; theta < 360; theta += 2) // Increased sampling density
                        {
                            double angle = theta * CV_PI / 180.0;
                            int px = static_cast<int>(x + radius * cos(angle));
                            int py = static_cast<int>(y + radius * sin(angle));

                            if (px >= 0 && px < cols && py >= 0 && py < rows)
                            {
                                total++;
                                if (edges.at<uchar>(py, px) > 0)
                                {
                                    votes++;
                                }
                            }
                        }

                        // Check if enough points on the perimeter are edges
                        float circleScore = static_cast<float>(votes) / total;
                        if (circleScore > 0.6f && votes >= minRequiredEdges) // Increased threshold to 60%
                        {
                            // Check minimum distance between circles
                            bool tooClose = false;
                            for (const Vec3f &circle : detectedCircles)
                            {
                                double dist = sqrt(pow(circle[0] - x, 2) + pow(circle[1] - y, 2));
                                if (dist < minDist)
                                {
                                    tooClose = true;
                                    break;
                                }
                            }

                            if (!tooClose)
                            {
                                detectedCircles.push_back(Vec3f(x, y, radius));
                            }
                        }
                    }
                }
            }
        }
    }

    // Copy results to output
    circles.create(1, detectedCircles.size(), CV_32FC3);
    Mat circlesMat = circles.getMat();
    for (size_t i = 0; i < detectedCircles.size(); i++)
    {
        circlesMat.at<Vec3f>(0, i) = detectedCircles[i];
    }
}