#include "eye_detection.h"

using namespace std;
using namespace cv;

#define PI 3.14

#define WHITE_THRESHOLD 140
#define OFF_WHITE_THRESHOLD 120
#define COLOR_DIFF_THRESHOLD 30
#define DARK_THRESHOLD 40

#define SKIN_RED_MIN 0.4
#define SKIN_RED_MAX 0.9
#define SKIN_GREEN_MIN 0.2
#define SKIN_GREEN_MAX 0.7
#define SKIN_BLUE_MIN 0.1
#define SKIN_BLUE_MAX 0.6
#define SKIN_SATURATION_MAX 0.6

#define RED_HUE_MIN 10
#define RED_HUE_MAX 200
#define RED_SATURATION_MIN 100
#define RED_VALUE_MIN 50

bool is_inside(int x, int y, int rows, int cols)
{
    return x >= 0 && y >= 0 && x < rows && y < cols;
}

double calculate_ratio(int count, int total)
{
    return total > 0 ? (double)count / total : 0.0;
}

HSV rgb_to_hsv_pixel(float r, float g, float b)
{
    HSV hsv;
    float M = max(r, max(g, b));
    float m = min(r, min(g, b));
    float C = M - m;

    hsv.v = M;
    hsv.s = (M != 0.0) ? (C / M) : 0.0;

    if (C != 0.0)
    {
        if (M == r)
        {
            hsv.h = 60 * (g - b) / C;
        }
        else if (M == g)
        {
            hsv.h = 120 + 60 * (b - r) / C;
        }
        else if (M == b)
        {
            hsv.h = 240 + 60 * (r - g) / C;
        }
    }
    else
    {
        hsv.h = 0.0;
    }

    if (hsv.h < 0.0)
    {
        hsv.h += 360;
    }

    return hsv;
}

bool is_red_pixel(float h, float s, float v)
{
    return (h < RED_HUE_MIN || h > RED_HUE_MAX) && s * 255 > RED_SATURATION_MIN && v * 255 > RED_VALUE_MIN;
}

Mat create_mask(Mat source)
{
    Mat mask = Mat::zeros(source.size(), CV_8UC1);

    for (int i = 0; i < source.rows; i++)
    {
        for (int j = 0; j < source.cols; j++)
        {
            Vec3b pixel = source.at<Vec3b>(i, j);
            float r = pixel[2] / 255.0f;
            float g = pixel[1] / 255.0f;
            float b = pixel[0] / 255.0f;

            HSV hsv = rgb_to_hsv_pixel(r, g, b);

            if (is_red_pixel(hsv.h, hsv.s, hsv.v))
            {
                mask.at<uchar>(i, j) = 255;
            }
        }
    }
    mask = erode(mask, 2);
    mask = dilate(mask, 2);
    return mask;
}

Mat dilate(Mat source, int no_iter)
{
    Mat dst = source.clone();
    Mat aux;

    int rows = source.rows;
    int cols = source.cols;

    for (int iter = 0; iter < no_iter; iter++)
    {
        aux = dst.clone();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (dst.at<uchar>(i, j) == 255)
                {
                    for (int k = 0; k < 8; k++)
                    {
                        int ni = i + dx[k];
                        int nj = j + dy[k];

                        if (is_inside(ni, nj, rows, cols))
                        {
                            aux.at<uchar>(ni, nj) = 255;
                        }
                    }
                }
            }
        }
        dst = aux.clone();
    }

    return dst;
}

Mat erode(Mat source, int no_iter)
{
    Mat dst = source.clone();
    Mat aux;

    int rows = source.rows;
    int cols = source.cols;

    for (int iter = 0; iter < no_iter; iter++)
    {
        aux = dst.clone();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (dst.at<uchar>(i, j) == 255)
                {
                    for (int k = 0; k < 8; k++)
                    {
                        int ni = i + dx[k];
                        int nj = j + dy[k];

                        if (!is_inside(ni, nj, rows, cols) || dst.at<uchar>(ni, nj) == 0)
                        {
                            aux.at<uchar>(i, j) = 0;
                            break;
                        }
                    }
                }
            }
        }

        dst = aux.clone();
    }
    return dst;
}

bool is_white_pixel(Vec3b pixel)
{
    bool isWhite = pixel[0] > WHITE_THRESHOLD &&
                   pixel[1] > WHITE_THRESHOLD &&
                   pixel[2] > WHITE_THRESHOLD;

    bool isOffWhite = pixel[0] > OFF_WHITE_THRESHOLD &&
                      pixel[1] > OFF_WHITE_THRESHOLD &&
                      pixel[2] > OFF_WHITE_THRESHOLD &&
                      abs(pixel[0] - pixel[1]) < COLOR_DIFF_THRESHOLD &&
                      abs(pixel[1] - pixel[2]) < COLOR_DIFF_THRESHOLD &&
                      abs(pixel[0] - pixel[2]) < COLOR_DIFF_THRESHOLD;

    return isWhite || isOffWhite;
}

bool is_dark_pixel(Vec3b pixel)
{
    return pixel[0] < DARK_THRESHOLD || pixel[1] < DARK_THRESHOLD || pixel[2] < DARK_THRESHOLD;
}

bool is_skin_tone(Vec3b pixel)
{
    float r = pixel[2] / 255.0f;
    float g = pixel[1] / 255.0f;
    float b = pixel[0] / 255.0f;

    bool redDominant = r > g && r > b;

    bool inRange = r > SKIN_RED_MIN && r < SKIN_RED_MAX &&
                   g > SKIN_GREEN_MIN && g < SKIN_GREEN_MAX &&
                   b > SKIN_BLUE_MIN && b < SKIN_BLUE_MAX;

    float maxVal = max(max(r, g), b);
    float minVal = min(min(r, g), b);
    float saturation = (maxVal - minVal) / maxVal;
    bool notTooSaturated = saturation < SKIN_SATURATION_MAX;

    return redDominant && inRange && notTooSaturated;
}

labels_ two_pass_labeling(Mat source)
{
    int rows = source.rows;
    int cols = source.cols;
    Mat labels = Mat::zeros(rows, cols, CV_32SC1);
    int no_newlabels = 0;
    int new_label = 0;

    vector<vector<int>> edges(10000);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (source.at<uchar>(i, j) == 255 && labels.at<int>(i, j) == 0)
            {
                vector<int> L;

                for (int k = 0; k < 4; k++)
                {
                    int ni = i + dx[k];
                    int nj = j + dy[k];

                    if (is_inside(ni, nj, rows, cols))
                    {
                        if (labels.at<int>(ni, nj) > 0)
                        {
                            L.push_back(labels.at<int>(ni, nj));
                        }
                    }
                }

                if (L.empty())
                {
                    no_newlabels++;
                    labels.at<int>(i, j) = no_newlabels;
                }
                else
                {
                    int mini = 256;
                    for (int k = 0; k < L.size(); k++)
                    {
                        if (L[k] < mini)
                        {
                            mini = L[k];
                        }
                    }

                    labels.at<int>(i, j) = mini;
                    for (int k : L)
                    {
                        if (k != mini)
                        {
                            edges[mini].push_back(k);
                            edges[k].push_back(mini);
                        }
                    }
                }
            }
        }
    }

    vector<int> new_labels(no_newlabels + 1, 0);
    for (int i = 1; i <= no_newlabels; i++)
    {
        if (new_labels[i] == 0)
        {
            new_label++;
            queue<int> Q;
            new_labels[i] = new_label;
            Q.push(i);
            while (!Q.empty())
            {
                int x = Q.front();
                Q.pop();
                for (int y : edges[x])
                {
                    if (new_labels[y] == 0)
                    {
                        new_labels[y] = new_label;
                        Q.push(y);
                    }
                }
            }
        }
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (labels.at<int>(i, j) > 0)
            {
                labels.at<int>(i, j) = new_labels[labels.at<int>(i, j)];
            }
        }
    }

    return {labels, new_label};
}

bool is_edge_pixel(const Mat &labelMat, int x, int y)
{
    int rows = labelMat.rows;
    int cols = labelMat.cols;
    int label = labelMat.at<int>(x, y);

    for (int k = 0; k < 4; k++)
    {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (!is_inside(nx, ny, rows, cols) || labelMat.at<int>(nx, ny) != label)
        {
            return true;
        }
    }
    return false;
}

bool check_surroundings(Mat original, int redX, int redY, int redWidth, int redHeight)
{
    int whiteCount = 0;
    int skinCount = 0;
    int darkCount = 0;
    int totalCount = 0;
    int margin = 2;

    int checkedAreaWidth = redWidth * 2;
    int checkedAreaHeight = redHeight * 2;
    int checkX = redX - checkedAreaWidth / 2;
    int checkY = redY - checkedAreaHeight / 2;

    for (int i = checkY; i < checkY + checkedAreaHeight; i++)
    {
        for (int j = checkX; j < checkX + checkedAreaWidth; j++)
        {
            if (i >= 0 && i < original.rows && j >= 0 && j < original.cols)
            {
                if (i >= redY - redHeight / 2 - margin && i < redY + redHeight / 2 + margin &&
                    j >= redX - redWidth / 2 - margin && j < redX + redWidth / 2 + margin)
                {
                    continue;
                }

                Vec3b pixel = original.at<Vec3b>(i, j);
                if (is_white_pixel(pixel))
                {
                    whiteCount++;
                }
                else if (is_skin_tone(pixel))
                {
                    skinCount++;
                }
                else if (is_dark_pixel(pixel))
                {
                    darkCount++;
                }
                totalCount++;
            }
        }
    }

    double whiteRatio = calculate_ratio(whiteCount, totalCount);
    double skinRatio = calculate_ratio(skinCount, totalCount);
    double darkRatio = calculate_ratio(darkCount, totalCount);

    return totalCount > 0 &&
           whiteRatio > 0.01 &&
           skinRatio < 0.5 &&
           darkRatio > 0.001;
}

bool is_label_validated(int label, const vector<int> &validatedLabels)
{
    for (int validatedLabel : validatedLabels)
    {
        if (label == validatedLabel)
        {
            return true;
        }
    }
    return false;
}

Mat detect_circular_components(Mat binary, Mat original, double circularityThreshold)
{
    labels_ labels = two_pass_labeling(binary);

    vector<int> area(labels.no_newlabels + 1, 0);
    vector<int> perimeter(labels.no_newlabels + 1, 0);

    for (int i = 0; i < labels.labels.rows; i++)
    {
        for (int j = 0; j < labels.labels.cols; j++)
        {
            int lbl = labels.labels.at<int>(i, j);
            if (lbl > 0)
            {
                area[lbl]++;
                if (is_edge_pixel(labels.labels, i, j))
                {
                    perimeter[lbl]++;
                }
            }
        }
    }

    vector<int> validatedLabels;

    for (int lbl = 1; lbl <= labels.no_newlabels; lbl++)
    {
        if (perimeter[lbl] > 0)
        {
            double thinness_ratio = (4.0 * PI * area[lbl]) / (perimeter[lbl] * perimeter[lbl]);

            if (thinness_ratio >= circularityThreshold)
            {
                int minX = labels.labels.cols, minY = labels.labels.rows;
                int maxX = 0, maxY = 0;

                for (int i = 0; i < labels.labels.rows; i++)
                {
                    for (int j = 0; j < labels.labels.cols; j++)
                    {
                        if (labels.labels.at<int>(i, j) == lbl)
                        {
                            minX = min(minX, j);
                            minY = min(minY, i);
                            maxX = max(maxX, j);
                            maxY = max(maxY, i);
                        }
                    }
                }

                int centerX = (minX + maxX) / 2;
                int centerY = (minY + maxY) / 2;
                int width = maxX - minX;
                int height = maxY - minY;

                if (check_surroundings(original, centerX, centerY, width, height))
                {
                    validatedLabels.push_back(lbl);
                }
            }
        }
    }

    Mat output = Mat::zeros(labels.labels.size(), CV_8UC1);
    for (int i = 0; i < labels.labels.rows; i++)
    {
        for (int j = 0; j < labels.labels.cols; j++)
        {
            int lbl = labels.labels.at<int>(i, j);
            if (lbl > 0 && is_label_validated(lbl, validatedLabels))
            {
                output.at<uchar>(i, j) = 255;
            }
        }
    }

    return output;
}

Mat correct_red_eye(Mat mask, Mat source)
{
    int rows = mask.rows;
    int cols = mask.cols;
    Mat corrected = source.clone();

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (mask.at<uchar>(i, j) == 255)
            {
                Vec3b &pixel = corrected.at<Vec3b>(i, j);
                uchar green = pixel[1];
                uchar blue = pixel[0];

                pixel[2] = min(green, blue);
            }
        }
    }

    return corrected;
}