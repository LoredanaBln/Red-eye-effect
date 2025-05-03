#include "eye_detection.h"

using namespace std;
using namespace cv;

bool isInside(int x, int y, int rows, int cols)
{
    return x >= 0 && y >= 0 && x < rows && y < cols;
}

image_channels_bgr break_channels(Mat source)
{
    int rows = source.rows;
    int cols = source.cols;
    Mat B = Mat(rows, cols, CV_8UC1);
    Mat G = Mat(rows, cols, CV_8UC1);
    Mat R = Mat(rows, cols, CV_8UC1);
    image_channels_bgr bgr_channels;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
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

image_channels_hsv bgr_to_hsv(image_channels_bgr bgr_channels)
{
    int rows = bgr_channels.B.rows;
    int cols = bgr_channels.B.cols;
    Mat H = Mat::zeros(rows, cols, CV_32FC1);
    Mat S = Mat::zeros(rows, cols, CV_32FC1);
    Mat V = Mat::zeros(rows, cols, CV_32FC1);
    image_channels_hsv hsv_channels;

    float M, m, C;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float r = (float)bgr_channels.R.at<uchar>(i, j) / 255;
            float g = (float)bgr_channels.G.at<uchar>(i, j) / 255;
            float b = (float)bgr_channels.B.at<uchar>(i, j) / 255;

            M = max(r, max(g, b));
            m = min(r, min(g, b));
            C = M - m;

            V.at<float>(i, j) = M;
            if (M != 0.0)
            {
                S.at<float>(i, j) = C / M;
            }
            else
            {
                S.at<float>(i, j) = 0.0;
            }

            if (C != 0.0)
            {
                if (M == r)
                {
                    H.at<float>(i, j) = 60 * (g - b) / C;
                }
                if (M == g)
                {
                    H.at<float>(i, j) = 120 + 60 * (b - r) / C;
                }
                if (M == b)
                {
                    H.at<float>(i, j) = 240 + 60 * (r - g) / C;
                }
            }
            else
            {
                H.at<float>(i, j) = 0.0;
            }
            if (H.at<float>(i, j) < 0.0)
            {
                H.at<float>(i, j) += 360;
            }
        }
    }

    hsv_channels.H = H;
    hsv_channels.S = S;
    hsv_channels.V = V;

    return hsv_channels;
}

Mat create_red_mask(image_channels_hsv hsv_channels)
{
    Mat mask = Mat::zeros(hsv_channels.H.size(), CV_8UC1);

    for (int i = 0; i < hsv_channels.H.rows; i++)
    {
        for (int j = 0; j < hsv_channels.H.cols; j++)
        {
            float hue = hsv_channels.H.at<float>(i, j);
            float sat = hsv_channels.S.at<float>(i, j) * 255;
            float val = hsv_channels.V.at<float>(i, j) * 255;

            if ((hue < 10 || hue > 170) && sat > 100 && val > 50)
            {
                mask.at<uchar>(i, j) = 255;
            }
        }
    }
    return mask;
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
                corrected.at<Vec3b>(i, j) = 0;
            }
        }
    }
    return corrected;
}

labels_ two_pass_labeling(Mat source)
{
    int rows = source.rows;
    int cols = source.cols;
    Mat labels = Mat::zeros(rows, cols, CV_32SC1);
    int no_newlabels = 0;
    int new_label = 0;

    vector<vector<int>> edges(1000);
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

                    if (isInside(ni, nj, rows, cols))
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

bool isEdgePixel(const Mat &labelMat, int x, int y)
{
    int rows = labelMat.rows;
    int cols = labelMat.cols;
    int label = labelMat.at<int>(x, y);
    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            if (dx == 0 && dy == 0)
                continue;
            int nx = x + dx, ny = y + dy;
            if (!isInside(x, y, rows, cols))
                return true;
            if (labelMat.at<int>(nx, ny) != label)
                return true;
        }
    }
    return false;
}

Mat detect_circular_components(Mat binary, double circularityThreshold)
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
                if (isEdgePixel(labels.labels, i, j))
                {
                    perimeter[lbl]++;
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
            if (lbl > 0 && perimeter[lbl] > 0)
            {
                double circ = (4.0 * CV_PI * area[lbl]) / (perimeter[lbl] * perimeter[lbl]);
                if (circ >= circularityThreshold && area[lbl] > 50)
                {
                    output.at<uchar>(i, j) = 255;
                }
            }
        }
    }

    return output;
}