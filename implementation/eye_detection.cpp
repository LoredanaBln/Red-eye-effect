#include "eye_detection.h"

using namespace std;
using namespace cv;

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


const float MIN_ASPECT_RATIO = 0.8f;
const float MAX_ASPECT_RATIO = 1.2f;

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

int *compute_histogram_naive(Mat source)
{
    int *histogram = (int *) calloc(256, sizeof(int));

    for (int i = 1; i < source.rows - 1; i++) {
        for (int j = 1; j < source.cols - 1; j++) {
            histogram[source.at<uchar>(i, j)]++;
        }
    }

    return histogram;
}

vector<float> compute_kernel_1D(int kernel_size)
{
    vector<float> kernel(kernel_size);
    float std = (float) kernel_size / 6.0;
    int ks2 = kernel_size / 2;
    for (int i = 0; i < kernel_size; i++) {
        float exponent = (-pow((i - ks2), 2)) / (2 * pow(std, 2));
        kernel[i] = exp(exponent) / (sqrt(2 * CV_PI) * std);
    }
    return kernel;
}

Mat apply_gaussian_filtering_1D(Mat source, int kernel_size)
{
    Mat result = source.clone();
    Mat temp = source.clone();
    vector<float> kernel = compute_kernel_1D(kernel_size);
    int ks2 = kernel_size / 2;

    for (int i = ks2; i < source.rows - ks2; i++) {
        for (int j = 0; j < source.cols; j++) {
            float sum_y = 0.0;
            float sum_y_g = 0.0;
            for (int k = -ks2; k <= ks2; k++) {
                float weight = kernel[k + ks2];
                sum_y += (float) source.at<uchar>(i + k, j) * weight;
                sum_y_g += weight;
            }
            temp.at<uchar>(i, j) = (uchar) (sum_y / sum_y_g);
        }
    }

    for (int i = 0; i < source.rows; i++) {
        for (int j = ks2; j < source.cols - ks2; j++) {
            float sum_x = 0.0;
            float sum_x_g = 0.0;
            for (int k = -ks2; k <= ks2; k++) {
                float weight = kernel[k + ks2];
                sum_x += (float) temp.at<uchar>(i, j + k) * weight;
                sum_x_g += weight;
            }
            result.at<uchar>(i, j) = (uchar) (sum_x / sum_x_g);
        }
    }

    return result;
}

gradients_structure
compute_gradients(Mat source, const int *filter_x, const int *filter_y, const int *di, const int *dj)
{
    gradients_structure gradients;
    int rows = source.rows;
    int cols = source.cols;

    Mat filtered_image = apply_gaussian_filtering_1D(source, 3);
    Mat result_x = Mat::zeros(rows, cols, CV_32SC1);
    Mat result_y = Mat::zeros(rows, cols, CV_32SC1);
    Mat magnitude = Mat::zeros(rows, cols, CV_32FC1);
    Mat direction = Mat::zeros(rows, cols, CV_32FC1);

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int sum_x = 0;
            int sum_y = 0;
            for (int k = 0; k < 8; k++) {
                sum_x += (int) filtered_image.at<uchar>(i + di[k], j + dj[k]) * filter_x[k];
                sum_y += (int) filtered_image.at<uchar>(i + di[k], j + dj[k]) * filter_y[k];
            }
            result_x.at<int>(i, j) = sum_x;
            result_y.at<int>(i, j) = sum_y;
            magnitude.at<float>(i, j) = (float) sqrt(sum_x * sum_x + sum_y * sum_y);
            direction.at<float>(i, j) = atan2((float) sum_y, (float) sum_x);
        }
    }

    gradients.x = result_x;
    gradients.y = result_y;
    gradients.magnitude = magnitude;
    gradients.direction = direction;

    return gradients;
}

int getArea(float direction)
{
    int dir = 0;
    double direction_deg = direction * 180 / CV_PI;
    if (direction_deg < 0) {
        direction_deg += 180.0;
    }

    if ((direction_deg >= 0 && direction_deg < 22.5) || (direction_deg >= 157.5 && direction_deg < 180)) {
        dir = 0;
    } else if (direction_deg >= 22.5 && direction_deg < 67.5) {
        dir = 1;
    } else if (direction_deg >= 67.5 && direction_deg < 112.5) {
        dir = 2;
    } else {
        dir = 3;
    }
    return dir;
}

Mat non_maxima_gradient_suppression(gradients_structure gradient)
{
    Mat magnitude = gradient.magnitude;
    Mat direction = gradient.direction;
    int rows = gradient.magnitude.rows;
    int cols = gradient.magnitude.cols;

    Mat result = Mat::zeros(rows, cols, CV_32FC1);

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int dir = getArea(direction.at<float>(i, j));
            float current_mag = magnitude.at<float>(i, j);

            if (dir == 0 && current_mag >= magnitude.at<float>(i, j - 1) &&
                current_mag >= magnitude.at<float>(i, j + 1) ||
                dir == 1 && current_mag >= magnitude.at<float>(i + 1, j - 1) &&
                current_mag >= magnitude.at<float>(i - 1, j + 1) ||
                dir == 2 && current_mag >= magnitude.at<float>(i - 1, j) &&
                current_mag >= magnitude.at<float>(i + 1, j) ||
                dir == 3 && current_mag >= magnitude.at<float>(i - 1, j - 1) &&
                current_mag >= magnitude.at<float>(i + 1, j + 1)) {
                result.at<float>(i, j) = current_mag;
            }
        }
    }

    return result;
}

filter_structure get_filter(string filter_type)
{
    filter_structure filter;
    int sobelx[8] = {2, 1, 0, -1, -2, -1, 0, 1};
    int sobely[8] = {0, 1, 2, 1, 0, -1, -2, -1};

    filter.filter_x = new int[8];
    filter.filter_y = new int[8];
    filter.di = new int[8];
    filter.dj = new int[8];

    if (filter_type == "sobel") {
        memcpy(filter.filter_x, sobelx, 8 * sizeof(int));
        memcpy(filter.filter_y, sobely, 8 * sizeof(int));
    }
    memcpy(filter.di, dx, 8 * sizeof(int));
    memcpy(filter.dj, dy, 8 * sizeof(int));

    return filter;
}

Mat normalize_supression(Mat supression, string filter_type)
{
    int rows = supression.rows;
    int cols = supression.cols;
    Mat result = Mat::zeros(rows, cols, CV_8UC1);

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            if (filter_type == "sobel") {
                result.at<uchar>(i, j) = (uchar) (supression.at<float>(i, j) / (4.0f * sqrt(2.0f)));
            }
        }
    }

    return result;
}

int adaptive_threshold(Mat magnitude, float p, bool verbose)
{
    int th;
    int rows = magnitude.rows;
    int cols = magnitude.cols;

    int *histogram = compute_histogram_naive(magnitude);
    int no_edge_pixels = (int) (p * (float) ((rows - 2) * (cols - 2) - histogram[0]));
    int sum = 0;
    int index = 255;
    while (sum < no_edge_pixels) {
        th = index;
        sum += histogram[index--];
    }

    return th;
}

Mat histeresis_thresholding(Mat source, int th)
{
    int rows = source.rows;
    int cols = source.cols;
    Mat result = Mat::zeros(rows, cols, CV_8UC1);

    int t_low = (int) (th * 0.4);

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            if (source.at<uchar>(i, j) >= t_low && source.at<uchar>(i, j) < th) {
                result.at<uchar>(i, j) = 127;
            } else if (source.at<uchar>(i, j) >= th) {
                result.at<uchar>(i, j) = 255;
            }
        }
    }

    return result;
}

Mat histeresis(Mat source)
{
    int rows = source.rows;
    int cols = source.cols;
    Mat result = source.clone();

    queue<Point> queue;

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            if (result.at<uchar>(i, j) == 255) {
                queue.push(Point(j, i));

                while (!queue.empty()) {
                    Point point = queue.front();
                    queue.pop();

                    for (int k = 0; k < 8; k++) {
                        int ni = point.y + dx[k];
                        int nj = point.x + dy[k];

                        if (result.at<uchar>(ni, nj) == 127) {
                            result.at<uchar>(ni, nj) = 255;
                            queue.push(Point(nj, ni));
                        }
                    }
                }
            }
        }
    }

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            if (result.at<uchar>(i, j) == 127) {
                result.at<uchar>(i, j) = 0;
            }
        }
    }

    return result;
}

Mat apply_Canny(Mat source, int low_threshold, int high_threshold, string filter_type, bool verbose)
{
    Mat result;
    Mat gauss = source.clone();

    gauss = apply_gaussian_filtering_1D(gauss, 3);

    filter_structure filter = get_filter(filter_type);
    gradients_structure gradient = compute_gradients(gauss, filter.filter_x, filter.filter_y, filter.di, filter.dj);

    Mat non_maxima = non_maxima_gradient_suppression(gradient);
    non_maxima = normalize_supression(non_maxima, filter_type);

    int th = adaptive_threshold(non_maxima, 0.1, verbose);

    Mat hist = histeresis_thresholding(non_maxima, th);

    result = histeresis(hist);

    return result;
}

Mat detect_circular_components(Mat binary, Mat original, double circularityThreshold) {
    labels_ labels = two_pass_labeling(binary);
    vector<int> validatedLabels;

    for (int lbl = 1; lbl <= labels.no_newlabels; lbl++) {
        Mat labelMask = Mat::zeros(binary.size(), CV_8UC1);
        int pixelCount = 0;
        int minX = labels.labels.cols, minY = labels.labels.rows;
        int maxX = 0, maxY = 0;

        for (int i = 0; i < labels.labels.rows; i++) {
            for (int j = 0; j < labels.labels.cols; j++) {
                if (labels.labels.at<int>(i, j) == lbl) {
                    labelMask.at<uchar>(i, j) = 255;
                    pixelCount++;
                    minX = min(minX, j);
                    minY = min(minY, i);
                    maxX = max(maxX, j);
                    maxY = max(maxY, i);
                }
            }
        }

        int width = maxX - minX;
        int height = maxY - minY;
        float aspectRatio = (float) width / height;

        Mat edges = apply_Canny(labelMask, 50, 15, "sobel", false);
        int perimeter = 0;
        for (int i = 0; i < edges.rows; i++) {
            for (int j = 0; j < edges.cols; j++) {
                if (edges.at<uchar>(i, j) > 0) {
                    perimeter++;
                }
            }
        }

        float circularity = (4.0f * CV_PI * pixelCount) / (perimeter * perimeter);

        if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) {
            continue;
        }

        int centerX = (minX + maxX) / 2;
        int centerY = (minY + maxY) / 2;

        bool surroundingsValid = check_surroundings(original, centerX, centerY, width, height);

        if (surroundingsValid) {
            validatedLabels.push_back(lbl);
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