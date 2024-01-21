#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "math.h"

using namespace cv;
using namespace std;

// 裁剪+边界扩充
// x,y 裁剪中心 size 裁剪直径
cv::Mat cropPadding(cv::Mat srcImage, int x, int y, int size)
{
    Mat part = Mat::ones(size, size, CV_8UC1);
    int r = size / 2;
    for (int i = 0; i < size; i++)
    {
        for (int u = 0; u < size; u++)
        {
            int position_x = x - r + i;
            if (position_x < 0 or position_x >= srcImage.rows)
            {
                part.at<uchar>(i, u) = 255;
                // printf("%d ",part.at<uchar>(i,u));
                continue;
            }
            int position_y = y - r + u;
            if (position_y < 0 or position_y >= srcImage.cols)
            {
                part.at<uchar>(i, u) = 255;
                // printf("%d ",part.at<uchar>(i,u));
                continue;
            }
            part.at<uchar>(i, u) = srcImage.at<uchar>(position_x, position_y);
            // printf("%d ",part.at<uchar>(i,u));
        }
        // puts("");
    }
    return part;
}

// 高斯滤波函数
void gaussFilter(Mat input, Mat &output)
{
    double gaussKernel[5][5];

    gaussKernel[0][0] = double(2) / double(159);
    gaussKernel[0][1] = double(4) / double(159);
    gaussKernel[0][2] = double(5) / double(159);
    gaussKernel[0][3] = double(4) / double(159);
    gaussKernel[0][4] = double(2) / double(159);

    gaussKernel[1][0] = double(4) / double(159);
    gaussKernel[1][1] = double(9) / double(159);
    gaussKernel[1][2] = double(12) / double(159);
    gaussKernel[1][3] = double(9) / double(159);
    gaussKernel[1][4] = double(4) / double(159);

    gaussKernel[2][0] = double(5) / double(159);
    gaussKernel[2][1] = double(12) / double(159);
    gaussKernel[2][2] = double(15) / double(159);
    gaussKernel[2][3] = double(12) / double(159);
    gaussKernel[2][4] = double(5) / double(159);

    gaussKernel[3][0] = double(4) / double(159);
    gaussKernel[3][1] = double(9) / double(159);
    gaussKernel[3][2] = double(12) / double(159);
    gaussKernel[3][3] = double(9) / double(159);
    gaussKernel[3][4] = double(4) / double(159);

    gaussKernel[4][0] = double(2) / double(159);
    gaussKernel[4][1] = double(4) / double(159);
    gaussKernel[4][2] = double(5) / double(159);
    gaussKernel[4][3] = double(4) / double(159);
    gaussKernel[4][4] = double(2) / double(159);

    int rown = input.rows;
    int coln = input.cols;

    Mat cropImage;
    for (int i = 0; i < rown; i++)
    {
        for (int u = 0; u < coln; u++)
        {
            cropImage = cropPadding(input, i, u, 5);
            double sum = 0;
            for (int dot_row = 0; dot_row < 5; dot_row++)
            {
                for (int dot_col = 0; dot_col < 5; dot_col++)
                {
                    sum += cropImage.at<uchar>(dot_row, dot_col) * gaussKernel[dot_row][dot_col];
                }
            }
            output.at<uchar>(i, u) = sum;
            // printf("%d %lf\n",input.at<uchar>(i,u),sum);
        }
    }
}

// 计算梯度幅值图像和方向图像
void gradDire(Mat input, Mat &Gradimage, Mat &Direimage)
{
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};
    int sobel_y[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}};
    for (int i = 0; i < Gradimage.rows; i++)
    {
        for (int u = 0; u < Gradimage.cols; u++)
        {
            Mat cropImage = cropPadding(input, i, u, 3);
            double grad_x = 0;
            double grad_y = 0;
            for (int dot_row = 0; dot_row < 3; dot_row++)
            {
                for (int dot_col = 0; dot_col < 3; dot_col++)
                {
                    grad_x += cropImage.at<uchar>(dot_row, dot_col) * sobel_x[dot_row][dot_col];
                    grad_y += cropImage.at<uchar>(dot_row, dot_col) * sobel_y[dot_row][dot_col];
                }
            }
            Direimage.at<uchar>(i, u) = atan2(grad_y, grad_x) * 180 / 3.1415926;
            // cout<<atan2(grad_y,grad_x)*180/3.1415926<<' ';
            Gradimage.at<uchar>(i, u) = sqrt(grad_x * grad_x + grad_y * grad_y);
        }
        // cout<<endl;
    }
}

// 非极大值抑制图像
// 需要注意的是，如何标志方向并不重要，重要的是梯度方向的计算要和梯度算子的选取保持一致
void nonmaxSuppression(Mat Gradimage, Mat Direimage, Mat &Suppimage)
{
    for (int i = 1; i < Gradimage.rows - 1; i++)
    {
        for (int u = 1; u < Gradimage.cols - 1; u++)
        {
            if ((Direimage.at<uchar>(i, u) <= 22.5 && Direimage.at<uchar>(i, u) >= -22.5) || (Direimage.at<uchar>(i, u) < -157.5 || Direimage.at<uchar>(i, u) > 157.5))
            {
                // 水平方向
                if (Gradimage.at<uchar>(i, u) >= Gradimage.at<uchar>(i + 1, u) && Gradimage.at<uchar>(i, u) >= Gradimage.at<uchar>(i - 1, u))
                {
                    Suppimage.at<uchar>(i, u) = Gradimage.at<uchar>(i, u);
                }
                else
                {
                    Suppimage.at<uchar>(i, u) = 0;
                }
            }
            else if ((Direimage.at<uchar>(i, u) > 112.5 && Direimage.at<uchar>(i, u) < 157.5) || (Direimage.at<uchar>(i, u) < -22.5 && Direimage.at<uchar>(i, u) > -67.5))
            {
                //+45
                if (Gradimage.at<uchar>(i, u) >= Gradimage.at<uchar>(i - 1, u - 1) && Gradimage.at<uchar>(i, u) >= Gradimage.at<uchar>(i + 1, u + 1))
                {
                    Suppimage.at<uchar>(i, u) = Gradimage.at<uchar>(i, u);
                }
                else
                {
                    Suppimage.at<uchar>(i, u) = 0;
                }
            }
            else if ((Direimage.at<uchar>(i, u) <= -67.5 && Direimage.at<uchar>(i, u) >= -112.5) || (Direimage.at<uchar>(i, u) <= 112.5 && Direimage.at<uchar>(i, u) >= 67.5))
            {
                // 垂直方向
                if (Gradimage.at<uchar>(i, u) >= Gradimage.at<uchar>(i, u + 1) && Gradimage.at<uchar>(i, u) >= Gradimage.at<uchar>(i, u - 1))
                {
                    Suppimage.at<uchar>(i, u) = Gradimage.at<uchar>(i, u);
                }
                else
                {
                    Suppimage.at<uchar>(i, u) = 0;
                }
            }
            else if ((Direimage.at<uchar>(i, u) > 22.5 && Direimage.at<uchar>(i, u) < 67.5) || (Direimage.at<uchar>(i, u) < -112.5 && Direimage.at<uchar>(i, u) > -157.5))
            {
                //-45
                if (Gradimage.at<uchar>(i, u) >= Gradimage.at<uchar>(i - 1, u + 1) && Gradimage.at<uchar>(i, u) >= Gradimage.at<uchar>(i + 1, u + 1))
                {
                    Suppimage.at<uchar>(i, u) = Gradimage.at<uchar>(i, u);
                }
                else
                {
                    Suppimage.at<uchar>(i, u) = 0;
                }
            }
        }
    }
}

void doubleThread(Mat Suppimage, Mat &Edgeimage, int th_low, int th_high)
{
    // 强边缘像素
    for (int i = 0; i < Suppimage.rows; i++)
    {
        for (int u = 0; u < Suppimage.cols; u++)
        {
            if (Suppimage.at<uchar>(i, u) >= th_high)
            {
                Edgeimage.at<uchar>(i, u) = Suppimage.at<uchar>(i, u);
            }
            if (Suppimage.at<uchar>(i, u) <= th_low)
            {
                Edgeimage.at<uchar>(i, u) = 0;
            }
        }
    }

    // 弱边缘像素
    for (int i = 1; i < Suppimage.rows - 1; i++) 
    {
        for (int u = 1; u < Suppimage.cols - 1; u++)
        {
            if (Suppimage.at<uchar>(i, u) < th_high && Suppimage.at<uchar>(i, u) > th_low)
            {
                if (Edgeimage.at<uchar>(i - 1, u - 1) !=0 || Edgeimage.at<uchar>(i - 1, u) !=0 || Edgeimage.at<uchar>(i - 1, u + 1) !=0 ||
                    Edgeimage.at<uchar>(i, u - 1) !=0 || Edgeimage.at<uchar>(i, u) !=0 || Edgeimage.at<uchar>(i, u + 1) !=0 ||
                    Edgeimage.at<uchar>(i + 1, u - 1) !=0 || Edgeimage.at<uchar>(i + 1, u) !=0 || Edgeimage.at<uchar>(i + 1, u + 1) !=0)
                {
                    Edgeimage.at<uchar>(i, u) = Suppimage.at<uchar>(i, u);
                }
                else
                {
                    Edgeimage.at<uchar>(i, u) = 0;
                }
            }
        }
    }
}



void demo2()
{
    int th_low = 100;
    int th_high = 200;
    string path = "/home/yangn0/opencv_demo/实验二原图.png";
    Mat image = imread(path, IMREAD_GRAYSCALE);
    // imshow("原始图像",image);
    // waitKey(0);

    int rown=image.rows;
    int coln=image.cols;
    for(int i=0;i<rown;i++){
        for(int u=0;u<coln;u++){
            int grey=image.at<uchar>(i,u)*6+10;
            if(grey>255){
                grey=255;
            }
            image.at<uchar>(i,u)=grey;
        }
    }
    // imshow("修改后的灰度图像",image);
    // waitKey(0);

    // opencv自带canny检测函数
    // Mat img_canny = Mat(image.size(), CV_8UC1);
    // Canny(image, img_canny, 203, 255);
    // imshow("img_canny", img_canny);
    // waitKey();

    Mat img2 = Mat(image.size(), CV_8UC1);
    gaussFilter(image, img2);
    image=img2;
    imshow("高斯滤波", image);
    waitKey();
    

    // GaussianBlur(image,image,Size(3,3), 0);
    // imshow("高斯滤波", image);
    // waitKey();

    Mat Gradimage = Mat(image.size(), CV_8UC1);
    Mat Direimage = Mat(image.size(), CV_8UC1);
    gradDire(image, Gradimage, Direimage);
    imshow("Gradimage", Gradimage);
    waitKey();
    imshow("Direimage", Direimage);
    waitKey();

    Mat Suppimage = Mat(image.size(), CV_8UC1, Scalar(0));
    nonmaxSuppression(Gradimage, Direimage, Suppimage);
    imshow("Suppimage", Suppimage);
    waitKey();

    // 滞后阈值处理（双阈值）
    Mat Edgeimage = Mat(image.size(), CV_8UC1, Scalar(0));
    doubleThread(Suppimage, Edgeimage, th_low, th_high);
    imshow("Edgeimage", Edgeimage);
    waitKey();

    path = "/home/yangn0/opencv_demo/实验二GT图.png";
    Mat image_GT = imread(path, IMREAD_GRAYSCALE);

    int diff = 0;
    for (int i = 0; i < image_GT.rows; i++)
    {
        for (int u = 0; u < image_GT.cols; u++)
        {
            diff += abs(image_GT.at<uchar>(i, u) - Edgeimage.at<uchar>(i, u));
        }
    }
    cout << "diff = " << diff << endl;
}
