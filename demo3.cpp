#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int T = 0;           // Otsu算法阈值
double varValue = 0; // 类间方差中间值保存
double w0 = 0;       // 前景像素点数所占比例
double w1 = 0;       // 背景像素点数所占比例
double u0 = 0;       // 前景平均灰度
double u1 = 0;       // 背景平均灰度

int OTSU(Mat image)
{
    int Histogram[256] = {0};
    for (int i = 0; i < image.rows; i++)
    {
        for (int u = 0; u < image.cols; u++)
        {
            Histogram[image.at<uchar>(i, u)]++;
            // cout<<(int)image.at<uchar>(i, u)<<' ';
        }
    }

    double totalNum = image.rows * image.cols; // 像素总数
    for (int i = 0; i < 255; i++)
    {
        // 每次遍历之前初始化各变量
        w1 = 0;
        u1 = 0;
        w0 = 0;
        u0 = 0;
        //***********背景各分量值计算**************************
        for (int j = 0; j <= i; j++) // 背景部分各值计算
        {
            w1 += Histogram[j];     // 背景部分像素点总数
            u1 += j * Histogram[j]; // 背景部分像素总灰度和
        }
        if (w1 == 0) // 背景部分像素点数为0时退出
        {
            continue;
        }
        u1 = u1 / w1;       // 背景像素平均灰度
        w1 = w1 / totalNum; // 背景部分像素点数所占比例
        //***********背景各分量值计算**************************

        //***********前景各分量值计算**************************
        for (int k = i + 1; k < 255; k++)
        {
            w0 += Histogram[k];     // 前景部分像素点总数
            u0 += k * Histogram[k]; // 前景部分像素总灰度和
        }
        if (w0 == 0) // 前景部分像素点数为0时退出
        {
            break;
        }
        u0 = u0 / w0;       // 前景像素平均灰度
        w0 = w0 / totalNum; // 前景部分像素点数所占比例
        //***********前景各分量值计算**************************

        //***********类间方差计算******************************
        double varValueI = w0 * w1 * (u1 - u0) * (u1 - u0); // 当前类间方差计算
        if (varValue < varValueI)
        {
            varValue = varValueI;
            T = i;
        }
    }
    return T;
}

void demo3()
{
    string path = "/home/yangn0/opencv_demo/实验一原图.jpg";
    Mat image = imread(path, IMREAD_GRAYSCALE);
    imshow("原始灰度图像", image);
    waitKey(0);

    // 掩膜
    Rect r1(192, 26, 156, 145); // (192, 26, 156, 145);
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
	mask(r1).setTo(255);
    Mat img2;
    image.copyTo(img2, mask);
    imshow("掩膜", img2);
    waitKey(0);

    image = image(r1);
    imshow("image", image);
    waitKey(0);
    
    

    Mat OTSU_image;
    // opencv自带大津算法
    // double opencv_threshold=threshold(image,OTSU_image, 0, 255, THRESH_BINARY + THRESH_OTSU);
    // cout << opencv_threshold << endl;

    double otsuThreshold = OTSU(image);
    cout << otsuThreshold << endl;
    threshold(image, OTSU_image, otsuThreshold, 255, THRESH_BINARY);

    OTSU_image.copyTo(img2(r1));
    image=img2;
    imshow("OTSU图像", image);
    waitKey(0);

    // 计算 jaccard相似性指数
    path = "/home/yangn0/opencv_demo/实验三GT图.bmp";
    Mat image_GT = imread(path);
    // cout<<image_GT;
    int x=0;
    int y=image.rows * image.cols*2 ; // 像素总数;
    for (int i = 0; i < image.rows; i++)
    {
        for (int u = 0; u < image.cols; u++)
        {
            if(image_GT.at<uchar>(i, u)==image.at<uchar>(i, u)){
                x++;
            }

        }
    }
    double jaccard= (double)x/double(y);
    cout<< jaccard << endl;
}