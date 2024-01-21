#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void demo4()
{
    string path = "/home/yangn0/opencv_demo/实验三GT图.bmp";
    Mat image_GT = imread(path, IMREAD_GRAYSCALE);
    // imshow("GT图像", image_GT);
    // waitKey(0);

    // 轮廓检测
    vector<vector<Point>> contours; // 轮廓
    vector<Vec4i> hierarchy;        // 存放轮廓结构变量
    findContours(image_GT, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

    double A = contourArea(contours[0]);
    cout << "面积=" << A << endl;

    double P = arcLength(contours[0], true);
    cout << "周长=" << P << endl;

    double max_d = 0, d = 0;

    for (int i = 0; i < contours[0].size(); i++)
    {
        d = sqrt(contours[0][i].x * contours[0][i].x + contours[0][i].y * contours[0][i].y);
        if (d > max_d)
            max_d = d;
    }
    cout << "直径=" << max_d << endl;

    cout << "薄度=" << P*P/A << endl;

    Moments moment;//矩
    moment = moments(contours[0], false);
    cv::Point pt1;
    if (moment.m00 != 0)//除数不能为0
    {
        pt1.x = cvRound(moment.m10 / moment.m00);//计算重心横坐标
        pt1.y = cvRound(moment.m01 / moment.m00);//计算重心纵坐标
    }
    cout << "重心=" << pt1 << endl;

    Rect r=boundingRect(contours[0]);
    cout << "X-Y纵横比=" << double(r.height)/double(r.width) << endl;
    
    RotatedRect r1=minAreaRect(contours[0]);
    cout << "最小纵横比=" << double(r1.size.height)/double(r1.size.width) << endl;
}