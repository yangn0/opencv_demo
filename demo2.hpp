//
//  demo2.hpp
//  opencv_demo
//
//  Created by 杨宁 on 2023/9/21.
//

#ifndef demo2_hpp
#define demo2_hpp

#include <opencv2/opencv.hpp>
using namespace cv;

void demo2();

// 高斯滤波函数
void gaussFilter(Mat input, Mat &output);

// 计算梯度幅值图像和方向图像
void gradDire(Mat input, Mat& Gradimage, Mat& Direimage);

// 非极大值抑制图像
void nonmaxSuppression(Mat Gradimage, Mat Direimage, Mat& Suppimage);

// 滞后阈值处理（双阈值）
void doubleThread(Mat Suppimage, Mat& Edgeimage, int th_high, int th_low);

#endif /* demo2_hpp */
