#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

void demo1() {
    string path="/home/yangn0/opencv_demo/实验一原图.jpg";
    Mat image = imread(path);
    imshow("原始图像",image);
    waitKey(0);
    
    Mat grey_image;
    grey_image.create(image.rows,image.cols,CV_8UC1);
    
    int rown=image.rows;
    int coln=image.cols;
    
    //BGR
    //GRAY=0.3*R+0.59*G+0.11*B
    ofstream OutFile("grey_image.txt");
    for(int i=0;i<rown;i++){
        for(int u=0;u<coln;u++){
            int B=image.at<Vec3b>(i,u)[0];
            int G=image.at<Vec3b>(i,u)[1];
            int R=image.at<Vec3b>(i,u)[2];
            int grey=B*0.11+G*0.59+R*0.3;
            grey_image.at<uchar>(i,u)=grey;
//            cout<<image.at<Vec3b>(i,u)<<endl<<grey_image.at<uchar>(i,u)<<endl;
            OutFile<< grey <<' ';
        }
        OutFile<<endl;
    }
    OutFile.close();
    
    imshow("灰度图像",grey_image);
    waitKey(0);
    
    for(int i=0;i<rown;i++){
        for(int u=0;u<coln;u++){
            int grey=grey_image.at<uchar>(i,u)*5+10;
            if(grey>255){
                grey=255;
            }
            grey_image.at<uchar>(i,u)=grey;
        }
    }
    
    imshow("修改后的灰度图像",grey_image);
    waitKey(0);
}
