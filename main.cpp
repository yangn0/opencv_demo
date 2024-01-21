#include <iostream>
#include <opencv2/opencv.hpp>
#include "demo1.hpp"
#include "demo2.hpp"
#include "demo3.hpp"
#include "demo4.hpp"
using namespace cv;
using std::string;

void test(string path){
    Mat image = imread(path);
    namedWindow("origin");
    imshow("origin", image);
    
    Mat gray;
    cvtColor(image, gray, COLOR_RGBA2GRAY);
    namedWindow("gray");
    imshow("gray", gray);
    waitKey(0);
};


int main(int argc, const char * argv[]) {
    // test("/home/yangn0/opencv_demo/实验一原图.jpg");
    // demo1();
    // demo2();
    // demo3();
    demo4();
    return 0;
}
