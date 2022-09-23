//
// Created by z on 22-7-10.
//

#ifndef VISION_RUNESDETECTOR_H
#define VISION_RUNESDETECTOR_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Predict.h"
#include "Kalm.h"
using namespace cv;
using namespace std;

class detector
{
protected:
    Mat image,binary,channels[3];
    VideoCapture video;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
public:
    VideoCapture cap(string);
    void transform(VideoCapture);
};

class RunesDetector:public detector{
protected:
//    Mat image,binary;
public:
    void pre_process();
    void select_predict(int,Prediction &pd,Kalm &X,Kalm &Y);
    void predict(RotatedRect,Kalm &X,Kalm &Y);
};

class ImgRunesDetector:public RunesDetector{
public:
    void cap(string);
};

class MultipleArmorDetector:public detector
{
protected:
    Point2f points[4],pt[30],pt1[30],pt2[30],left_up[30],right_up[30],left_down[305],right_down[30];
    Mat smallImg, hsv_img, img,struct1;
    double k1,k2,d1,d2,x_distance_i,y_distance_i,area_i,x_distance_j,y_distance_j,area_j,areaRatio;
    RotatedRect rrect;
    int counter;
public:
    void pre_process();
    void find_contours();
    void matching(int);
};

class SingleArmorDetector:public MultipleArmorDetector{
public:
    void pre_process();
    void find_contours();
    void matching(int,Prediction &pd);
};
#endif //VISION_RUNESDETECTOR_H
