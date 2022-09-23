//
// Created by z on 7/25/22.
//
#include <opencv2//opencv.hpp>
#include "Predict.h"
#include "iostream"
#include "Detector.h"
using namespace std;
using namespace cv;
void Prediction::prediction(int k,Mat &image,Point2f pt)
{
    if(k<2)
    {
        coordinate_x[k]=pt.x;
        coordinate_y[k]=pt.y;
    }
    else{
        float delta_x = coordinate_x[1] - coordinate_x[0];//x轴方向单位时间的变化量
        float delta_y = coordinate_y[1] -coordinate_y[0];//y轴方向单位时间的变化量
        coordinate_x[2] = pt.x + delta_x*2;//预测点的x坐标
        coordinate_y[2] = pt.y + delta_y*2;//预测点的y坐标
        cout << pt.x << " " << delta_x << endl;
        circle(image,Point(coordinate_x[2],coordinate_y[2]),
               6,Scalar(0,255,0),-1);
        //更新坐标
        for(int m=0;m<1;m++)
        {
            coordinate_x[m] = coordinate_x[m+1];
            coordinate_y[m] = coordinate_y[m+1];
        }
        coordinate_x[1] = pt.x;
        coordinate_y[1] = pt.y;
    }
}
