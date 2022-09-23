#include <iostream>
#include "opencv2/opencv.hpp"
#include <math.h>
#include "Kalm.h"

using namespace std;
using namespace cv;

void cap(string dir)
{
    VideoCapture video1(dir);
    VideoCapture video = video1;
}

void pre_process()
{
    Mat image,binary;
    image.copyTo(binary);
    resize(image,image,Size(image.cols*0.25,binary.rows*0.25));
    resize(binary,binary,Size(binary.cols*0.25,binary.rows*0.25));

    cvtColor(image,image,COLOR_BGR2GRAY);

    threshold(image, image, 80, 255, THRESH_BINARY);

    dilate(image,image,Mat());
    dilate(image,image,Mat());
    floodFill(image,Point(5,50),Scalar(255),0,FLOODFILL_FIXED_RANGE);

    threshold(image, image, 80, 255, THRESH_BINARY_INV);
}

void select()
{
    vector<vector<Point>> contours;
    findContours(image, contours, RETR_LIST, CHAIN_APPROX_NONE);
    for (size_t i = 0; i < contours.size(); i++){

        vector<Point> points;
        double area = contourArea(contours[i]);
        if (area < 50 || 1e4 < area) continue;
        drawContours(image, contours, static_cast<int>(i), Scalar(0), 2);
        imshow("binary",image);
        //找到轮廓的最小外接矩形
        points = contours[i];
        RotatedRect rrect = fitEllipse(points);
        cv::Point2f* vertices = new cv::Point2f[4];
        rrect.points(vertices);
        //通过最小外接矩形的宽高比筛选出目标矩形
        float aim = rrect.size.height/rrect.size.width;
        if(aim > 1.7 && aim < 2.6){
            for (int j = 0; j < 4; j++)
            {
                cv::line(binary, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0),4);
            }
            float middle = 100000;

            for(size_t j = 1;j < contours.size();j++){

                vector<Point> pointsA;
                double area = contourArea(contours[j]);
                if (area < 50 || 1e4 < area) continue;

                pointsA = contours[j];

                RotatedRect rrectA = fitEllipse(pointsA);

                float aimA = rrectA.size.height/rrectA.size.width;
                //筛选出宝剑叶片，计算叶片到矩形中心距离
                if(aimA > 3.0){
                    float distance = sqrt((rrect.center.x-rrectA.center.x)*(rrect.center.x-rrectA.center.x)+
                                          (rrect.center.y-rrectA.center.y)*(rrect.center.y-rrectA.center.y));

                    if (middle > distance  )
                        middle = distance;
                }
            }
            if( middle > 40){
                //cv::circle(binary,Point(rrect.center.x,rrect.center.y),15,cv::Scalar(0,255,0),4);
                X.kalmanFilterFun(rrect.center.x);
                Y.kalmanFilterFun(rrect.center.y);
                circle(binary, Point(X.info.predictValue, Y.info.predictValue),
                       5, Scalar(0, 0, 255), -1);
                cout<< X.info.predictValue<<" "<<Y.info.predictValue<<endl;
            }
        }
    }
}

Kalm X;
Kalm Y;
int main()
{

        imshow("frame",binary);
        //imshow("Original", image);
        waitKey(10);

}
