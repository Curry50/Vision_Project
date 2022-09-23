//
// Created by z on 22-7-10.
//
#include "Detector.h"
#include "opencv2/opencv.hpp"
#include "Predict.h"
#include "Kalm.h"
using namespace cv;
using namespace std;
using namespace dnn;


VideoCapture detector::cap(string dir)
{
    VideoCapture video1(dir);
    video = video1;
    return video;
}

void detector::transform(VideoCapture video)
{
    video >> image;
}

void RunesDetector::pre_process() {
    Mat struct1 = getStructuringElement(0, Size(3, 3));
    binary = image;
    resize(image,image,Size(image.cols*1.2,image.rows*1.2));//1.2
    resize(binary,binary,Size(binary.cols*1.2,binary.rows*1.2));
    split(image, channels);
    Mat diff = channels[2] - channels[0] != channels[2] - channels[2];
    bool eq = countNonZero(diff) != 0;
    if (eq) {
        channels[2] = channels[2] - channels[0];
    }
    Mat diff_2 = channels[1] - channels[0] != channels[2] - channels[2];
    bool eq_2 = countNonZero(diff_2) != 0;
    if (eq_2) {
        channels[1] = channels[1] - channels[0];
    }
    merge(channels, 3, image);
    imshow("inv_binary_1",image);
    cvtColor(image,image,COLOR_BGR2GRAY);
    imshow("inv_binary_2",image);
    threshold(image, image, 10, 255, THRESH_BINARY);//80;
    imshow("inv_binary",image);
    dilate(image,image,struct1);
    dilate(image,image,struct1);
    floodFill(image,Point(5,50),Scalar(255),0,FLOODFILL_FIXED_RANGE);
    threshold(image, image, 80, 255, THRESH_BINARY_INV);
}

void RunesDetector::select_predict(int m,Prediction &pd,Kalm &X,Kalm &Y) {
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
            //cout << middle <<endl;
            if( middle > 90){//90
                RunesDetector::predict(rrect,X,Y);
                pd.prediction(m,binary,rrect.center);
            }
        }
    }
    imshow("frame",binary);
}

void RunesDetector::predict(RotatedRect rrect,Kalm &X,Kalm &Y) {


    circle(binary,Point(rrect.center.x,rrect.center.y),6,cv::Scalar(0,0,255),-1);
    X.kalmanFilterFun(rrect.center.x);
    Y.kalmanFilterFun(rrect.center.y);
    circle(binary, Point(X.info.predictValue, Y.info.predictValue),6, Scalar(255, 0, 0), -1);
}

void ImgRunesDetector::cap(string dir)
{
    image = imread(dir);
}

void MultipleArmorDetector::pre_process() {
//    video >> image;
    struct1 = getStructuringElement(0, Size(3, 3));
    //红蓝通道相减
    split(image, channels);
    Mat diff = channels[2] - channels[0] != channels[2] - channels[2];
    bool eq = countNonZero(diff) != 0;
    if (eq) {
        channels[2] = channels[2] - channels[0];
    }
    Mat diff_2 = channels[1] - channels[0] != channels[2] - channels[2];
    bool eq_2 = countNonZero(diff_2) != 0;
    if (eq_2) {
        channels[1] = channels[1] - channels[0];
    }
    merge(channels, 3, image);
    resize(image, smallImg, Size(), 0.5, 0.5, INTER_AREA);
    cvtColor(smallImg, hsv_img, COLOR_BGR2HSV);	//转换颜色空间
    inRange(hsv_img, Scalar(100, 43, 46), Scalar(124, 255, 255), img);//二值化
    dilate(img, img, struct1);//膨胀
    erode(img, img, struct1);//腐蚀
    imshow("binary", img);
}

void MultipleArmorDetector::find_contours() {
    counter = 0;
    findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//寻找轮廓
    for (int t = 0; t < contours.size(); t++) {
        Rect rect = boundingRect(contours[t]);
        double aspectRatio = double(rect.width) / double(rect.height);
        double area = contourArea(contours[t]);
        float AreaRatio = contourArea(contours[t]) / (rect.width * rect.height);
        if (aspectRatio <= 0.65 && aspectRatio > 0.2 && AreaRatio > 0.1) { //通过外接矩形的宽高比、面积比进行第一轮筛选
            //rectangle(smallImg, rect, Scalar(0, 0, 255), 2, 8, 0);
            rrect = minAreaRect(contours[t]);
            rrect.points(points);
            //存储最小外接矩形的边界点和中心点，便于后续灯条配对
            pt[counter] = (points[0] + points[2]) / 2;
            pt1[counter] = (points[0] + points[1]) / 2;
            pt2[counter] = (points[2] + points[3]) / 2;
            left_up[counter] = points[0];
            right_up[counter] = points[1];
            right_down[counter] = points[2];
            left_up[counter] = points[3];
            counter += 1;
        }
    }

}

void MultipleArmorDetector::matching(int m) {
    for (int i = 0; i < counter; i++) {
        for (int j = 1 + i; j < counter; j++) {
            Prediction pd;
            //进行灯条配对，通过灯条的斜率和灯条间的相对距离以及灯条的面积大小进行第二轮筛选
            k1 = double(pt[i].y - pt[j].y) / double(pt[i].x - pt[j].x);
            k2 = double(pt1[i].x - pt2[i].x) / double(pt1[i].y - pt2[i].y);
            d1 = sqrt(pow(pt1[i].x - pt2[i].x, 2) + pow(pt1[i].y - pt2[i].y, 2));
            d2 = sqrt(pow(pt[i].x - pt[j].x, 2) + pow(pt[i].y - pt[j].y, 2));
            x_distance_i = sqrt(pow(left_up[i].x - right_up[i].x, 2) + pow(right_up[i].y - left_up[i].y, 2));
            y_distance_i = sqrt(pow(left_up[i].x - left_down[i].x, 2) + pow(left_up[i].y - left_down[i].y, 2));
            area_i = x_distance_i * y_distance_i;
            x_distance_j = sqrt(pow(left_up[j].x - right_up[j].x, 2) + pow(right_up[j].y - left_up[j].y, 2));
            y_distance_j = sqrt(pow(left_up[i].x - left_down[i].x, 2) + pow(left_up[i].y - left_down[i].y, 2));
            area_j = x_distance_j * y_distance_j;
            areaRatio = (area_i) / (area_j);

            if (k1 - k2 < 0.08 && k1-k2 > -0.07 && d1 / d2 > 0.2 + 0.2  && d1 / d2 < 0.6 && areaRatio > 0.8 &&
                areaRatio < 1.2) {
                line(smallImg, pt[i], pt[j], Scalar(0, 255, 0), 2, 8, 0);
                Point2f center = (pt[i] + pt[j]) / 2;
                pd.prediction(m, smallImg, center);//预测
            }

        }
    }
    imshow("result", smallImg);
}

void SingleArmorDetector::pre_process(){
    struct1 = getStructuringElement(0, Size(3, 3));
    resize(image, smallImg, Size(), 0.6, 0.6, INTER_AREA);
    cvtColor(smallImg, hsv_img, COLOR_BGR2HSV);	//转换颜色空间
    inRange(hsv_img, Scalar(26, 43, 46), Scalar(34, 255, 255), img);//二值化
    dilate(img, img, struct1);//膨胀
    erode(img, img, struct1);//腐蚀
    imshow("binary", img);
}

void SingleArmorDetector::find_contours()
{
    counter = 0;
    findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//寻找轮廓
    for (int t = 0; t < contours.size(); t++) {
        Rect rect = boundingRect(contours[t]);
        double aspectRatio = double(rect.width) / double(rect.height);
        double area = contourArea(contours[t]);
        float AreaRatio = contourArea(contours[t]) / (rect.width * rect.height);
        if (aspectRatio <= 0.9 && area > 30) {
            //rectangle(smallImg, rect, Scalar(0, 0, 255), 2, 8, 0);
            rrect = minAreaRect(contours[t]);
            rrect.points(points);
            pt[counter] = (points[0] + points[2]) / 2;
            pt1[counter] = (points[0] + points[1]) / 2;
            pt2[counter] = (points[2] + points[3]) / 2;
            left_up[counter] = points[0];
            right_up[counter] = points[1];
            right_down[counter] = points[2];
            left_up[counter] = points[3];
            counter += 1;
        }
    }
}

void SingleArmorDetector::matching(int m,Prediction &pd) {
    for (int i = 0; i < counter; i++) {
        for (int j = 1 + i; j < counter; j++) {
            int x_coor = min(pt1[i].x, pt1[j].x);
            int y_coor = min(pt1[i].y, pt2[i].y);
            auto height = max(rrect.size.height,rrect.size.width);
            Rect rect(x_coor,y_coor-height,abs(pt[j].x-pt[i].x)*1.2+1,height*2+1);
            Mat ROI = smallImg(rect);
            ROI.convertTo(ROI, CV_64F, 1.0 / 255, 0);
            pow(ROI, 0.4, ROI);
            ROI.convertTo(ROI, CV_8U, 255, 0);
            imshow("ROI",ROI);
            cvtColor(ROI,ROI,COLOR_BGR2GRAY);
            imshow("ROI_1",ROI);
            String modelFile = "/home/z/Desktop/git/Vision/Vision-Video/model/MnistCNN.onnx";
            dnn::Net net = cv::dnn::readNetFromONNX(modelFile); //读取网络和参数

            Mat inputBlob = blobFromImage(ROI, 1.0/255, Size(28, 28), Scalar(0, 0, 0), true, false);

            net.setInput(inputBlob); //输入图像

            Mat result = net.forward(); //前向计算


            double minValue, maxValue;
            int labels[10] = {0,1,2,3,4,5,6,7,8,9};
            cv::Point minIdx, maxIdx;
            cv::minMaxLoc(result, &minValue, &maxValue, &minIdx, &maxIdx);

            int res = labels[maxIdx.x];
            cout << "prediction " << res <<endl;
            putText(smallImg, to_string(res),Point(x_coor,y_coor),FONT_HERSHEY_COMPLEX,2,Scalar(255,0,0),2);
            line(smallImg, pt[i], pt[j], Scalar(0, 255, 0), 2, 8, 0);
            circle(smallImg,Point2f((pt[i]+pt[j])/2),
                   5,Scalar(255,0,0),-1);
            Point2f center = (pt[i] + pt[j]) / 2;
            pd.prediction(m, smallImg, center);//预测
        }
    }
    imshow("result", smallImg);

}



