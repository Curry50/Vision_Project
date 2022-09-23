//
// Created by z on 9/22/22.
//
#include "NumberClassify.h"
void classification::classify(Mat ROI){

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
}