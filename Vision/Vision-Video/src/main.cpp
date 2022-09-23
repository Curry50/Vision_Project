#include "Detector.h"
#include "Predict.h"
#include <string>
#include "Kalm.h"
using namespace std;

string input;
Kalm X;
Kalm Y;
Prediction pd;

int main() {
    cin >> input;
    if(input == "MAD")
    {
        MultipleArmorDetector MAD;
        VideoCapture video = MAD.cap("/home/z/Desktop/git/Vision/Vision-Video/video/1920_1.avi");
        for(int k=0;;k++)
        {
            MAD.transform(video);
            MAD.pre_process();//图像预处理
            MAD.find_contours();//寻找轮廓
            MAD.matching(k);//匹配灯条
            waitKey(30);
        }
    }
    else if(input == "SAD")
    {
        SingleArmorDetector SAD;
        VideoCapture video = SAD.cap("/home/z/Desktop/git/Vision/Vision-Video/video/1.avi");
        for(int k=0;;k++)
        {
            SAD.transform(video);
            SAD.pre_process();//图像预处理
            SAD.find_contours();//寻找轮廓
            SAD.matching(k,pd);//匹配灯条
            waitKey(120);
        }
    }
    else if(input == "RD")
    {
        RunesDetector RD;
        VideoCapture video = RD.cap("/home/z/Desktop/git/Vision/Vision-Video/video/Blue.avi");
        for(int k=0;;k++)
        {
            RD.transform(video);
            RD.pre_process();//图像预处理
            RD.select_predict(k,pd,X,Y);
            waitKey(100);
        }
    }
    else if(input == "IRD")
    {
        ImgRunesDetector IRD;
        int k = 0;
        IRD.cap("/home/z/Desktop/git/Vision/Vision-Video/video/Img.png");
        IRD.pre_process();
        IRD.select_predict(k,pd,X,Y);
        waitKey(0);
    }
}

