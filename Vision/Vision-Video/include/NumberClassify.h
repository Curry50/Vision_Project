//
// Created by z on 9/22/22.
//

#ifndef IMGTEST_NUMBERCLASSIFY_H
#define IMGTEST_NUMBERCLASSIFY_H
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace dnn;

class classification {
protected:
public:
    void classify(Mat);
};


#endif //IMGTEST_NUMBERCLASSIFY_H
