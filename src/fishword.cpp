#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <glob.h>
#include <sys/stat.h>

// 相機參數
const cv::Size DIM(1280, 960);

// const cv::Mat K = (cv::Mat_<double>(3, 3) << 355.06125392,   0,         638.97006582,
//    0,         354.24134759, 480.87469226,
//    0,           0,           1);

// const cv::Mat D = (cv::Mat_<double>(1, 4) << 0.04974578, -0.02019139,  0.01303418, -0.00325658);

const cv::Mat K = (cv::Mat_<double>(3, 3) << 354.69554042,   0,         639.31964898,
   0,         353.75291574, 481.37149087,
   0,           0,           1);

const cv::Mat D = (cv::Mat_<double>(1, 4) << 0.05199583, -0.02759001,  0.01995445, -0.00505114);

cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

void undistortFisheyePoints(const std::vector<cv::Point2f> &distortedPoints, std::vector<cv::Point2f> &undistortedPoints){
    // 校正像素座標
    // cv::fisheye::undistortPoints(distortedPoints, undistortedPoints, K, D);
    cv::fisheye::undistortPoints(distortedPoints, undistortedPoints, K, D, R, K);
}
void distortFisheyePoints(const std::vector<cv::Point2f> &undistortedPoints, std::vector<cv::Point2f> &distortedPoints){
    // 將校正後的像素座標轉換回魚眼像素座標
    cv::fisheye::distortPoints(undistortedPoints, distortedPoints, K, D);
}
// 將魚眼像素座標轉成世界座標的函式
cv::Point2f undistortedToWorld(const cv::Point2f &fisheyePoint, const cv::Mat &homographyMatrix){
    // 校正魚眼像素座標
    std::vector<cv::Point2f> distortedPoints = {fisheyePoint};
    std::vector<cv::Point2f> undistortedPoints;
    undistortFisheyePoints(distortedPoints, undistortedPoints);

    // 取校正後的座標點
    cv::Point2f undistortedPoint = undistortedPoints[0];

    // 將點轉換為齊次坐標
    cv::Mat pointMat = (cv::Mat_<double>(3, 1) << undistortedPoint.x, undistortedPoint.y, 1.0);
    // 應用單應性矩陣
    cv::Mat worldMat = homographyMatrix * pointMat;
    // 轉換回非齊次坐標
    return cv::Point2f(worldMat.at<double>(0, 0) / worldMat.at<double>(2, 0),
                       worldMat.at<double>(1, 0) / worldMat.at<double>(2, 0));
}

// 將世界坐標轉成魚眼像素座標的函式
cv::Point2f worldToFisheye(const cv::Point2f &worldPoint, const cv::Mat &homographyMatrix){
    // 將世界座標轉換為齊次坐標
    cv::Mat pointMat = (cv::Mat_<double>(3, 1) << worldPoint.x, worldPoint.y, 1.0);
    // 應用逆單應性矩陣
    cv::Mat invHomographyMatrix = homographyMatrix.inv();
    cv::Mat imageMat = invHomographyMatrix * pointMat;

    // 轉換回非齊次坐標
    cv::Point2f normPt(imageMat.at<double>(0, 0) / imageMat.at<double>(2, 0),
                       imageMat.at<double>(1, 0) / imageMat.at<double>(2, 0));

    cv::Point2f normalizedCameraPt((normPt.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                                   (normPt.y - K.at<double>(1, 2)) / K.at<double>(1, 1));

    // 應用畸變將其轉為魚眼影像座標
    std::vector<cv::Point2f> normPts = {normalizedCameraPt};
    std::vector<cv::Point2f> fisheyePts;
    distortFisheyePoints(normPts, fisheyePts);

    return fisheyePts[0];

    // return normPt;
}
// 定義全局變數來存儲單應性矩陣
cv::Mat homographyMatrix_front;
cv::Mat homographyMatrix_back;
cv::Mat homographyMatrix_left;
cv::Mat homographyMatrix_right;
std::vector<cv::Point2f> H_worldPoint_front;
std::vector<cv::Point2f> H_undistort_front;
std::vector<cv::Point2f> H_fishPoint_front;

std::vector<cv::Point2f> H_worldPoint_back;
std::vector<cv::Point2f> H_undistort_back;
std::vector<cv::Point2f> H_fishPoint_back;

std::vector<cv::Point2f> H_worldPoint_left;
std::vector<cv::Point2f> H_undistort_left;
std::vector<cv::Point2f> H_fishPoint_left;

std::vector<cv::Point2f> H_worldPoint_right;
std::vector<cv::Point2f> H_undistort_right;
std::vector<cv::Point2f> H_fishPoint_right;
//單應性矩陣副涵式
void gethomographyMatrix(){ 
    //front
    // 世界坐標點
    H_worldPoint_front.push_back(cv::Point2f(-300, 0));
    H_worldPoint_front.push_back(cv::Point2f(-300, 500));
    H_worldPoint_front.push_back(cv::Point2f(-300, 1000));
    H_worldPoint_front.push_back(cv::Point2f(300, 1000));
    H_worldPoint_front.push_back(cv::Point2f(300, 500));
    H_worldPoint_front.push_back(cv::Point2f(300, 0));
    // 魚眼像素座標點
    H_fishPoint_front.push_back(cv::Point2f(76, 581));
    H_fishPoint_front.push_back(cv::Point2f(442, 393));
    H_fishPoint_front.push_back(cv::Point2f(530, 367));
    H_fishPoint_front.push_back(cv::Point2f(742, 368));
    H_fishPoint_front.push_back(cv::Point2f(835, 393));
    H_fishPoint_front.push_back(cv::Point2f(1199,575));
    undistortFisheyePoints(H_fishPoint_front, H_undistort_front);
    homographyMatrix_front = cv::findHomography(H_undistort_front, H_worldPoint_front);

    //back
    // 世界坐標點
    H_worldPoint_back.push_back(cv::Point2f(300, -500));
    H_worldPoint_back.push_back(cv::Point2f(300, -1000));
    H_worldPoint_back.push_back(cv::Point2f(300, -1500));
    H_worldPoint_back.push_back(cv::Point2f(-300, -1500));
    H_worldPoint_back.push_back(cv::Point2f(-300, -1000));
    H_worldPoint_back.push_back(cv::Point2f(-300, -500));
    // 魚眼像素座標點
    H_fishPoint_back.push_back(cv::Point2f(65 , 623));
    H_fishPoint_back.push_back(cv::Point2f(383, 501));
    H_fishPoint_back.push_back(cv::Point2f(470, 477));
    H_fishPoint_back.push_back(cv::Point2f(666, 476));
    H_fishPoint_back.push_back(cv::Point2f(738, 497));
    H_fishPoint_back.push_back(cv::Point2f(1067,606));
    undistortFisheyePoints(H_fishPoint_back, H_undistort_back);
    homographyMatrix_back=cv::findHomography(H_undistort_back, H_worldPoint_back);

    //left
    // 世界坐標點
    H_worldPoint_left.push_back(cv::Point2f(-100, 0));
    H_worldPoint_left.push_back(cv::Point2f(-300, 0));
    H_worldPoint_left.push_back(cv::Point2f(-300, -500));
    H_worldPoint_left.push_back(cv::Point2f(-100, -500));
    // 魚眼像素座標點
    H_fishPoint_left.push_back(cv::Point2f(1071,705));
    H_fishPoint_left.push_back(cv::Point2f(834,484));
    H_fishPoint_left.push_back(cv::Point2f(265, 496));
    H_fishPoint_left.push_back(cv::Point2f(96, 609));
    undistortFisheyePoints(H_fishPoint_left, H_undistort_left);
    homographyMatrix_left = cv::findHomography(H_undistort_left, H_worldPoint_left);
    
    //right
    // 世界坐標點
    H_worldPoint_right.push_back(cv::Point2f(100, -500));
    H_worldPoint_right.push_back(cv::Point2f(300, -500));
    H_worldPoint_right.push_back(cv::Point2f(300, 0));
    H_worldPoint_right.push_back(cv::Point2f(100, 0));
    // 魚眼像素座標點
    H_fishPoint_right.push_back(cv::Point2f(1167,619));
    H_fishPoint_right.push_back(cv::Point2f(1001,496));
    H_fishPoint_right.push_back(cv::Point2f(429,468));
    H_fishPoint_right.push_back(cv::Point2f(184,695));
    undistortFisheyePoints(H_fishPoint_right, H_undistort_right);
    homographyMatrix_right = cv::findHomography(H_undistort_right, H_worldPoint_right);
}

// 去畸變影像函式
cv::Mat distortImage(const std::string &imgPath) {
    cv::Mat img = cv::imread(imgPath);
    cv::Mat map1, map2;
    cv::Mat undistortedImg;
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_64F), K, DIM, CV_16SC2, map1, map2);
    cv::remap(img, undistortedImg, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return  undistortedImg;
}


int main() {
    // cv::Mat image_front=distortImage("../input/front.png");
    // cv::Mat image_back=distortImage("../input/back.png");
    // cv::Mat image_left=distortImage("../input/left.png");
    // cv::Mat image_right=distortImage("../input/right.png");

    cv::Mat image_front=cv::imread("../input/front.png");
    cv::Mat image_back=cv::imread("../input/back.png");
    cv::Mat image_left=cv::imread("../input/left.png");
    cv::Mat image_right=cv::imread("../input/right.png");

    gethomographyMatrix();
    //cv::namedWindow("test",cv::WINDOW_NORMAL);

    //front
    for(int i=0;i<1001;i=i+50){
        for(int j=-300;j<301;j=j+50){
            cv::circle(image_front, worldToFisheye(cv::Point2f(static_cast<float>(j), static_cast<float>(i)), homographyMatrix_front), 3, cv::Scalar(0, 255, 255), -1);
        }
    }
    cv::circle(image_front, worldToFisheye(cv::Point2f(static_cast<float>(0), static_cast<float>(0)), homographyMatrix_front), 3, cv::Scalar(0, 0, 255), -1);

    //back
    for(int i=-1500;i<-499;i=i+50){
        for(int j=-300;j<301;j=j+50){
            cv::circle(image_back, worldToFisheye(cv::Point2f(static_cast<float>(j), static_cast<float>(i)), homographyMatrix_back), 3, cv::Scalar(0, 255, 255), -1);
        }
    }
    cv::circle(image_back, worldToFisheye(cv::Point2f(static_cast<float>(0), static_cast<float>(-500)), homographyMatrix_back), 3, cv::Scalar(0, 0, 255), -1);

    //left
    for(int i=-500;i<1;i=i+50){
        for(int j=-300;j<-99;j=j+50){
            cv::circle(image_left, worldToFisheye(cv::Point2f(static_cast<float>(j), static_cast<float>(i)), homographyMatrix_left), 3, cv::Scalar(0, 255, 255), -1);
        }
    }

    //right
    for(int i=-500;i<1;i=i+50){
        for(int j=100;j<301;j=j+50){
            cv::circle(image_right, worldToFisheye(cv::Point2f(static_cast<float>(j), static_cast<float>(i)), homographyMatrix_right), 3, cv::Scalar(0, 255, 255), -1);
        }
    }

    // std::cout << worldToFisheye(undistortedToWorld(cv::Point2f(65 , 623), homographyMatrix_back), homographyMatrix_back) << std::endl;
    // std::cout << worldToFisheye(undistortedToWorld(cv::Point2f(65 , 623), homographyMatrix_front), homographyMatrix_front) << std::endl;
    // std::cout << worldToFisheye(undistortedToWorld(cv::Point2f(65 , 623), homographyMatrix_left), homographyMatrix_left) << std::endl;
    // std::cout << worldToFisheye(undistortedToWorld(cv::Point2f(65 , 623), homographyMatrix_right), homographyMatrix_right) << std::endl;
    
    // cv::circle(image_front, H_undistort_front[0], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_front, H_undistort_front[1], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_front, H_undistort_front[2], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_front, H_undistort_front[3], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_front, H_undistort_front[4], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_front, H_undistort_front[5], 3, cv::Scalar(255, 0, 0), -1);

    // cv::circle(image_back, H_undistort_back[0], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_back, H_undistort_back[1], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_back, H_undistort_back[2], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_back, H_undistort_back[3], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_back, H_undistort_back[4], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_back, H_undistort_back[5], 3, cv::Scalar(255, 0, 0), -1);

    // cv::circle(image_left, H_undistort_left[0], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_left, H_undistort_left[1], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_left, H_undistort_left[2], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_left, H_undistort_left[3], 3, cv::Scalar(255, 0, 0), -1);

    // cv::circle(image_right, H_undistort_right[0], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_right, H_undistort_right[1], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_right, H_undistort_right[2], 3, cv::Scalar(255, 0, 0), -1);
    // cv::circle(image_right, H_undistort_right[3], 3, cv::Scalar(255, 0, 0), -1);

    cv::imwrite("../output/front-yp.jpg",image_front);
    cv::imwrite("../output/back-yp.jpg",image_back);
    cv::imwrite("../output/left-yp.jpg",image_left);
    cv::imwrite("../output/right-yp.jpg",image_right);

    //cv::imshow("test",image_front);
    //cv::waitKey(100000);
    cv::destroyAllWindows();

    return 0;
}