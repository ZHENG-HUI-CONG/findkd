#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

// 定義棋盤格尺寸
const int CHECKERBOARD_ROWS = 6;  // 行數（棋盤格的內角點）
const int CHECKERBOARD_COLS = 8;  // 列數（棋盤格的內角點）

struct KD {
    // 相機內參矩陣與失真係數
    cv::Mat K;
    cv::Mat D;
}kd;

struct KD findKD() {   //(std::string& path) {
    // 定義棋盤格的內角點數量
    cv::Size patternSize(CHECKERBOARD_COLS, CHECKERBOARD_ROWS);

    // 儲存校正用圖像中的3D世界座標和2D圖像座標點
    std::vector<std::vector<cv::Point3f>> objectPoints; // 世界座標系中的3D點
    std::vector<std::vector<cv::Point2f>> imagePoints;  // 圖像座標系中的2D點

    // 準備棋盤格的3D世界座標點（假設棋盤格大小為1單位）
    std::vector<cv::Point3f> objectCorners;
    for (int i = 0; i < CHECKERBOARD_ROWS; ++i) {
        for (int j = 0; j < CHECKERBOARD_COLS; ++j) {
            objectCorners.push_back(cv::Point3f(j, i, 0.0f));
        }
    }

    // 讀取校正用的多張圖像（在實際應用中，應該使用自己的圖片）
    std::vector<cv::String> images;
    cv::glob("../calibration_images/*.jpg", images); // 讀取所有棋盤格圖像

    cv::Size imageSize;
    for (const auto& imageFile : images) {
        cv::Mat image = cv::imread(imageFile);
        if (image.empty()) {
            std::cerr << "Could not open image: " << imageFile << std::endl;
            continue;
        }

        // 轉換為灰度圖
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // 找到棋盤格的內角點
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, patternSize, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // 進一步提高角點精度
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

            // 當找到所有角點後，儲存這些點
            imagePoints.push_back(corners);
            objectPoints.push_back(objectCorners);

           // 畫出角點並顯示
        //    cv::drawChessboardCorners(image, patternSize, corners, found);
        //    cv::imshow("Chessboard Corners", image);
        //    cv::waitKey(10);  // 短暫顯示每張棋盤格圖像
        }
    }

    std::vector<cv::Mat> rvecs, tvecs;  // 儲存旋轉向量和位移向量

    // 進行相機標定
    imageSize = cv::imread(images[0]).size(); // 使用第一張圖片的大小
    double rms = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, kd.K, kd.D, rvecs, tvecs,
        cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
        // cv::fisheye::CALIB_CHECK_COND |
        cv::fisheye::CALIB_FIX_SKEW);

    std::cout << "RMS error: " << rms << std::endl;

    cv::destroyAllWindows();//
    cv::FileStorage fs("../kd.yml", cv::FileStorage::WRITE); // 修改此為儲存檔案的路徑
    if (fs.isOpened()) {
        fs << "K" << kd.K;
        fs << "D" << kd.D;
        fs.release();
    }
    return kd;
}

int main()
{
    //std::string& path="../calibration_images/*.jpg";
    kd=findKD();
    std::cout << "Camera matrix (K):" << std::endl << kd.K << std::endl;
    std::cout << "Distortion coefficients (D):" << std::endl << kd.D << std::endl;
    return 0;
}

