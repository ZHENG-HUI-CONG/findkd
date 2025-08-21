cmake_minimum_required(VERSION 3.10)

# 專案名稱
project(CameraCalibration)

# 專案所需的 C++ 標準
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 找到 OpenCV 包（確保你的系統已經安裝 OpenCV）
find_package(OpenCV REQUIRED)

# 將執行檔的名稱設置為 calibration
add_executable(findkd src/findkd.cpp)
add_executable(fishwold src/fishword.cpp)
# 將 OpenCV 的庫鏈接到執行檔
target_link_libraries(findkd ${OpenCV_LIBS})
target_link_libraries(fishwold ${OpenCV_LIBS})
