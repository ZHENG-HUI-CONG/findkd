#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fnidKD.py — 魚眼鏡頭內參與畸變參數校正
HFOV = 196°
影像尺寸 = 1280×960
棋盤格：8×6 內角點，格邊長 38 mm
"""

import glob
import numpy as np
import cv2
import os

# --- 1. 參數設定 ---
images_dir     = "calib_images"                      # 校正影像資料夾
images_pattern = os.path.join(images_dir, "*.jpg")   # 檔名模式
CHECKERBOARD   = (8, 6)    # 內角點數 (橫 8, 直 6)
square_size    = 38.0      # 方格邊長 (mm)
img_width      = 1280
img_height     = 960

# 角點精煉終止條件
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30, 1e-6
)

# fisheye 校正 flags (不使用 CALIB_CHECK_COND)
flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
    cv2.fisheye.CALIB_FIX_SKEW
)

# --- 2. 準備世界座標 (object points) ---
# 注意 shape 要是 (N,1,3)，dtype 要是 float32 或 float64
objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 1, 3),
    dtype=np.float32
)
# 填入 x,y ；z 預設為 0
objp[:, 0, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2) * square_size

objpoints = []  # List of (N,1,3) arrays
imgpoints = []  # List of (N,1,2) arrays

# --- 3. 角點偵測並收集座標 ---
for fname in glob.glob(images_pattern):
    img = cv2.imread(fname)
    if img is None:
        print(f"[WARN] 讀取失敗：{fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 全域 findChessboardCorners
    chess_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_FAST_CHECK
    )
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, chess_flags
    )
    if not ret:
        print(f"[INFO] {os.path.basename(fname)}：找不到角點")
        continue

    # 亞像素精煉
    corners = cv2.cornerSubPix(
        gray, corners, (3, 3), (-1, -1), criteria
    )

    # 收集物點與像點
    objpoints.append(objp.copy())
    imgpoints.append(corners)

    # 顯示配對結果
    cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
    cv2.imshow('Corners', img)
    cv2.waitKey(50)

cv2.destroyAllWindows()

# 確認有足夠張數
n_ok = len(objpoints)
print(f"成功配對 {n_ok} 張影像")
if n_ok < 10:
    raise RuntimeError("配對影像太少，建議至少 10 張，不同姿態的棋盤格影像")

# --- 4. 執行魚眼校正 ---
K = np.zeros((3, 3), dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)
rvecs = []
tvecs = []

# calibrate 會回傳 (rms, K, D, rvecs, tvecs)
rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    (img_width, img_height),
    K, D, rvecs, tvecs,
    flags,
    criteria
)

# 輸出結果
print(f"\nCalibration RMS error: {rms:.6f}")
print("內參矩陣 K =")
print(K)
print("\n畸變參數 D =")
print(D.ravel())

# --- 5. 選擇性：示範去畸變 ---
sample = cv2.imread(glob.glob(images_pattern)[0])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), K,
    (img_width, img_height), cv2.CV_16SC2
)
undistorted = cv2.remap(
    sample, map1, map2, interpolation=cv2.INTER_LINEAR
)
cv2.imshow('Undistorted', undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
