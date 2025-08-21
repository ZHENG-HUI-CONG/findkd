#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_undistort_bigcanvas.py
一次把 1280×960、HFOV196° 魚眼攤平到大畫布
‧ 直接修改檔頭常數即可改輸入、輸出、畫布倍率、balance、fov_scale
"""

import cv2
import numpy as np
import os

# =========【自己改這裡】=========
INPUT_IMG  = "val/front.jpg"           # 輸入影像
OUTPUT_IMG = "front_undist.jpg"        # 輸出影像

SCALE      = 2.0   # 畫布放大倍率 (>=1)
BALANCE    = 1.0   # 0~1；越大保留視野多
FOV_SCALE  = 1.0   # 0.1~1.0；越小再裁切一些

# 相機 K、D（直接貼上校正結果）
K = np.array([[355.67876243,   0.        , 638.84520097],
              [  0.        , 354.84207135, 480.89256853],
              [  0.        ,   0.        ,   1.        ]], dtype=np.float64)

D = np.array([[ 0.0517522 ],
              [-0.0267777 ],
              [ 0.01933738],
              [-0.0052901 ]], dtype=np.float64)
# ===============================

# 讀影像
img = cv2.imread(INPUT_IMG)
if img is None:
    raise RuntimeError(f"讀檔失敗：{INPUT_IMG}")
h0, w0 = img.shape[:2]

# 大畫布尺寸
w_out, h_out = int(w0 * SCALE), int(h0 * SCALE)

# 新相機參數
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w0, h0), np.eye(3),
    balance=BALANCE,
    new_size=(w_out, h_out),
    fov_scale=FOV_SCALE
)

# 產生映射並去畸變
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K,
    (w_out, h_out), cv2.CV_16SC2
)
undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

# 存檔並顯示
cv2.imwrite(OUTPUT_IMG, undist)
print(f"✅ 已輸出：{os.path.abspath(OUTPUT_IMG)}")

cv2.imshow("Undistorted BigCanvas", undist)
cv2.waitKey(0)
cv2.destroyAllWindows()
