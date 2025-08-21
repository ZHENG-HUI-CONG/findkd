#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fisheye_simple_control.py
——————————————
滑桿 (Trackbar) 刻度：
  • Yaw         : -180°  …  +180°，每格 10°   (共 37 格)
  • Balance     : 0.0   …  1.0  ，每格 0.1   (共 11 格)
  • FOV-scale   : 0.1   …  1.0  ，每格 0.1   (共 10 格)

鍵盤：
  • q / Esc  離開
"""

import cv2
import numpy as np
import os, glob
from math import atan, degrees

# -------- 1. 你的 K、D -------------
K = np.array([[355.67876243,   0.        , 638.84520097],
              [  0.        , 354.84207135, 480.89256853],
              [  0.        ,   0.        ,   1.        ]], dtype=np.float64)
D = np.array([[ 0.0517522 ],
              [-0.0267777 ],
              [ 0.01933738],
              [-0.0052901 ]], dtype=np.float64)

# -------- 2. 讀一張影像當示例 ----------
img_path = sorted(glob.glob(os.path.join("val", "*.*")))[0]
img = cv2.imread(img_path)
if img is None:
    raise RuntimeError("找不到 val/ 影像")
h, w = img.shape[:2]

# -------- 3. 工具函式 -------------
def euler_to_R(yaw_deg=0):
    y = np.deg2rad(yaw_deg)
    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [        0, 1,        0],
                   [-np.sin(y), 0, np.cos(y)]])
    return Ry

def hfov_from_K(fx, img_w):
    """由焦距 fx 與影像寬計算水平視角 (deg)"""
    return 2 * degrees(atan((img_w / 2) / fx))

def redraw(yaw_deg, balance, fov_scale):
    R = euler_to_R(yaw_deg)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), R,
        balance=balance,
        new_size=(w, h),
        fov_scale=fov_scale
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, new_K, (w, h), cv2.CV_16SC2
    )
    und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # 在影像右上角顯示目前 HFOV
    hfov = hfov_from_K(new_K[0, 0], w)
    cv2.putText(und, f"HFOV: {hfov:.1f} deg",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0,255,0), 2, cv2.LINE_AA)
    return und

# -------- 4. Trackbar 回呼 -------------
def on_change(_=None):
    yaw_idx = cv2.getTrackbarPos("Yaw(10deg)",  win)
    bal_idx = cv2.getTrackbarPos("Balance",     win)
    fov_idx = cv2.getTrackbarPos("FOVscale",    win)

    yaw_deg = (yaw_idx - 18) * 10        # -180…+180
    balance = bal_idx / 10.0             # 0.0 … 1.0
    fov_s   = max(0.1, fov_idx / 10.0)   # 0.1 … 1.0

    out = redraw(yaw_deg, balance, fov_s)
    cv2.imshow(win, out)

# -------- 5. 視窗與滑桿 -------------
win = "Fisheye Viewer"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, w//2, h//2)

cv2.createTrackbar("Yaw(10deg)", win, 18, 36, on_change)   # 18 → 0°
cv2.createTrackbar("Balance",    win,  5, 10, on_change)   # 0.5
cv2.createTrackbar("FOVscale",   win,  9, 10, on_change)   # 0.9

on_change()  # 初始顯示

print("滑桿即時生效；按 q 或 Esc 離開")
while True:
    k = cv2.waitKey(50) & 0xFF
    if k in (27, ord('q')):
        break
cv2.destroyAllWindows()
