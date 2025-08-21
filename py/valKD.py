#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
undistort_val_display_expanded.py — 去畸變並擴大顯示範圍
使用 estimateNewCameraMatrixForUndistortRectify 調整輸出 FOV
"""

import os
import glob
import cv2
import numpy as np

# --- 1. 你的 K 和 D （請確認與先前校正結果一致） ---
K = np.array([[355.67876243,   0.        , 638.84520097],
              [  0.        , 354.84207135, 480.89256853],
              [  0.        ,   0.        ,   1.        ]], dtype=np.float64)
D = np.array([[ 0.0517522 ],
              [-0.0267777 ],
              [ 0.01933738],
              [-0.0052901 ]], dtype=np.float64)

# --- 2. 讀取範例影像取得尺寸 ---
input_dir = "val"
paths = sorted(glob.glob(os.path.join(input_dir, "*.*")))
if not paths:
    raise RuntimeError(f"找不到 {input_dir} 中的影像")
sample = cv2.imread(paths[0])
h, w = sample.shape[:2]

# --- 3. 計算新的相機矩陣以擴大視野 ---
# balance: 0.0 → 最小裁切（畫面邊緣黑邊最少）
#          1.0 → 保留最大 FOV（可能有明顯黑邊）
balance   = 0.8
# fov_scale: >1.0 可再放大視角（有時會更黑）
fov_scale = 0.8

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3),
    balance=balance, new_size=(w, h), fov_scale=fov_scale
)

# 產生去畸變 mapping
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K,
    (w, h), cv2.CV_16SC2
)

# --- 4. 逐張去畸變並顯示 ---
for img_path in paths:
    img = cv2.imread(img_path)
    if img is None:
        continue

    undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # 並排顯示
    combined = np.hstack([cv2.resize(img, (w//2, h//2)),
                          cv2.resize(undist, (w//2, h//2))])
    cv2.imshow('原圖 (左)  vs.  去畸變擴展 (右)', combined)

    print(f"Showing: {os.path.basename(img_path)}  — press any key")
    cv2.waitKey(0)

cv2.destroyAllWindows()
