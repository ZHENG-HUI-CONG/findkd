#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fisheye_point_converter.py
--------------------------
輸入魚眼影像 (u,v) 像素座標 → 輸出去畸變後 (u',v') 座標
依據已校正的內參 K 與畸變 D 計算
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np

# 1. 你的相機內參 (K) 與畸變參數 (D) －－請替換成校正結果
K = np.array([[355.67876243,   0.        , 638.84520097],
              [  0.        , 354.84207135, 480.89256853],
              [  0.        ,   0.        ,   1.        ]], dtype=np.float64)

D = np.array([[ 0.0517522 ],
              [-0.0267777 ],
              [ 0.01933738],
              [-0.0052901 ]], dtype=np.float64)

IMG_W, IMG_H = 1280, 960   # 影像解析度

# 2. 建立 GUI 介面
root = tk.Tk()
root.title("魚眼座標轉換工具")

main = ttk.Frame(root, padding=20)
main.grid()

# 輸入欄位
ttk.Label(main, text="u (0–1279):").grid(row=0, column=0, sticky="e")
u_entry = ttk.Entry(main, width=10)
u_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(main, text="v (0–959):").grid(row=1, column=0, sticky="e")
v_entry = ttk.Entry(main, width=10)
v_entry.grid(row=1, column=1, padx=5, pady=5)

# 輸出欄位
ttk.Label(main, text="校正後 u':").grid(row=2, column=0, sticky="e")
u_corr_var = tk.StringVar(value="—")
ttk.Label(main, textvariable=u_corr_var).grid(row=2, column=1)

ttk.Label(main, text="校正後 v':").grid(row=3, column=0, sticky="e")
v_corr_var = tk.StringVar(value="—")
ttk.Label(main, textvariable=v_corr_var).grid(row=3, column=1)

def convert():
    try:
        u = int(u_entry.get())
        v = int(v_entry.get())
    except ValueError:
        messagebox.showerror("格式錯誤", "請輸入整數座標！")
        return

    if not (0 <= u < IMG_W and 0 <= v < IMG_H):
        messagebox.showerror("範圍錯誤", f"u 應在 0~{IMG_W-1}，v 應在 0~{IMG_H-1}")
        return

    # 3. 使用 OpenCV fisheye.undistortPoints 計算
    pt = np.array([[[u, v]]], dtype=np.float64)     # (1,1,2)
    undist = cv2.fisheye.undistortPoints(pt, K, D, R=np.eye(3), P=K)
    u_corr, v_corr = undist[0,0]

    # 顯示結果（保留 3 位小數）
    u_corr_var.set(f"{u_corr:.3f}")
    v_corr_var.set(f"{v_corr:.3f}")

ttk.Button(main, text="轉換", command=convert).grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
