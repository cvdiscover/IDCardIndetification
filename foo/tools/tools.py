import cv2
import matplotlib.pylab as plt
import numpy as np
import copy
import os
import skimage.transform.radon_transform as transform
import math
from PIL import Image
from datetime import datetime


def get_u_d_l_r(rect):
    """
    获取rect的上下左右边界值
    :param rect: 矩形形式：x, y, w, h
    :return:
    """
    #
    upper, down = rect[1], rect[1] + rect[3]
    left, right = rect[0], rect[0] + rect[2]
    return upper, down, left, right


def intersect_height(y01, y02, y11, y12):
    """
    计算两个矩形在height轴方向交叉占比
    :param self:
    :param y01: 第一个矩形y1
    :param y02: 第一个矩形y2
    :param y11: 第二个矩形y1
    :param y12: 第二个矩形y2
    :return: height交叉段占比
    """
    row = float(min(y02, y12) - max(y01, y11))
    return max(row / float(y02 - y01), row / float(y12 - y11))


def remove_inside(rects=[]):
    """
    去除包含的矩形
    :param rects:
    :return:
    """
    i = 0
    while (i < len(rects)):

        rect_curr = rects[i]
        # 获取当前rect的上下左右边界信息
        u_curr, d_curr, l_curr, r_curr = get_u_d_l_r(rect_curr)
        # 判断当前rect是否在内部
        for rect_ in rects:
            u_, d_, l_, r_ = get_u_d_l_r(rect_)

            if u_curr > u_ and d_curr < d_ and l_curr > l_ and r_curr < r_:
                rects.remove(rect_curr)
                i -= 1
                break
        i += 1
    return rects


def merge(rects=[]):
    """
    合并同一行的矩形
    :param rects:
    :return:
    """
    i = 0
    while (i < len(rects)):

        rect_curr = rects[i]
        # 获取当前rect的上下左右边界信息
        u_curr, d_curr, l_curr, r_curr = get_u_d_l_r(rect_curr)
        # 判断当前rect是否在内部
        for rect_ in rects:
            if rect_ == rect_curr:
                continue
            u_, d_, l_, r_ = get_u_d_l_r(rect_)
            if intersect_height(u_curr, d_curr, u_, d_) > 0.8:
                new_rect = np.array(copy.deepcopy(rect_curr))

                new_rect[0] = min(rect_curr[0], rect_[0])
                new_rect[1] = min(rect_curr[1], rect_[1])
                new_rect[2] = max(rect_curr[0] + rect_curr[2], rect_[0] + rect_[2]) - new_rect[0]

                new_rect[3] = max(rect_curr[3], rect_[3])
                rects.remove(rect_curr)
                rects.remove(rect_)
                rects.append(list(new_rect))
                i -= 1
                break
        i += 1
    return rects


def separate_color(img):
    """
         提取黑色像素
         :param img: 图片
         :return: 根据颜色分割后的二值图片
         """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv.copy(), np.array([0, 0, 0]), np.array([180, 255, 180]))
    return img_mask


def morphology(img, scale, is_date=0):
    """
          对图片进行形态学处理（膨胀腐蚀）
          :param img: 图片
          :param scale: 倍数
          :param is_date :是否是出生日期
          :return: 膨胀腐蚀后的图片
          """
    # scale = int(img.shape[0] / 40)
    if is_date == 1:
        dilate_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        img_dilate = cv2.dilate(img, dilate_elment, iterations=1)
        return img_dilate

    if scale == 1:
        erode_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilate_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))
    elif scale == 2:
        erode_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilate_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    elif scale == 3:
        erode_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        dilate_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    else:
        erode_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilate_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    # 腐蚀图像
    img_erode = cv2.erode(img, erode_elment, iterations=1)
    #  膨胀图像
    img_dilate = cv2.dilate(img_erode, dilate_elment, iterations=1)
    return img_dilate


def cal_cross_ratio(y11, y12, y21, y22):
    """
    计算交并比
    :param y11: 第一个矩形上边位置
    :param y12: 第一个矩形下边位置
    :param y21: 第二个矩形上边位置
    :param y22: 第二个矩形下边位置
    :return:
    """
    return (min(y12, y22) - max(y11, y21)) / (max(y12, y22) - min(y11, y21))