import cv2
import copy
import numpy as np
import matplotlib.pylab as plt
import math
import os
import matplotlib.pylab as plt
import dlib
from PIL import Image
from datetime import datetime
from foo.tools.back_correct_skew import back_correct_skew
from foo.tools.config import *
from foo.tools.tools import *


def box_get_back(img, save_name, imgHeight, imgWidth):
    """
        获取反面文字位置
        :param img: 图片
        :param imgHeight: 图片高度
        :param imgWidth: 图片宽度
        :param index: 图片序号
        :return: 文字切片

        """
    regions = []
    # 签发机关
    issuing_authority_height = int(imgHeight / 1.5)
    issuing_authority_width = int(imgWidth / 2.5)
    issuing_authority_addHeight = int(imgHeight / 1.5 + imgHeight / 6)
    issuing_authority_addWidth = int(imgWidth / 2.5 + imgWidth / 2)
    img_issuing_authority = img[issuing_authority_height:issuing_authority_addHeight,
                            issuing_authority_width:issuing_authority_addWidth]
    # plt.imshow(img, cmap=plt.gray())
    # plt.show()
    scale = int(img.shape[1] / 500)
    # print(scale, img.shape)
    issuing_authority_region = get_regions(img_issuing_authority, scale, is_front = 0)
    issuing_authority_region[0][0] += issuing_authority_width
    issuing_authority_region[0][1] += issuing_authority_height
    regions.append(issuing_authority_region)
    # 有效日期
    effective_date_height = int(imgHeight / 1.25)
    effective_date_width = int(imgWidth / 2.5)
    effective_date_addHeight = int(imgHeight / 1.25 + imgHeight / 6)
    effective_date_addWidth = int(imgWidth / 2.5 + imgWidth / 2)
    img_effective_date = img[effective_date_height:effective_date_addHeight,
                         effective_date_width:effective_date_addWidth]
    # plt.imshow(img_effective_date, cmap=plt.gray())
    # plt.show()
    scale = int(img.shape[1] / 500)
    # print(scale, img.shape)
    effective_date_region = get_regions(img_effective_date, scale, is_front = 0)
    effective_date_region[0][0] += effective_date_width
    effective_date_region[0][1] += effective_date_height
    regions.append(effective_date_region)

    img_copy = copy.deepcopy(img)
    for i in range(len(regions)):
        # rect[0],rect[1],rect[2],rect[3]  分别为矩形左上角点的横坐标，纵坐标，宽度，高度
        for j in range(len(regions[i])):
            rect = regions[i][j]
            x1, x2 = rect[0], rect[0] + rect[2]
            y1, y2 = rect[1], rect[1] + rect[3]
            # print(w1,h1,w2,h2)
            # w1, h1, w2, h2 =  8,4,256,49
            box = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
            cv2.drawContours(img_copy, np.array([box]), 0, (0, 255, 0), 2)
    if is_debug == 1:
        plt.imshow(img_copy, cmap=plt.gray())
        plt.show()
    # plt.imshow(img_copy, cmap=plt.gray())
    # plt.show()
    cv2.imencode('.jpg', img_copy)[1].tofile(str(save_name))


def get_regions(img, scale, is_address=0, is_name=0, is_date=0, is_front = 1, is_consider_color = 1):
    """
        对单块区域处理后，获取文文本位置
        :param img: 图片
        :param is_address: 是否是地址
        :return: 文本位置
        """
    img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    # _, dst_binary = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)
    # if is_address:
    #     _, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV )
    # else:
    th, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #_,img_binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV )
    # 根据背景文字颜色为蓝色特点去除背景文字

    if is_consider_color == 1:
        if is_address == 1:
            img_binary[(np.where(img[:, :, 0] > img[:, :, 1] + 10) or np.where(img[:, :, 0] > img[:, :, 2] + 10))] = 0
        elif is_address == 0 and is_front == 1:
            img_binary[np.where(img[:, :, 0] > img[:, :, 1] + 10)] = 0
            img_binary[np.where(img[:, :, 0] > img[:, :, 2] + 10)] = 0

    # print(is_address,th)
    # _, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV )
    # img_binary = separate_color(img)

    if is_address:
        # plt.imshow(img_binary, cmap=plt.gray())
        # plt.show()
        for i in reversed(range(img.shape[1] - 20, img.shape[1])):
            if cv2.countNonZero(img_binary[:, i]) > 25:
                img_binary[:, i] = 0

        # plt.imshow(img_binary, cmap=plt.gray())
        # plt.show()
    # plt.imshow(img_binary, cmap=plt.gray())
    # plt.show()
    #  进行形态学开运算
    img_open = morphology(img_binary, scale, is_date)
    # if is_address:
    #     plt.imshow(img_open, cmap=plt.gray())
    #     plt.show()
    # plt.imshow(img_open, cmap=plt.gray())
    # plt.show()
    # 获取文字区域位置
    text_region = find_word_regions(img_open, is_address, is_name=is_name, is_date=is_date)
    # if is_address == 1:
    #     print(text_region)
    # 在原图片上绘制文字区域矩形和剪切文字区域
    for i in range(len(text_region)):
        # rect[0],rect[1],rect[2],rect[3]  分别为矩形左上角点的横坐标，纵坐标，宽度，高度
        rect = text_region[i]
        x1, x2 = rect[0], rect[0] + rect[2]
        y1, y2 = rect[1], rect[1] + rect[3]
        # print(w1,h1,w2,h2)
        # w1, h1, w2, h2 =  8,4,256,49
        box = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
        cv2.drawContours(img, np.array([box]), 0, (0, 255, 0), 2)
    # plt.imshow(img, cmap=plt.gray())
    # plt.show()
    return np.array(text_region)


def find_word_regions(img, is_address=0, is_name=0, is_date=0):
    """
       获取一个二值图片中的文本位置
       :param img: 图片
       :param is_address: 是否是地址
       :return: 文本位置
           """
    regions = []
    # 1. 查找轮廓
    # plt.imshow(img)
    # plt.show()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # 取面积最大的轮廓
    if is_address == 0:
        # if is_name == 0:
        #     max_contour = contours_sorted[0]
        #     rect = cv2.boundingRect(max_contour)
        #     regions.append(rect)
        # else:
        pre_regions = []
        for i in range(len(contours_sorted)):
            cnt = contours_sorted[i]
            # 计算该轮廓的面积
            area = cv2.contourArea(cnt)
            # 面积小的都筛选掉
            if (area < 100 and is_date == 0) or (area < 60 and is_date == 1):
                continue

            # 找到最小的矩形，该矩形可能有方向
            rect = cv2.boundingRect(cnt)
            pre_regions.append(rect)

        if len(pre_regions) > 1:
            name_rect = np.array(pre_regions[0])
            for i in range(1, len(pre_regions)):
                if pre_regions[i][3] > name_rect[3] * 4 / 5:
                    name_rect[1] = min(name_rect[1], pre_regions[i][1])
                    name_rect[2] = max(name_rect[0] + name_rect[2], pre_regions[i][0] + pre_regions[i][2]) - min(
                        name_rect[0], pre_regions[i][0])
                    name_rect[0] = min(name_rect[0], pre_regions[i][0])
                    name_rect[3] = max(name_rect[3], pre_regions[i][3])
            regions.append(name_rect)
        else:
            regions = pre_regions

        regions = np.array(regions)
        regions[0][3] = regions[0][3] if regions[0][3] > 15 else 15
        regions[0][1] = regions[0][1] - 3
        regions[0][3] = regions[0][3] + 3
    else:
        for i in range(len(contours_sorted)):
            cnt = contours_sorted[i]
            # 计算该轮廓的面积
            area = cv2.contourArea(cnt)
            # 面积小的都筛选掉
            if (area < 150):
                continue

            # 找到最小的矩形，该矩形可能有方向
            rect = cv2.boundingRect(cnt)

            # # 计算高和宽
            width = rect[2]
            hight = rect[3]

            # 筛选那些太细的矩形，留下扁的
            if hight > width * 1.3 or hight < 10:
                continue

            regions.append(rect)
        regions = remove_inside(regions)
        regions = merge(regions)
        for rect in regions:
            if rect[0] > 30 and rect[2] < 20:
                regions.remove(rect)
                continue
            # 如果地址第一二行连接在一起，则将其分离
            if rect[3] > 40 and rect[3] < 60 and rect[1] < 20:
                regions.append([rect[0], rect[1], rect[2], int(rect[3] / 2 - 3)])
                regions.append([rect[0], rect[1] + int(rect[3] / 2 + 3), rect[2], int(rect[3] / 2 - 3)])
                regions.remove(rect)
            #  如果地址第二三行连接在一起，则将其分离
            if rect[3] > 40 and rect[1] > 20:
                regions.append([rect[0], rect[1], rect[2], int(rect[3] / 2 - 3)])
                regions.append([rect[0], rect[1] + int(rect[3] / 2 + 3), rect[2], int(rect[3] / 2 - 3)])
                regions.remove(rect)

    return remove_inside(regions)


