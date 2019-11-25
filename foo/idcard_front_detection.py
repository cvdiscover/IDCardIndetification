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


# from foo.tools.front_tools import *  #  稍后进行测试


def check_location(img, region):
    """
    检测位置是否正确
    :param img: 原图
    :param region: 各信息位置
    :return:
    """
    # print(regions, regions[0][0][2])

    if region[0][0][3] < 15 or region[0][0][3] > 30 or region[0][0][2] < 25:
        return 1
    elif cal_cross_ratio(region[1][0][1], region[1][0][1] + region[1][0][3], region[2][0][1],
                         region[2][0][1] + region[2][0][3]) < 0.5:
        return 1
    elif cal_cross_ratio(region[3][0][1], region[3][0][1] + region[3][0][3], region[4][0][1],
                         region[4][0][1] + region[4][0][3]) < 0.5 or \
            cal_cross_ratio(region[4][0][1], region[4][0][1] + region[4][0][3], region[5][0][1],
                            region[5][0][1] + region[5][0][3]) < 0.5:
        return 1
    elif region[7][0][2] < 200:
        return 1
    elif check_idnumber(img):
        return 1
    elif img.shape[0] / img.shape[1] < 290 / 500 or img.shape[0] / img.shape[1] > 350 / 500:
        return 1
    elif region[7][0][1] + 1/2* region[7][0][3] <  region[8][0][1] + region[8][0][3]:
        return 1
    else:
        return 0


# def box_get_front_correction():



def box_get_front_correction(img, save_name, imgHeight, imgWidth, face_rect):
    """
        获取正面文字位置。先定位地址位置，再根据地址位置校正其他信息位置
        :param img: 图片
        :param imgHeight: 图片高度
        :param imgWidth: 图片宽度
        :param face_rect: 人脸位置
        :return: 文字位置
        """

    regions = []

    #精准确定照片位置
    photo_x1 = face_rect[0][0] - 40
    photo_x2 = face_rect[0][0] + face_rect[0][2] + 40
    photo_y1 = face_rect[0][1] - 40
    photo_y2 = face_rect[0][1] + face_rect[0][3] + 60
    photo_cut = img[photo_y1:photo_y2, photo_x1:photo_x2]
    img=cv2.rectangle(img,(photo_x1,photo_y1),(photo_x2,photo_y2),(0,0,255),2)
    if is_debug ==1:
        plt.imshow(img,cmap=plt.gray())
        #plt.imshow(photo_cut, cmap=plt.gray())
        plt.show()
    photo_region = get_photo_position(photo_cut)
    photo_region[0][0] += photo_x1
    photo_region[0][1] += photo_y1


    # 地址
    address_y = int(imgHeight / 2.11)
    address_x = 87
    address_addHeight = int(address_y + imgHeight / 3.17)
    # address_addWidth = int(address_x + 230)
    #address_addWidth = face_rect[0][0] + int(face_rect[0][2] / 2) - 78
    address_addWidth = photo_region[0][0]
    address = img[address_y:address_addHeight, address_x:address_addWidth]
    img = cv2.rectangle(img, (address_x, address_y), (address_addWidth, address_addHeight), (0, 0, 255), 2)
    #plt.imshow(img, cmap=plt.gray())
    #plt.show()
    try:
        # address_region = get_regions(address, 1, 1)
        # if len(address_region) == 0:
        address_region = get_regions(address, 1, 1, is_consider_color=0)
    except Exception as e:
        address_region = get_regions(address, 1, 1, is_consider_color = 0)
    for i in range(len(address_region)):
        address_region[i][0] += address_x
        address_region[i][1] += address_y
    left_position = address_region[:, 0].min() - 15

    # 名字
    name_y = int(imgHeight / 10.8) - 10
    # name_width = int(imgWidth / 5.7)
    name_x = left_position
    name_addHeight = int(name_y + 46) + 10
    name_addWidth = int(name_x + imgWidth / 4)
    name = img[name_y:name_addHeight, name_x:name_addWidth]
    img = cv2.rectangle(img, (name_x, name_y), (name_addWidth, name_addHeight), (0, 0, 255), 2)
    #plt.imshow(name, cmap=plt.gray())
    #plt.show()
    # scale = int(img.shape[1] / 500)
    # print(scale, img.shape)
    try:
        name_region = get_regions(name, 1, is_name=1)
        if len(name_region) == 0:
            name_region = get_regions(name, 1, is_name=1, is_consider_color=0)
    except Exception as e:
        name_region = get_regions(name, 1, is_name=1, is_consider_color=0)
    name_region[0][0] += name_x
    name_region[0][1] += name_y
    regions.append(name_region)

    # 性别
    sex_y = int(imgHeight / 4.5)
    sex_x = left_position
    sex_addHeight = int(sex_y + imgHeight / 7.71)
    sex_addWidth = int(sex_x + 45)
    sex = img[sex_y:sex_addHeight, sex_x:sex_addWidth]
    img = cv2.rectangle(img, (sex_x, sex_y), (sex_addWidth, sex_addHeight), (0, 0, 255), 2)
    #plt.imshow(sex, cmap=plt.gray())
    #plt.show()
    try:
        sex_region = get_regions(sex, 1)
        if len(sex_region) == 0:
            sex_region = get_regions(sex, 1, is_consider_color=0)
    except Exception as e:
        sex_region = get_regions(sex, 1, is_consider_color=0)

    sex_region[0][0] += sex_x
    sex_region[0][1] += sex_y
    regions.append(sex_region)
    # cv2.imwrite(str(PATH) + '\\' + "sex.png", sex)

    # 民族
    nationality_y = int(imgHeight / 4.5)
    nationality_x = left_position + 90
    nationality_addHeight = int(nationality_y + imgHeight / 7.71)
    nationality_addWidth = int(nationality_x + 50)
    nationality = img[nationality_y:nationality_addHeight, nationality_x:nationality_addWidth]
    img = cv2.rectangle(img, (nationality_x, nationality_y), (nationality_addWidth, nationality_addHeight), (0, 0, 255), 2)
    # plt.imshow(nationality, cmap=plt.gray())
    # plt.show()
    try:
        nationality_region = get_regions(nationality, 1)
        if len(nationality_region) == 0:
            nationality_region = get_regions(nationality, 1, is_consider_color=0)
    except Exception as e:
        nationality_region = get_regions(nationality, 1, is_consider_color=0)
    nationality_region[0][0] += nationality_x
    nationality_region[0][1] += nationality_y
    regions.append(nationality_region)

    # 生日 年

    birth_year_y = int(imgHeight / 2.84)
    birth_year_x = left_position
    birth_year_addHeight = int(birth_year_y + imgHeight / 7.71)
    birth_year_addWidth = int(birth_year_x + 68)
    birth_year = img[birth_year_y:birth_year_addHeight, birth_year_x:birth_year_addWidth]
    img = cv2.rectangle(img, (birth_year_x, birth_year_y), (birth_year_addWidth, birth_year_addHeight), (0, 0, 255), 2)
    # plt.imshow(birth_year, cmap=plt.gray())
    # plt.show()
    try:
        birth_year_region = get_regions(birth_year, 1, is_date=1)
        if len(birth_year_region) == 0:
            birth_year_region = get_regions(birth_year, 1, is_date=1, is_consider_color = 0)
    except Exception as e:
        birth_year_region = get_regions(birth_year, 1, is_date=1, is_consider_color=0)

    birth_year_region[0][0] += birth_year_x
    birth_year_region[0][1] += birth_year_y
    regions.append(birth_year_region)

    # 生日 月
    birth_month_y = int(imgHeight / 2.84)
    birth_month_x = left_position + 80
    birth_month_addHeight = int(birth_month_y + imgHeight / 7.71)
    birth_month_addWidth = int(birth_month_x + 37)
    birth_month = img[birth_month_y:birth_month_addHeight, birth_month_x:birth_month_addWidth]
    img = cv2.rectangle(img, (birth_month_x, birth_month_y), (birth_month_addWidth, birth_month_addHeight), (0, 0, 255), 2)

    try:
        birth_month_region = get_regions(birth_month, 1, is_date=1)
        if len(birth_month_region) == 0:
            birth_month_region = get_regions(birth_month, 1, is_date=1, is_consider_color = 0)
    except Exception as e:
        birth_month_region = get_regions(birth_month, 1, is_date=1, is_consider_color=0)
    birth_month_region[0][0] += birth_month_x
    birth_month_region[0][1] += birth_month_y
    regions.append(birth_month_region)

    # 生日 日

    birth_day_y = int(imgHeight / 2.84)
    birth_day_x = left_position + 125
    birth_day_addHeight = int(birth_day_y + imgHeight / 7.71)
    birth_day_addWidth = int(birth_day_x + 37)
    birth_day = img[birth_day_y:birth_day_addHeight, birth_day_x:birth_day_addWidth]
    img = cv2.rectangle(img, (birth_day_x, birth_day_y), (birth_day_addWidth, birth_day_addHeight), (0, 0, 255), 2)
    # plt.imshow(birth_day, cmap=plt.gray())
    # plt.show()
    try:
        birth_day_region = get_regions(birth_day, 1, is_date=1)
        if len(birth_day_region) == 0:
            birth_day_region = get_regions(birth_day, 1, is_date=1, is_consider_color = 0)
    except Exception as e:
        birth_day_region = get_regions(birth_day, 1, is_date=1, is_consider_color=0)
    birth_day_region[0][0] += birth_day_x
    birth_day_region[0][1] += birth_day_y
    regions.append(birth_day_region)

    # 地址
    regions.append(address_region)

    # 身份证号码

    id_y = int(imgHeight / 1.3)
    id_x = int(imgWidth / 3.06)
    id_addHeight = int(id_y + imgHeight / 6.70)
    id_addWidth = int(id_x + 322)
    id = img[id_y:id_addHeight, id_x:id_addWidth]
    img = cv2.rectangle(img, (id_x, id_y), (id_addWidth, id_addHeight), (0, 0, 255), 2)
    # plt.imshow(id, cmap=plt.gray())
    # plt.show()
    try:
        # id_region = get_regions(id, 1)
        # if len(id_region) == 0:
        id_region = get_regions(id, 1, is_consider_color = 0)
    except Exception as e:
        id_region = get_regions(id, 1, is_consider_color=0)
    id_region[0][0] += id_x
    id_region[0][1] += id_y
    regions.append(id_region)
    # plt.imshow(birth_year, cmap=plt.gray())
    # plt.show()
    # 照片
    # photo_y = int(imgHeight / 6)
    # photo_x = int(imgWidth / 1.6)
    # photo_addHeight = int(photo_y + imgHeight / 1.7)
    #
    # photo_addWidth = int(photo_x + imgWidth / 3.2)
    # photo = img[photo_y:photo_addHeight, photo_x:photo_addWidth]
    # print(photo_y,photo_addHeight, photo_x,photo_addWidth)

    # regions.append(
    #     np.array([[photo_x, photo_y, photo_addWidth - photo_x, photo_addHeight - photo_y]]))
    # regions.append(
    #     np.array([[face_rect[0][0]-15, face_rect[0][1]-40, face_rect[0][2]+40, face_rect[0][3]+60]]))
    regions.append(photo_region)

    regions = complete_box(regions)
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
            cv2.drawContours(img_copy, np.array([box]), 0, (0, 255, 0), 1)
    if is_debug == 1:
        plt.imshow(img_copy, cmap=plt.gray())
        plt.show()
    cv2.imencode('.jpg', img_copy)[1].tofile(str(save_name))
    return regions


def get_photo_position(img):
    """

    :param img:
    :return:
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (7, 7), 1)
    # edges = cv2.Canny(img_gray, 10, 100, apertureSize=3)  # Canny算子边缘检测
    edges = cv2.Canny(img_gray_blur, 70, 20)  # Canny算子边缘检测
    _, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_contour = img_binary | edges

    erode_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_dilate = cv2.dilate(img_contour, dilate_elment, iterations=1)
    #img_erode = cv2.erode(img, erode_elment, iterations=1)


    # res,contours,hi = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    _, contours, _ = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    cnt_1 = contours_sorted[0]
    rect_1 = cv2.boundingRect(cnt_1)
    h, w = img.shape[:2]
    rect_1 = np.array(rect_1)
    if rect_1[2] < 120 or rect_1[3] < 130 or  rect_1[2] > 180 or rect_1[3] > 200:
        rect_1[0] = int(w / 2) - 68 if int(w / 2) - 68 > 0 else 0
        rect_1[1] = int(h / 2) - 98 if int(h / 2) - 98 > 0 else 0
        rect_1[2] = 150
        rect_1[3] = 180

    return np.array([rect_1])


def complete_box(regions):
    """
    补全缺损的矩形。以地址位置为基准，判断姓名、性别、出生年的左侧位置是否合格，如果不合格则根据地址位置纠正。
        判断其宽度是否合格，如果不合格则根据先验信息将其扩充。
    :param regions: 文字位置：姓名、性别、民族、出生年月日、住址、身份证号、头像
    :return: 补全后的位置
    """
    #address_x = min(regions[6][:, 0].min(),np.array(regions[0:5])[:, :, 0].min())
    address_x = regions[6][:, 0].min()
    # print("regions:",np.array(regions[0:5])[:, 0].max() )
    #address_x = regions[6][:, 0].max()
    # 扩充姓名
    if abs(regions[0][0][0] - address_x) > 5:
        regions[0][0][2] += regions[0][0][0] - address_x
        regions[0][0][0] = address_x

    # 扩充性别
    if abs(regions[1][0][0] - address_x) > 5:
        regions[1][0][2] += regions[1][0][0] - address_x
        regions[1][0][0] = address_x
        regions[1][0][2] = 30 if regions[1][0][2] < 30 else regions[1][0][2]
    # 对齐性别
    if cal_cross_ratio(regions[1][0][1], regions[1][0][1] + regions[1][0][3], regions[2][0][1],
                         regions[2][0][1] + regions[2][0][3]) < 0.5:
        regions[1][0][1] = regions[2][0][1]

    # 剪短民族
    if regions[2][0][2] > 30:
        regions[2][0][0] = regions[2][0][0] + regions[2][0][2] - 30
        regions[2][0][2] = 30
    # 扩充民族
    if regions[2][0][2] < 25:
        regions[2][0][2] = 30


    # 扩充年份
    if abs(regions[3][0][0] - address_x) > 5:
        regions[3][0][2] += regions[1][0][0] - address_x
        regions[3][0][0] = address_x
    regions[3][0][2] = 45 if regions[3][0][2] < 45 else regions[3][0][2]

    # 对齐地址

    if len(regions[6]) > 1 and regions[6][:, 0].max() - regions[6][:, 0].min() > 10:
        address_x_max = regions[6][:, 0].max()
        for i in range(len(regions[6])):
            regions[6][i, 2] -= (address_x_max - regions[6][i, 0])
            regions[6][i, 0] = address_x_max
    if regions[6][:,0].max() > address_x:
        for i in range(len(regions[6])):
            regions[6][i, 2] -= (address_x - regions[6][i, 0])
            regions[6][i, 0] = address_x
    return regions

#获取框内文字信息
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

# 寻找文字区域
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
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
