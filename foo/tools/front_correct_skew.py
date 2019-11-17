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


# from predict_location import predict_location


def find_cross_point(line1, line2):
    """
    计算两直线交点
    :param line1: 直线一两端点坐标
    :param line2: 直线二两端点坐标
    :return: 交点坐标
    """
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if x2 == x1:
        return False
    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        if k1 == k2:
            return False
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return int(x), int(y)


def cal_area(x1, y1, x2, y2, x3, y3):
    """
    计算三点围成图形的面积
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param x3:
    :param y3:
    :return:
    """
    return abs((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y1 - y3 * x1))


def cal_distance(x1, y1, x2, y2, x3, y3):
    """
    计算点（x3, y3）到直线((x1, x2), (x2, y2))的距离
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param x3:
    :param y3:
    :return:
    """
    area = 1 / 2 * abs((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y1 - y3 * x1))
    length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    distance = 2 * area / length
    # print(area,length,distance)
    return distance


def test_get_id(img):
    get_id_by_binary(img.copy())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    # plt.imshow(binary, cmap=plt.gray())
    # plt.show()
    # plt.imshow(img_binary, cmap=plt.gray())
    # plt.show()


def get_id_by_binary(img, face_rect):
    img = cv2.pyrMeanShiftFiltering(img, 10, 50)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)

    #  膨胀图像
    #print(face_rect)
    erode_elment_x = int(face_rect[2] / 90 * 20)
    erode_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_elment_x, 3))
    img_dilate = cv2.dilate(img_binary, erode_elment, iterations=1)
    if is_show_id_binary:
        plt.imshow(img_dilate, cmap=plt.gray())
        plt.show()
    id_rect = [0, 0, 0, 0]
    # 1. 查找轮廓

    binary,contours,hierarchy= cv2.findContours(img_dilate, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # plt.imshow(img_dilate, cmap=plt.gray())
    # plt.show()
    # plt.imshow(img, cmap=plt.gray())
    # plt.show()
    for i in range(len(contours)):
        cnt = contours[i]
        rect = cv2.boundingRect(cnt)
        # # 计算高和宽
        width = rect[2]
        hight = rect[3]

        # 根据身份证号特征，筛选轮廓
        if hight * 8  < width < hight * 20 and hight > 10 and rect[2] > id_rect[2] and face_rect[1] + 1.2 * face_rect[3] <rect[1] < face_rect[1] + 4 * face_rect[3]:
            id_rect = rect

    if is_show_id_rect:
        x1, x2 = id_rect[0], id_rect[0] + id_rect[2]
        y1, y2 = id_rect[1], id_rect[1] + id_rect[3]

        box = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
        cv2.drawContours(img, np.array([box]), 0, (0, 255, 0), 2)
        plt.imshow(img, cmap=plt.gray())
        plt.show()


    id_rect_minArea = [[id_rect[0] + id_rect[2] // 2, id_rect[1] + id_rect[3] // 2], [id_rect[2], id_rect[3]], 0]
    return id_rect_minArea


def get_id_by_corner(img, max_face):
    """
    获取身份证号位置
    :param img: 图片
    :param max_face: 人脸位置
    :return: 身份证号位置
    """

    img_mark = img.copy()
    img_mark[:, :] = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # 输入图像必须是 float32 ,最后一个参数在 0.04 到 0.05 之间
    # dst = cv2.cornerHarris(gray,2,3,0.04)
    # dst = cv2.goodFeaturesToTrack(gray, 200, 0.04, 7)
    # result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst,None)
    #
    # # Threshold for an optimal value, it may vary depending on the image.
    # img[dst>0.01*dst.max()]=[0,0,255]
    #
    # cv2.imshow('dst',img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    corners = cv2.goodFeaturesToTrack(gray, 500, 0.06, 1)

    corners = np.int0(corners)
    # print(corners)
    corners_list = []
    for i in corners:
        x, y = i.ravel()
        # if y < 329 + 1.5 * 74 or y > 329 + 2.5 * 74:
        #     continue
        corners_list.append([x, y])
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.circle(img_mark, (x, y), 3, (0, 0, 255), -1)

    # 显示原图和处理后的图像
    img_mark_gray = img_mark[:, :, 2]  # cv2.cvtColor(img_mark, cv2.COLOR_BGR2GRAY)
    img_mark_gray = cv2.medianBlur(img_mark_gray, 7)

    # _, img_mark_binary = cv2.threshold(img_mark_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # print(max_face[2])
    if max_face[2] > 70:
        elment = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
    else:
        elment = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    img_mark_gray_dilate = cv2.dilate(img_mark_gray, elment, iterations=1)
    # plt.imshow(img_mark_gray_dilate, cmap=plt.gray())
    # plt.show()
    contours, _ = cv2.findContours(img_mark_gray_dilate, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    regions = []
    regions_1 = []
    regions_2 = []
    for i in range(len(contours_sorted)):
        cnt = contours_sorted[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        if (area < 800):
            continue
        rect = cv2.minAreaRect(cnt)
        if (rect[0][1] < (max_face[1] + 1.5 * max_face[3])):
            continue
        if (rect[2] < -45 and rect[1][1] / rect[1][0] < 5) or (-45 < rect[2] < 0 and rect[1][0] / rect[1][1] < 5):
            print(rect[2], rect[1][0], rect[1][1])
            continue
        if -45 < rect[2] <= 0:
            regions_1.append(rect)
        else:
            regions_2.append(rect)
        regions.append(rect)
    # regions = sorted(regions, key = lambda l: l[2],reverse=True)
    # 根据长度排序
    regions_1 = sorted(regions_1, key=lambda l: l[1][0], reverse=True)
    regions_2 = sorted(regions_2, key=lambda l: l[1][1], reverse=True)
    if len(regions_1) == 0:
        max_rect = regions_2[0]
    elif len(regions_2) == 0:
        max_rect = regions_1[0]
    else:
        if regions_1[0][1][0] > regions_2[0][1][1]:
            max_rect = regions_1[0]
        else:
            max_rect = regions_2[0]

    box = cv2.boxPoints(max_rect)
    box = np.int0(box)
    cv2.drawContours(img_mark, np.array([box]), 0, (0, 255, 0), 2)

    # cv2.imshow('dst', img_mark)
    # # cv2.imwrite("F:/idcard/3(1)/3/business licence corner/" + name,img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    test_get_id(img.copy())

    return max_rect


def get_border_by_sobel(img):
    """
    通过canny算子计算图像的梯度，并进行hough直线检测
    :param img: 图片
    :return: 检测直线位置
    """
    img3 = cv2.GaussianBlur(img, (5, 5), 0)
    img_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    # print(len(contours))
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按面积排序

    fill = cv2.rectangle(img_binary.copy(), (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)  # 将图片涂黑
    fill = cv2.drawContours(fill.copy(), contours, 0, (255, 255, 255), -1)  # 将最大轮廓涂白
    x = cv2.Sobel(fill, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(fill, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 80, minLineLength=100,
                            maxLineGap=20)
    # plt.imshow(edges, cmap=plt.gray())
    # plt.show()
    return lines


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):  #
    """
    使用插值方法对图片 resize
    :param image:图片
    :param width:宽度
    :param height:高度
    :param inter:插值方法
    :return:调整大小后的图片
    """
    dim = None
    (h, w) = image.shape[:2]  #
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # print(dim)
    resized = cv2.resize(image, dim, interpolation=inter)  # interpo;ation为插值方法，这里选用的是
    return resized


def predict_location(land_mark, face, id_rect):
    """
    根据人眼睛鼻子眉心位置，估算身份证边界位置
    :param land_mark:左眼右眼眉心鼻子位置
    :param face:人脸位置左上角点坐标及高宽
    :return:
        估算的4边位置
    """
    id_width = max(id_rect[1][0], id_rect[1][1])
    # id_width = id_rect[2]
    # print(id_width,face[2])
    if id_width < 2.2 * face[2] or id_width > 5 * face[2]:
        id_width = int(3.1 * face[2])

    # print(id_width, face,id_rect)
    horizontal_x_distance = land_mark[1][0] - land_mark[0][0]
    horizontal_y_distance = land_mark[1][1] - land_mark[0][1]

    eye_id_distance_y = int(id_rect[0][1]) - int((land_mark[1][1] + land_mark[0][1]) / 2 - 2 * horizontal_y_distance)

    if eye_id_distance_y < 1.1 * face[2]:
        eye_id_distance_y = int(1.72 * face[2])

    #  上线
    top_line_pre = [land_mark[0][0] - 5 * horizontal_x_distance, land_mark[0][1] - 5 * horizontal_y_distance,
                    land_mark[1][0] + horizontal_x_distance, land_mark[1][1] + horizontal_y_distance]
    # top_line = [top_line_pre[0], top_line_pre[1] - int(0.40 * id_width), top_line_pre[2],
    #             top_line_pre[3] - int(0.40 * id_width)]
    top_line = [top_line_pre[0], top_line_pre[1] + eye_id_distance_y, top_line_pre[2],
                top_line_pre[3] + eye_id_distance_y]
    top_line = [top_line[0], top_line[1] - int(0.97 * id_width), top_line[2],
                top_line[3] - int(0.97 * id_width)]
    for i in range(4):
        if top_line[i] < 0:
            top_line[i] = 0
    """cv2.circle(img, (top_line[0], top_line[1]), 8, (0, 0, 255), 2)
    cv2.circle(img, (top_line[2], top_line[3]), 8, (0, 0, 255), 2)
    cv2.line(img, (top_line[0], top_line[1]), (top_line[2], top_line[3]), (0, 0, 255), 2)
    """

    # 下线


    # print(eye_id_distance_y,int(id_rect[0][1]))
    bottom_line_pre = [land_mark[0][0] - 5 * horizontal_x_distance, land_mark[0][1] - 5 * horizontal_y_distance,
                       land_mark[1][0] + horizontal_x_distance, land_mark[1][1] + horizontal_y_distance]
    #print(bottom_line_pre, eye_id_distance_y,horizontal_y_distance)
    bottom_line = [bottom_line_pre[0], bottom_line_pre[1] + eye_id_distance_y, bottom_line_pre[2],
                   bottom_line_pre[3] + eye_id_distance_y]
    bottom_line = [bottom_line[0], bottom_line[1] + int(0.12 * id_width), bottom_line[2],
                   bottom_line[3] + int(0.12 * id_width)]
    # print(bottom_line, bottom_line_pre[1],eye_id_distance_y,id_width)
    # cv2.circle(img, (bottom_line[0], bottom_line[1]), 8, (0, 255, 0), 2)
    # cv2.circle(img, (bottom_line[2], bottom_line[3]), 8, (0, 255, 0), 2)
    # cv2.line(img, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (0, 0, 255), 2)
    vertical_x_distance = land_mark[3][0] - land_mark[2][0]
    vertical_y_distance = land_mark[3][1] - land_mark[2][1]

    # 左线
    eye_id_distance_x = int((land_mark[2][0] + land_mark[3][0]) / 2 ) - int(id_rect[0][0])
    if eye_id_distance_x > 1.5 * face[2]:
        eye_id_distance_x = int(0.82 * face[2])
    left_line_pre = [land_mark[2][0] - 2 * vertical_x_distance, land_mark[2][1] - 2 * vertical_y_distance,
                     land_mark[3][0] + 2 * vertical_x_distance, land_mark[3][1] + 2 * vertical_y_distance]
    # left_line = [left_line_pre[0] - int(1.42 * id_width), left_line_pre[1], left_line_pre[2] - int(1.42 * id_width),
    #              left_line_pre[3]]
    left_line = [left_line_pre[0] - eye_id_distance_x, left_line_pre[1], left_line_pre[2] - eye_id_distance_x,
                 left_line_pre[3]]
    left_line = [left_line[0] - int(1.07 * id_width), left_line[1], left_line[2] - int(1.07 * id_width),
                 left_line[3]]
    # cv2.circle(img, (left_line[0], left_line[1]), 8, (255, 0, 0), 2)
    # cv2.circle(img, (left_line[2], left_line[3]), 8, (255, 0, 0), 2)
    # cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 2)

    for i in range(4):
        if left_line[i] < 0:
            left_line[i] = 0
    # 右线
    right_line_pre = [land_mark[2][0] - 2 * vertical_x_distance, land_mark[2][1] - 2 * vertical_y_distance,
                      land_mark[3][0] + 2 * vertical_x_distance, land_mark[3][1] + 2 * vertical_y_distance]
    # right_line = [right_line_pre[0] + int(0.36 * id_width), right_line_pre[1], right_line_pre[2] + int(0.36 * id_width),
    #               right_line_pre[3]]
    right_line = [right_line_pre[0] - eye_id_distance_x, right_line_pre[1], right_line_pre[2] - eye_id_distance_x,
                  right_line_pre[3]]
    right_line = [right_line[0] + int(0.68 * id_width), right_line[1], right_line[2] + int(0.68 * id_width),
                  right_line[3]]


    # cv2.circle(img, (right_line[0], right_line[1]), 8, (255, 0, 0), 2)
    # cv2.circle(img, (right_line[2], right_line[3]), 8, (255, 0, 0), 2)
    # cv2.line(img, (right_line[0], right_line[1]), (right_line[0], right_line[1]), (0, 0, 255), 2)
    # plt.imshow(img)
    # plt.show()

    top_line = [top_line[0], (top_line[1] + top_line[3]) // 2, top_line[2], (top_line[1] + top_line[3]) // 2]
    bottom_line = [bottom_line[0], (bottom_line[1] + bottom_line[3]) // 2, bottom_line[2],
                   (bottom_line[1] + bottom_line[3]) // 2]
    left_line = [(left_line[0] + left_line[2]) // 2, top_line[1], (left_line[0] + left_line[2]) // 2, left_line[3]]
    right_line = [(right_line[0] + right_line[2]) // 2, right_line[1], (right_line[0] + right_line[2]) // 2,
                  right_line[3]]
    predict_border_lines = [top_line, bottom_line, left_line, right_line]
    # print(predict_border_lines)
    return predict_border_lines


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst


def get_border_by_canny(img, is_filter=0):
    """
    使用canny算子计算图像的梯度，并进行hough直线检测
    :param img:
    :return:
    """

    # custom_blur_demo(src)
    # img = cv2.GaussianBlur(img, (3, 3), 1)
    # img = cv2.bilateralFilter(img, 0, 100, 30)

    if is_filter:
        img = cv2.pyrMeanShiftFiltering(img, 10, 50)
    # img = cv2.pyrMeanShiftFiltering(img, 50, 50)

    # from PIL import Image
    # from PIL import ImageEnhance
    # image =  Image.fromarray(np.array(img))
    # enh_sha = ImageEnhance.Sharpness(image)
    # sharpness = 3.0
    # image_sharped = enh_sha.enhance(sharpness)
    # img = np.array(image_sharped)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    edges = cv2.Canny(img_gray, 70, 30)  # Canny算子边缘检测
    elment = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, elment, iterations=1)


    # from back_correct import mark_corner_image
    # img_corner = mark_corner_image(img, 5)
    #
    # edges_ = edges / 255
    # img_corner_ = (255 - img_corner) / 255
    #
    # edges = (edges_.astype(np.uint8) & img_corner_.astype(np.uint8)) * 255

    # lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 80, minLineLength=80,
    #                         maxLineGap=15)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 80, minLineLength=40,
                            maxLineGap=5)

    lines_2d_original = lines[:, 0, :]

    lines_2d = []
    for line in lines_2d_original:
        x1, y1, x2, y2 = line
        lines_2d.append([x1, y1, x2, y2])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if is_show_lines:
        plt.imshow(img)
        plt.show()
    return lines


def get_border_gradient(img):
    """
    根据梯度提取边缘
    :param img: 图片
    :return:
    """
    img2 = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=5)
    y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=5)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst = cv2.medianBlur(dst, 3)
    _, dst_binary = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)
    if is_debug == 1:
        plt.imshow(dst_binary, cmap=plt.gray())
        plt.show()
    contours, hierarchy = cv2.findContours(dst_binary.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按面积排序

    fill = cv2.rectangle(dst_binary.copy(), (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)  # 将图片涂黑
    fill = cv2.drawContours(fill.copy(), contours, 0, (255, 255, 255), -1)  # 将最大轮廓涂白

    x = cv2.Sobel(fill, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(fill, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    if is_debug == 1:
        plt.imshow(edges, cmap=plt.gray())
        plt.show()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 80, minLineLength=100,
                            maxLineGap=10)

    lines_2d_original = lines[:, 0, :]

    lines_2d = []
    for line in lines_2d_original:
        x1, y1, x2, y2 = line
        lines_2d.append([x1, y1, x2, y2])
        cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # plt.imshow(img2)
    # plt.show()

    return lines


def get_border_by_binary_max_contour(img):
    """
    根据梯度提取边缘
    :param img: 图片
    :return:
    """
    img2 = img.copy()
    h, w = img.shape[:2]
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # plt.imshow(img_binary, cmap=plt.gray())
    # plt.show()
    contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按面积排序

    fill = cv2.rectangle(img_binary.copy(), (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)  # 将图片涂黑
    fill = cv2.drawContours(fill.copy(), contours, 0, (255, 255, 255), -1)  # 将最大轮廓涂白

    # plt.imshow(fill, cmap=plt.gray())
    # plt.show()
    x = cv2.Sobel(fill, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(fill, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    # 对边缘进行膨胀
    elment = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, elment, iterations=1)

    # plt.imshow(edges, cmap=plt.gray())
    # plt.show()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 80, minLineLength=50,
                            maxLineGap=10)

    if lines is None:
        lines = np.array([[[0, 0, w, 0]], [[0, h, w, h]], [[0, 0, 0, h]], [[w, 0, w, h]]])
        return lines
    lines_2d_original = lines[:, 0, :]

    lines_2d = []
    for line in lines_2d_original:
        x1, y1, x2, y2 = line
        lines_2d.append([x1, y1, x2, y2])
        cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if is_show_lines:
        plt.imshow(img2)
        plt.show()

    return lines


def get_border_by_grabcut(img, predict_border_lines):
    # print(predict_border_lines)
    h, w = img.shape[:2]
    if predict_border_lines[0][1] - 20 > 0:
        predict_border_lines[0][1] -= 20
        predict_border_lines[0][3] -= 20
    else:
        predict_border_lines[0][1] = 0
        predict_border_lines[0][3] = 0

    if predict_border_lines[1][1] + 20 < h:
        predict_border_lines[1][1] += 20
        predict_border_lines[1][3] += 20
    else:
        predict_border_lines[1][1] = h
        predict_border_lines[1][3] = h

    if predict_border_lines[2][0] - 20 > 0:
        predict_border_lines[2][0] -= 20
        predict_border_lines[2][2] -= 20
    else:
        predict_border_lines[2][1] = 0
        predict_border_lines[2][2] = 0

    if predict_border_lines[3][0] + 20 < w:
        predict_border_lines[3][0] += 20
        predict_border_lines[3][2] += 20
    else:
        predict_border_lines[3][0] = w
        predict_border_lines[3][2] = w

    rect = [predict_border_lines[2][0], predict_border_lines[0][1],
            predict_border_lines[3][0] - predict_border_lines[2][0],
            predict_border_lines[1][1] - predict_border_lines[0][1]]
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect_copy = tuple(rect)
    # print(rect_copy)
    cv2.grabCut(img, mask, rect_copy, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_show = img * mask2[:, :, np.newaxis]

    img_binary = mask2 * 255
    contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按面积排序

    fill = cv2.rectangle(img_binary.copy(), (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)  # 将图片涂黑
    fill = cv2.drawContours(fill.copy(), contours, 0, (255, 255, 255), -1)  # 将最大轮廓涂白

    plt.imshow(mask2, cmap=plt.gray())
    plt.show()
    x = cv2.Sobel(fill, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(fill, cv2.CV_16S, 0, 1, ksize=3)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    elment = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, elment, iterations=1)

    # plt.imshow(edges, cmap=plt.gray())
    # plt.show()

    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 80, minLineLength=50,
                            maxLineGap=10)

    if lines is None:
        lines = np.array([[[0, 0, w, 0]], [[0, h, w, h]], [[0, 0, 0, h]], [[w, 0, w, h]]])
        return lines
    lines_2d_original = lines[:, 0, :]

    lines_2d = []
    for line in lines_2d_original:
        x1, y1, x2, y2 = line
        lines_2d.append([x1, y1, x2, y2])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if  is_show_lines:
        plt.title("grabcut lines")
        plt.imshow(img)
        plt.show()
    return lines

# 比较图片中心和边缘亮度 未用
def compare_light(img):
    """
    比较图片中心和边缘亮度
    :param img:
    :return:
    """
    img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    h, w, _ = img.shape
    # print(h, w)
    top_mean = img_hsv[0:3, :, 2].mean()
    bottom_mean = img_hsv[-3:h + 1, :, 2].mean()
    left_mean = img_hsv[:, 0:3, 2].mean()
    right_mean = img_hsv[:, -3:w + 1, 2].mean()
    border_mean = (top_mean + bottom_mean + left_mean + right_mean) / 4
    center_mean = img_hsv[int(h / 2) - 20:int(h / 2) + 20, int(w / 2) - 20:int(w / 2) + 20, 2].mean()
    # print(border_mean, center_mean)
    if center_mean > 1.3 * border_mean:
        return True
    else:
        return False


def merge_lines(lines_2d, distance=10):
    """
    直线合并，如果两条直线可能为一条直线，则合并
    :param lines_2d: 输入直线数组
    :return: 合并后的直线
    """
    i = 0
    while (i < len(lines_2d)):
        line = lines_2d[i]
        merge_line = []
        j = 0
        for h_line in lines_2d:
            # print(h_line,line,h_line==line,(h_line == line))
            if h_line == line:
                j += 1
                continue
            # 两直线合并条件：一直线两端点到另一条直线的距离都小于distance
            if cal_distance(line[0], line[1], line[2], line[3], h_line[0], h_line[1]) < distance and \
                    cal_distance(line[0], line[1], line[2], line[3], h_line[2], h_line[3]) < distance:
                merge_line.append([line[0], line[1]])
                merge_line.append([line[2], line[3]])
                merge_line.append([h_line[0], h_line[1]])
                merge_line.append([h_line[2], h_line[3]])
                merge_line_arr = np.array(merge_line)
                # 两条直线都是横线，合并后的直线两端点分别取最左侧和最右侧端点
                if (abs(line[0] - line[2]) > abs(line[1] - line[3])) and (
                        abs(h_line[0] - h_line[2]) > abs(h_line[1] - h_line[3])):
                    point1 = merge_line_arr[np.where(merge_line_arr[:, 0] == merge_line_arr[:, 0].min())]
                    point2 = merge_line_arr[np.where(merge_line_arr[:, 0] == merge_line_arr[:, 0].max())]
                # 两条直线都是竖线，合并后的直线两端点分别取最上端和最下端端点
                elif (abs(line[0] - line[2]) < abs(line[1] - line[3])) and (
                        abs(h_line[0] - h_line[2]) < abs(h_line[1] - h_line[3])):
                    point1 = merge_line_arr[np.where(merge_line_arr[:, 1] == merge_line_arr[:, 1].min())]
                    point2 = merge_line_arr[np.where(merge_line_arr[:, 1] == merge_line_arr[:, 1].max())]
                else:
                    j += 1
                    continue
                # print(point1, point2,merge_line,cal_area(line[0],line[1],line[2],line[3],h_line[0],h_line[1]))
                lines_2d.append([point1[0][0], point1[0][1], point2[0][0], point2[0][1]])
                lines_2d.remove(line)
                lines_2d.remove(h_line)
                i -= 1
                if j < i:
                    i -= 1
                break
            j += 1
        i += 1
    return lines_2d
    # print(lines_2d)


def get_best_line_by_prediction(top_lines, bottom_lines, left_lines, right_lines, predict_border_lines, img):
    """
    根据预测边界线位置获取最优边界线
    :param top_lines: 上线数组
    :param bottom_lines: 下线数组
    :param left_lines: 左线数组
    :param right_lines: 右线数组
    :param predict_border_lines: 预测边界位置
    :return:
    """
    top_predict_line = predict_border_lines[0]
    bottom_predict_line = predict_border_lines[1]
    left_predict_line = predict_border_lines[2]
    right_predict_line = predict_border_lines[3]

    best_lines = []
    # 上线
    if len(top_lines) == 0:
        top_line = top_predict_line
    else:
        # 获取初始选择直线，直线应满足斜率近似
        for i in range(len(top_lines)):
            top_line = top_lines[i]
            k_top_predict = abs(
                (top_predict_line[3] - top_predict_line[1]) / (top_predict_line[2] - top_predict_line[0]))
            k_top_current = abs((top_line[3] - top_line[1]) / (top_line[2] - top_line[0]))
            k_d_c = abs(k_top_predict - k_top_current)
            start = i
            if k_d_c < 0.2:
                break
            if i == len(top_lines) - 1:
                top_line = top_predict_line

        k_d = abs(k_top_predict - k_top_current)

        # 考虑斜率找最近的直线
        y_predict = (top_predict_line[1] + top_predict_line[3]) / 2
        y_current = (top_line[1] + top_line[3]) / 2
        y_d = abs(y_predict - y_current)
        for i in range(start + 1, len(top_lines)):
            top_current = top_lines[i]
            k_top_current = abs((top_current[3] - top_current[1]) / (top_current[2] - top_current[0]))
            y_current = (top_current[1] + top_current[3]) / 2
            k_d_c = abs(k_top_predict - k_top_current)
            y_d_c = abs(y_predict - y_current)
            # print(y_d_c)
            if k_d_c < 0.2 and y_d_c < y_d:
                top_line = top_current
                # k_d = k_d_c
                y_d = y_d_c
        if y_d > 30:
            top_line = top_predict_line

        # 不考虑斜率找最近的直线
        if top_line == top_predict_line:
            top_line = top_lines[0]
            y_predict = (top_predict_line[1] + top_predict_line[3]) / 2
            y_current = (top_line[1] + top_line[3]) / 2
            y_d = abs(y_predict - y_current)

            for i in range(1, len(top_lines)):
                top_current = top_lines[i]
                y_current = (top_current[1] + top_current[3]) / 2
                y_d_r = abs(y_predict - y_current)
                if y_d_r < y_d:
                    top_line = top_current
                    y_d = y_d_r
            if y_d > 20:
                top_line = top_predict_line
    best_lines.append(top_line)

    # 下线
    if len(bottom_lines) == 0:
        bottom_line = bottom_predict_line
    else:
        for i in range(len(bottom_lines)):
            bottom_line = bottom_lines[i]
            k_bottom_predict = abs(
                (bottom_predict_line[3] - bottom_predict_line[1]) / (bottom_predict_line[2] - bottom_predict_line[0]))
            k_bottom_current = abs((bottom_line[3] - bottom_line[1]) / (bottom_line[2] - bottom_line[0]))
            k_d_c = abs(k_bottom_predict - k_bottom_current)
            start = i
            if k_d_c < 0.2:
                break
            if i == len(bottom_lines) - 1:
                bottom_line = bottom_predict_line

        k_d = abs(k_bottom_predict - k_bottom_current)

        y_predict = (bottom_predict_line[1] + bottom_predict_line[3]) / 2
        y_current = (bottom_line[1] + bottom_line[3]) / 2
        y_d = y_predict - y_current
        # print(y_predict)
        for i in range(start + 1, len(bottom_lines)):
            bottom_current = bottom_lines[i]
            k_bottom_current = abs((bottom_current[3] - bottom_current[1]) / (bottom_current[2] - bottom_current[0]))
            y_current = (bottom_current[1] + bottom_current[3]) / 2
            k_d_c = abs(k_bottom_predict - k_bottom_current)
            y_d_c = y_predict - y_current
            if k_d_c < 0.2 and abs(y_d_c) < abs(y_d) and y_d_c < 10:
                bottom_line = bottom_current
                # k_d = k_d_c
                y_d = y_d_c
        if abs(y_d) > 30 or y_d > 10:
            bottom_line = bottom_predict_line

        if bottom_line == bottom_predict_line:
            bottom_line = bottom_lines[0]
            y_predict = (bottom_predict_line[1] + bottom_predict_line[3]) / 2
            y_current = (bottom_line[1] + bottom_line[3]) / 2
            y_d = abs(y_predict - y_current)

            for i in range(1, len(bottom_lines)):
                bottom_current = bottom_lines[i]
                y_current = (bottom_current[1] + bottom_current[3]) / 2
                y_d_r = abs(y_predict - y_current)
                if y_d_r < y_d:
                    bottom_line = bottom_current
                    y_d = y_d_r
            if abs(y_d) > 20:
                bottom_line = bottom_predict_line
    best_lines.append(bottom_line)

    # 左线
    if len(left_lines) == 0:
        left_line = left_predict_line
    else:
        # 获取初始选择直线，直线应满足斜率近似
        for i in range(len(left_lines)):
            left_line = left_lines[i]
            # print(left_lines,left_predict_line)
            k_left_predict = abs((left_predict_line[2] - left_predict_line[0]) /
                                 (left_predict_line[3] - left_predict_line[1]))
            k_left_current = abs((left_line[2] - left_line[0]) / (left_line[3] - left_line[1]))
            k_d_c = abs(k_left_predict - k_left_current)
            start = i
            if k_d_c < 0.2:
                break
            if i == len(left_lines) - 1:
                left_line = left_predict_line

        k_d = abs(k_left_predict - k_left_current)

        x_predict = (left_predict_line[0] + left_predict_line[2]) / 2
        x_current = (left_line[0] + left_line[2]) / 2
        x_d = abs(x_predict - x_current)

        for i in range(start + 1, len(left_lines)):
            left_current = left_lines[i]
            k_left_current = abs((left_current[2] - left_current[0]) / (left_current[3] - left_current[1]))
            x_current = (left_current[0] + left_current[2]) / 2
            k_d_c = abs(k_left_predict - k_left_current)
            x_d_r = abs(x_predict - x_current)
            if k_d_c < 0.2 and x_d_r < x_d:
                left_line = left_current
                # k_d = k_d_c
                x_d = x_d_r

        if x_d > 30:
            left_line = left_predict_line

        # 如果未选择到斜率距离最优直线，则选择与预测直线最近的直线
        if left_line == left_predict_line:
            left_line = left_lines[0]
            x_predict = (left_predict_line[0] + left_predict_line[2]) / 2
            x_current = (left_line[0] + left_line[2]) / 2
            x_d = abs(x_predict - x_current)

            for i in range(1, len(left_lines)):
                left_current = left_lines[i]
                x_current = (left_current[0] + left_current[2]) / 2
                x_d_r = abs(x_predict - x_current)
                if x_d_r < x_d:
                    left_line = left_current
                    x_d = x_d_r

            if x_d > 20:
                left_line = left_predict_line
        # left_line = left_predict_line
    best_lines.append(left_line)

    # 右线
    if len(right_lines) == 0:
        right_line = right_predict_line
    else:
        for i in range(len(right_lines)):
            right_line = right_lines[i]
            k_right_predict = abs((right_predict_line[2] - right_predict_line[0]) /
                                  (right_predict_line[3] - right_predict_line[1]))
            k_right_current = abs((right_line[2] - right_line[0]) / (right_line[3] - right_line[1]))
            k_d_c = abs(k_right_predict - k_right_current)
            start = i
            if k_d_c < 0.2:
                break
            if i == len(right_lines) - 1:
                right_line = right_predict_line

        k_d = abs(k_right_predict - k_right_current)

        x_predict = (right_predict_line[0] + right_predict_line[2]) / 2
        x_current = (right_line[0] + right_line[2]) / 2
        x_d = abs(x_predict - x_current)

        for i in range(start + 1, len(right_lines)):
            right_current = right_lines[i]
            k_right_current = abs((right_current[2] - right_current[0]) / (right_current[3] - right_current[1]))
            x_current = (right_current[0] + right_current[2]) / 2
            k_d_c = abs(k_right_predict - k_right_current)
            x_d_r = abs(x_predict - x_current)
            if k_d_c < 0.2 and x_d_r < x_d:
                right_line = right_current
                # k_d = k_d_c
                x_d = x_d_r
        if x_d > 30:
            right_line = right_predict_line

        if right_line == right_predict_line:
            right_line = right_lines[0]
            x_predict = (right_predict_line[0] + right_predict_line[2]) / 2
            x_current = (right_line[0] + right_line[2]) / 2
            x_d = abs(x_predict - x_current)

            for i in range(1, len(right_lines)):
                right_current = right_lines[i]
                x_current = (right_current[0] + right_current[2]) / 2
                x_d_r = abs(x_predict - x_current)
                if x_d_r < x_d:
                    right_line = right_current
                    x_d = x_d_r

            if x_d > 20:
                right_line = right_predict_line
    best_lines.append(right_line)

    for line in best_lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if is_show_select_lines:
        plt.imshow(img)
        plt.show()

    return best_lines


def find_same_from_three_method(line1, line2, line3, orientation = 0):
    """
    在三条直线中根据直线类型寻找相似的线，并返回相似线中的一条
    :param line1: 直线1
    :param line2: 直线2
    :param line3: 直线3
    :param orientation:直线方向 0:水平 1：竖直
    :return:
    """
    # #直线line1到line2的距离
    # line12_d1 = cal_distance(line1[0], line1[1], line1[2], line1[3], line2[0], line2[1])
    # line12_d2 = cal_distance(line1[0], line1[1], line1[2], line1[3], line2[2], line2[3])
    # line12 = (line12_d1 + line12_d2) / 2
    #
    # # 直线line1到line3的距离
    # line13_d1 = cal_distance(line1[0], line1[1], line1[2], line1[3], line3[0], line3[1])
    # line13_d2 = cal_distance(line1[0], line1[1], line1[2], line1[3], line3[2], line3[3])
    # line13 = (line13_d1 + line13_d2) / 2
    #
    # # 直线line2到line3的距离
    # line23_d1 = cal_distance(line2[0], line2[1], line2[2], line2[3], line3[0], line3[1])
    # line23_d2 = cal_distance(line2[0], line2[1], line2[2], line2[3], line3[2], line3[3])
    # line23 = (line23_d1 + line23_d2) / 2

    # print(line12_d1, line12_d2, line13_d1, line13_d2, line23_d1, line23_d2, line12, line13, line23)
    # print(line12, line13, line23)
    # if line12 == min(line12, line13, line23):
    #     return line2
    # elif line13 == min(line12, line13, line23):
    #     return line3
    # else:
    #     return line2

    if orientation:
        line12_distance = abs((line1[0] + line1[2]) / 2 - (line2[0] + line2[2]) / 2)
        line13_distance = abs((line1[0] + line1[2]) / 2 - (line3[0] + line3[2]) / 2)
        line23_distance = abs((line2[0] + line2[2]) / 2 - (line3[0] + line3[2]) / 2)
    else:
        line12_distance = abs((line1[1] + line1[3]) / 2 - (line2[1] + line2[3]) / 2)
        line13_distance = abs((line1[1] + line1[3]) / 2 - (line3[1] + line3[3]) / 2)
        line23_distance = abs((line2[1] + line2[3]) / 2 - (line3[1] + line3[3]) / 2)
    # print((line1[1] + line1[3])//2,(line2[1] + line2[3])//2,(line3[1] + line3[3])//2)
    # print(line12_distance, line13_distance, line23_distance)
    if line12_distance == min(line12_distance, line13_distance, line23_distance):
        return line2
    elif line13_distance == min(line12_distance, line13_distance, line23_distance):
        return line3
    else:
        return line2


def get_best_from_two_method(line1, line2):
    """
    如果第二条直线近似第一条，则选择第二条直线
    :param line1: 估算出来的直线
    :param line2: 直线2
    :return:
    """

    #直线line1到line2的距离
    line_d1 = cal_distance(line1[0], line1[1], line1[2], line1[3], line2[0], line2[1])
    line_d2 = cal_distance(line1[0], line1[1], line1[2], line1[3], line2[2], line2[3])
    line = (line_d1 + line_d2) / 2

    if line < 25:
        return line2
    else:
        return line1


def select_best_border(id_rect, lines1, lines2, lines3):
    """
    从不同算法得到的边界线中选择最佳的边界线
    :param lines3: grabcut
    :param lines1: 通过边缘检测计算得到的边界线
    :param lines2: 通过最大轮廓得到的边界线
    :return: 选择的最佳边界线
    """
    # print(lines1)
    # print(lines2)
    # print(lines3)
    # print(id_rect)
    best_lines = []
    # 两直线中点y坐标差值
    # y1_d = (lines1[0][1] + lines1[0][3]) // 2 - (lines2[0][3] + lines2[0][3]) //2
    # if abs(y1_d) < 20:
    #     best_lines.append(lines2[0])
    # else:
    #     best_lines.append(lines1[0])
    #
    #
    # y2_d = (lines1[1][1] + lines1[1][3]) // 2 - (lines2[1][3] + lines2[1][3]) // 2
    # if abs(y2_d) < 20:
    #     best_lines.append(lines2[1])
    # else:
    #     best_lines.append(lines1[1])
    #
    # x1_d = (lines1[2][0] + lines1[2][2]) // 2 - (lines2[2][0] + lines2[2][2]) //2
    # if abs(x1_d) < 20:
    #     best_lines.append(lines2[2])
    # else:
    #     best_lines.append(lines1[2])
    #
    # x2_d = (lines1[3][0] + lines1[3][2]) // 2 - (lines2[3][0] + lines2[3][2]) //2
    # if abs(x2_d) < 20:
    #     best_lines.append(lines2[3])
    # else:
    #     best_lines.append(lines1[3])

    # return lines2
    #print(lines1, lines2, lines3)

    if len(lines1) == 4 and len(lines2) == 4 and len(lines3) == 4:
        for i in range(4):
            orientation = 0 if i < 2 else 1
            line = find_same_from_three_method(lines1[i], lines2[i], lines3[i], orientation)
            best_lines.append(line)
    else:
        if len(lines2) != 4 and len(lines3) == 4:
            for i in range(4):
                line = get_best_from_two_method(lines1[i], lines3[i])
                best_lines.append(line)
        elif len(lines3) != 4 and len(lines2) == 4:
                for i in range(4):
                    line = get_best_from_two_method(lines1[i], lines2[i])
                    best_lines.append(line)
        elif len(lines2) != 4 and len(lines3) != 4:
            return lines1


    # print(best_lines)
    return best_lines


def select_border_lines_by_max_contour(top_lines, bottom_lines, left_lines, right_lines, img):
    """
    # 选择从最大轮廓中获取的边界线
    :param top_lines: 上线数组
    :param bottom_lines: 下线数组
    :param left_lines: 左线数组
    :param right_lines: 右线数组
    :return: 选择的4条边界线
    """
    # print(top_lines, bottom_lines, left_lines, right_lines)
    best_lines = []
    best_lines.append(top_lines[0])
    best_lines.append(bottom_lines[0])
    best_lines.append(left_lines[0])
    best_lines.append(right_lines[0])

    for line in best_lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if is_show_select_lines:
        plt.imshow(img)
        plt.show()
    return best_lines


def select_border_lines_by_max_contour_1(top_lines, bottom_lines, left_lines, right_lines, predict_border_lines, img):
    """
    根据预测边界线位置获取最优边界线
    :param top_lines: 上线数组
    :param bottom_lines: 下线数组
    :param left_lines: 左线数组
    :param right_lines: 右线数组
    :param predict_border_lines: 预测边界位置
    :return:
    """
    top_predict_line = predict_border_lines[0]
    bottom_predict_line = predict_border_lines[1]
    left_predict_line = predict_border_lines[2]
    right_predict_line = predict_border_lines[3]

    best_lines = []
    # 上线
    if len(top_lines) == 0:
        top_line = top_predict_line
    else:
        # 获取初始选择直线，直线应满足斜率近似
        for i in range(len(top_lines)):
            top_line = top_lines[i]
            k_top_predict = abs(
                (top_predict_line[3] - top_predict_line[1]) / (top_predict_line[2] - top_predict_line[0]))
            k_top_current = abs((top_line[3] - top_line[1]) / (top_line[2] - top_line[0]))
            k_d_c = abs(k_top_predict - k_top_current)
            start = i
            if k_d_c < 0.1:
                break
            if i == len(top_lines) - 1:
                top_line = top_predict_line

        k_d = abs(k_top_predict - k_top_current)

        # 考虑斜率找最近的直线
        y_predict = (top_predict_line[1] + top_predict_line[3]) / 2
        y_current = (top_line[1] + top_line[3]) / 2
        y_d = abs(y_predict - y_current)
        for i in range(start + 1, len(top_lines)):
            top_current = top_lines[i]
            k_top_current = abs((top_current[3] - top_current[1]) / (top_current[2] - top_current[0]))
            y_current = (top_current[1] + top_current[3]) / 2
            k_d_c = abs(k_top_predict - k_top_current)
            y_d_c = abs(y_predict - y_current)
            # print(y_d_c)
            if k_d_c < 0.2 and y_d_c < y_d:
                top_line = top_current
                # k_d = k_d_c
                y_d = y_d_c
        if y_d > 30:
            top_line = top_predict_line

    best_lines.append(top_line)

    # 下线
    if len(bottom_lines) == 0:
        bottom_line = bottom_predict_line
    else:
        for i in range(len(bottom_lines)):
            bottom_line = bottom_lines[i]
            k_bottom_predict = abs(
                (bottom_predict_line[3] - bottom_predict_line[1]) / (bottom_predict_line[2] - bottom_predict_line[0]))
            k_bottom_current = abs((bottom_line[3] - bottom_line[1]) / (bottom_line[2] - bottom_line[0]))
            k_d_c = abs(k_bottom_predict - k_bottom_current)
            start = i
            if k_d_c < 0.1:
                break
            if i == len(bottom_lines) - 1:
                bottom_line = bottom_predict_line

        k_d = abs(k_bottom_predict - k_bottom_current)

        y_predict = (bottom_predict_line[1] + bottom_predict_line[3]) / 2
        y_current = (bottom_line[1] + bottom_line[3]) / 2
        y_d = y_predict - y_current
        # print(y_predict)
        for i in range(start + 1, len(bottom_lines)):
            bottom_current = bottom_lines[i]
            k_bottom_current = abs((bottom_current[3] - bottom_current[1]) / (bottom_current[2] - bottom_current[0]))
            y_current = (bottom_current[1] + bottom_current[3]) / 2
            k_d_c = abs(k_bottom_predict - k_bottom_current)
            y_d_c = y_predict - y_current
            if k_d_c < 0.2 and abs(y_d_c) < abs(y_d) and y_d_c < 10:
                bottom_line = bottom_current
                # k_d = k_d_c
                y_d = y_d_c
        if abs(y_d) > 30 or y_d > 10:
            bottom_line = bottom_predict_line

    best_lines.append(bottom_line)

    # 左线
    if len(left_lines) == 0:
        left_line = left_predict_line
    else:
        # 获取初始选择直线，直线应满足斜率近似
        for i in range(len(left_lines)):
            left_line = left_lines[i]
            # print(left_lines,left_predict_line)
            k_left_predict = abs((left_predict_line[2] - left_predict_line[0]) /
                                 (left_predict_line[3] - left_predict_line[1]))
            k_left_current = abs((left_line[2] - left_line[0]) / (left_line[3] - left_line[1]))
            k_d_c = abs(k_left_predict - k_left_current)
            start = i
            if k_d_c < 0.2:
                break
            if i == len(left_lines) - 1:
                left_line = left_predict_line

        k_d = abs(k_left_predict - k_left_current)

        x_predict = (left_predict_line[0] + left_predict_line[2]) / 2
        x_current = (left_line[0] + left_line[2]) / 2
        x_d = abs(x_predict - x_current)

        for i in range(start + 1, len(left_lines)):
            left_current = left_lines[i]
            k_left_current = abs((left_current[2] - left_current[0]) / (left_current[3] - left_current[1]))
            x_current = (left_current[0] + left_current[2]) / 2
            k_d_c = abs(k_left_predict - k_left_current)
            x_d_r = abs(x_predict - x_current)
            if k_d_c < 0.2 and x_d_r < x_d:
                left_line = left_current
                # k_d = k_d_c
                x_d = x_d_r

        if x_d > 30:
            left_line = left_predict_line


    best_lines.append(left_line)

    # 右线
    if len(right_lines) == 0:
        right_line = right_predict_line
    else:
        for i in range(len(right_lines)):
            right_line = right_lines[i]
            k_right_predict = abs((right_predict_line[2] - right_predict_line[0]) /
                                  (right_predict_line[3] - right_predict_line[1]))
            k_right_current = abs((right_line[2] - right_line[0]) / (right_line[3] - right_line[1]))
            k_d_c = abs(k_right_predict - k_right_current)
            start = i
            if k_d_c < 0.2:
                break
            if i == len(right_lines) - 1:
                right_line = right_predict_line

        k_d = abs(k_right_predict - k_right_current)

        x_predict = (right_predict_line[0] + right_predict_line[2]) / 2
        x_current = (right_line[0] + right_line[2]) / 2
        x_d = abs(x_predict - x_current)

        for i in range(start + 1, len(right_lines)):
            right_current = right_lines[i]
            k_right_current = abs((right_current[2] - right_current[0]) / (right_current[3] - right_current[1]))
            x_current = (right_current[0] + right_current[2]) / 2
            k_d_c = abs(k_right_predict - k_right_current)
            x_d_r = abs(x_predict - x_current)
            if k_d_c < 0.2 and x_d_r < x_d:
                right_line = right_current
                # k_d = k_d_c
                x_d = x_d_r
        if x_d > 30:
            right_line = right_predict_line


    best_lines.append(right_line)

    for line in best_lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if is_show_select_lines:
        plt.imshow(img)
        plt.show()
    # print(best_lines)
    return best_lines


def filter_and_classify_lines(img, lines, img2, face_rect):
    """
    过滤并将直线分类（上、下、左、右）
    :param lines: 直线
    :param img: 原图
    :param img2: 调试标记图片
    :return: 分类后的直线
    """
    horizontal, vertical = [], []  # 创建水平和垂直线list
    lines_2d_original = lines[:, 0, :]
    # print(lines_2d_original)
    lines_2d = []
    for line in lines_2d_original:
        x1, y1, x2, y2 = line
        lines_2d.append([x1, y1, x2, y2])
        # cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # plt.imshow(img2)
    # plt.show()
    lines_2d = merge_lines(lines_2d)
    for line in lines_2d:
        x1, y1, x2, y2 = line
        if (abs(y1 - y2) > abs(x1 - x2) > 0.3 * abs(y1 - y2)) or (
                (abs(x1 - x2) > abs(y1 - y2)) and abs(y1 - y2) > 0.3 * abs(x1 - x2)):
            continue
        if abs(x1 - x2) > abs(y1 - y2) and math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 100:
            horizontal.append([x1, y1, x2, y2])
            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif abs(x1 - x2) < abs(y1 - y2) and math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 50:
            vertical.append([x1, y1, x2, y2])
            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # plt.imshow(img2)
    # plt.show()
    top_lines = []
    bottom_lines = []
    left_lines = []
    right_lines = []

    max_face = [face_rect.left(), face_rect.top(), face_rect.right() - face_rect.left(),
                face_rect.bottom() - face_rect.top()]
    face_top, face_bottom, face_left, face_right = max_face[1], max_face[1] + 1.5 * max_face[3], max_face[0], \
                                                   max_face[0] + max_face[2]
    x1, y1 = face_rect.left(), face_rect.top()
    x2, y2 = face_rect.right(), face_rect.bottom()
    # print(max_face)
    # w1, h1, w2, h2 =  8,4,256,49
    box = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
    cv2.drawContours(img2, np.array([box]), 0, (0, 255, 0), 2)

    # print("face_top",face_top)
    for line in horizontal:
        x1, y1, x2, y2 = line
        if (y1 + y2) / 2 < face_top:
            # print(y1,y2)
            top_lines.append([x1, y1, x2, y2])
            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if (y1 + y2) / 2 > face_bottom:
            bottom_lines.append([x1, y1, x2, y2])
            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for line in vertical:
        x1, y1, x2, y2 = line
        if (x1 + x2) / 2 < face_left:
            left_lines.append([x1, y1, x2, y2])
            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if (x1 + x2) / 2 > face_right:
            right_lines.append([x1, y1, x2, y2])
            cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)

    h, w = img.shape[:2]
    # top_lines.append([0, 0, w, 0])
    # bottom_lines.append([0, h, w, h])
    # left_lines.append([0, 0, 0, h])
    # right_lines.append([w, 0, w, h])

    top_lines = sorted(top_lines, key=lambda l: (l[0] - l[2]) ** 2 + (l[1] - l[3]) ** 2)
    bottom_lines = sorted(bottom_lines, key=lambda l: (l[0] - l[2]) ** 2 + (l[1] - l[3]) ** 2)
    left_lines = sorted(left_lines, key=lambda l: (l[0] - l[2]) ** 2 + (l[1] - l[3]) ** 2)
    right_lines = sorted(right_lines, key=lambda l: (l[0] - l[2]) ** 2 + (l[1] - l[3]) ** 2)

    return top_lines, bottom_lines, left_lines, right_lines


def draw_point_lines(img, lines):
    top = lines[0]
    bottom = lines[1]
    left = lines[2]
    right = lines[3]

    t_l_point = find_cross_point(top, left)
    t_r_point = find_cross_point(top, right)
    b_l_point = find_cross_point(bottom, left)
    b_r_point = find_cross_point(bottom, right)

    # 用红色画出四个顶点
    for point in t_l_point, t_r_point, b_l_point, b_r_point:
        cv2.circle(img, point, 8, (0, 0, 255), 2)

    #  用蓝色画出四条边
    cv2.line(img, t_l_point, t_r_point, (255, 0, 0), 3)
    cv2.line(img, b_r_point, t_r_point, (255, 0, 0), 3)
    cv2.line(img, b_r_point, b_l_point, (255, 0, 0), 3)
    cv2.line(img, b_l_point, t_l_point, (255, 0, 0), 3)
    #
    # cv2.line(img2, t_l_point, t_r_point, (255, 255, 255), 1)
    # cv2.line(img2, b_r_point, t_r_point, (255, 255, 255), 1)
    # cv2.line(img2, b_r_point, b_l_point, (255, 255, 255), 1)
    # cv2.line(img2, b_l_point, t_l_point, (255, 255, 255), 1)
    if is_show_point_lines == 1:
        plt.imshow(img)
        plt.show()

    return np.array([t_l_point, t_r_point, b_l_point, b_r_point])


def test_point(img, lines1, lines2, lines3):
    print(lines1, lines2, lines3)
    if len(lines1) == 4:
        draw_point_lines(img.copy(), lines1)
    if len(lines2) == 4:
        draw_point_lines(img.copy(), lines2)
    if len(lines3) == 4:
        draw_point_lines(img.copy(), lines3)


def front_correct_skew(img):
    """
    正面纠偏
    :param img: 图像
    :return: 纠偏后的图像
    """
    # img = img[5:-5, 5: -5]
    img2 = copy.deepcopy(img)
    img3 = copy.deepcopy(img)
    # lines = get_border_by_canny(img.copy())
    # if compare_light(img):
    #     lines = get_border_by_canny(img.copy())
    # else:
    #     lines = get_border_gradient(img.copy())

    # get_border_gradient(img.copy())
    # 人脸及人脸特征点检测

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("tools/shape_predictor_68_face_landmarks.dat") # C:/Users/Alexi/Desktop/IDCard_Identification/foo/shape_predictor_68_face_landmarks.dat
    faces = detector(img, 1)
    face_rect = faces[0]
    max_face = [face_rect.left(), face_rect.top(), face_rect.right() - face_rect.left(),
                face_rect.bottom() - face_rect.top()]
    shape = predictor(img, faces[0])
    # print(shape,shape.part(0).x,np.array(shape.part(0)))
    land_mark = []
    land_mark.append((shape.part(36).x, shape.part(36).y))
    land_mark.append((shape.part(45).x, shape.part(45).y))
    land_mark.append((shape.part(27).x, shape.part(27).y))
    land_mark.append((shape.part(57).x, shape.part(57).y))

    # print(land_mark)
    cv2.circle(img2, land_mark[0], 8, (0, 0, 255), 2)
    cv2.circle(img2, land_mark[1], 8, (0, 0, 255), 2)
    cv2.circle(img2, land_mark[2], 8, (0, 0, 255), 2)
    cv2.circle(img2, land_mark[3], 8, (0, 0, 255), 2)

    if is_debug == 1:
        plt.imshow(img2)
        plt.show()
    # id_rect = get_id_by_corner(img, max_face)

    id_rect = get_id_by_binary(img.copy(), max_face)
    predict_border_lines = predict_location(land_mark, max_face, id_rect)
    #print(predict_border_lines)
    if is_show_predict_lines:
        for line in predict_border_lines:
            x1, y1, x2, y2 = line
            cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 5)
        plt.imshow(img2)
        plt.show()
    # cv2.imshow("pred_line", img2)
    # cv2.waitKey(0)

    # 通过canny算子计算梯度，并检测直线
    try:
        lines_by_canny = get_border_by_canny(img.copy())
        if len(lines_by_canny) > 100:
            lines_by_canny = get_border_by_canny(img.copy(), 1)
        top_lines, bottom_lines, left_lines, right_lines = filter_and_classify_lines(img.copy(), lines_by_canny, img2.copy(),
                                                                                     face_rect)
        best_lines_by_canny = get_best_line_by_prediction(top_lines, bottom_lines, left_lines, right_lines,
                                                          predict_border_lines, img.copy())
    except Exception as e:
        best_lines_by_canny = []


    # 获取二值化后的最大轮廓的边界线
    try:
        lines_by_contour = get_border_by_binary_max_contour(img.copy())
        top_lines, bottom_lines, left_lines, right_lines = filter_and_classify_lines(img.copy(), lines_by_contour, img2.copy(),
                                                                                     face_rect)
        # best_lines_by_contour = get_best_line_by_prediction(top_lines, bottom_lines, left_lines, right_lines,
        #                                                   predict_border_lines, img.copy())
        best_lines_by_contour = select_border_lines_by_max_contour_1(top_lines, bottom_lines, left_lines, right_lines, predict_border_lines, img.copy())
        # best_lines_by_contour = select_border_lines_by_max_contour(top_lines, bottom_lines, left_lines, right_lines, img.copy())
    except Exception as e:
        best_lines_by_contour = []

    # lines_by_grabcut = get_border_by_grabcut(img.copy(), predict_border_lines)
    # top_lines, bottom_lines, left_lines, right_lines = filter_and_classify_lines(img.copy(), lines_by_grabcut,
    #                                                                              img2.copy(),
    #                                                                              face_rect)
    # # best_lines_by_grabcut = get_best_line_by_prediction(top_lines, bottom_lines, left_lines, right_lines,
    # #                                                     predict_border_lines, img.copy())
    # best_lines_by_grabcut = select_border_lines_by_max_contour(top_lines, bottom_lines, left_lines, right_lines,img.copy())
    try:
        lines_by_grabcut = get_border_by_grabcut(img.copy(), copy.deepcopy(predict_border_lines))
        top_lines, bottom_lines, left_lines, right_lines = filter_and_classify_lines(img.copy(), lines_by_grabcut, img2.copy(),
                                                                                     face_rect)
        # best_lines_by_grabcut = get_best_line_by_prediction(top_lines, bottom_lines, left_lines, right_lines,
        #                                                     predict_border_lines, img.copy())
        best_lines_by_grabcut = select_border_lines_by_max_contour_1(top_lines, bottom_lines, left_lines, right_lines, predict_border_lines, img.copy())
        #best_lines_by_grabcut = select_border_lines_by_max_contour(top_lines, bottom_lines, left_lines, right_lines,img.copy())
    except Exception as e:
        best_lines_by_grabcut = []

    #test_point(img.copy(), best_lines_by_canny, best_lines_by_contour, best_lines_by_grabcut)

    best_lines = select_best_border(id_rect, best_lines_by_canny, best_lines_by_contour, best_lines_by_grabcut)

    if is_debug == 1:
        plt.imshow(img2)
        plt.show()
    # 按直线中点位置排序
    # horizontal = sorted(horizontal, key=lambda l: l[1] + l[3])
    # vertical = sorted(vertical, key=lambda l: l[0] + l[2])

    return best_lines, img2


def correct_skew(img, is_front, max_face=[0, 0, 0, 0]):
    """
        检测最大轮廓并进行透视变换和裁剪
        默认大小1400x900 （身份证比例
        :param save_path: 存储路径, 处理后的图像保存在指定路径, 文件名和源文件相同
        :param show_process: 显示处理过程
        :param pic_path: 原图路径
        :return:
        """
    # img = cv2.GaussianBlur(img, (5, 5), 1)
    if is_front == 1:
        # try:
        img_original = copy.deepcopy(img)
        best_lines, img2 = front_correct_skew(img)

        # 计算交点
        # print(best_lines)
        top = best_lines[0]
        bottom = best_lines[1]
        left = best_lines[2]
        right = best_lines[3]

        t_l_point = find_cross_point(top, left)
        t_r_point = find_cross_point(top, right)
        b_l_point = find_cross_point(bottom, left)
        b_r_point = find_cross_point(bottom, right)
        # 用红色画出四个顶点
        for point in t_l_point, t_r_point, b_l_point, b_r_point:
            cv2.circle(img, point, 8, (0, 0, 255), 2)

        # 用蓝色画出四条边
        cv2.line(img, t_l_point, t_r_point, (255, 0, 0), 3)
        cv2.line(img, b_r_point, t_r_point, (255, 0, 0), 3)
        cv2.line(img, b_r_point, b_l_point, (255, 0, 0), 3)
        cv2.line(img, b_l_point, t_l_point, (255, 0, 0), 3)
        # plt.imshow(img)
        # plt.show()
        # cv2.line(img2, t_l_point, t_r_point, (255, 255, 255), 1)
        # cv2.line(img2, b_r_point, t_r_point, (255, 255, 255), 1)
        # cv2.line(img2, b_r_point, b_l_point, (255, 255, 255), 1)
        # cv2.line(img2, b_l_point, t_l_point, (255, 255, 255), 1)
        if is_debug == 1:
            plt.imshow(img2)
            plt.show()
            cv2.imwrite("img_test1.jpg", img2)
        if is_debug == 1:
            plt.imshow(img)
            plt.show()
        # return  img
        # 透视变换
        width = 500  # 生成图的宽
        height = 316  # 生成图的高
        # 原图中的四个角点
        pts1 = np.float32([list(t_l_point), list(t_r_point), list(b_l_point), list(b_r_point)])
        # 变换后分别在左上、右上、左下、右下四个点
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        # 生成透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # 进行透视变换
        dst = cv2.warpPerspective(img_original, M, (width, height))
        # plt.imshow(dst)
        # plt.show()
        if is_debug == 1:
            plt.imshow(dst)
            plt.show()
        # plt.imshow(dst)
        # plt.show()
        # 保存图片
        # cv2.imwrite(output_dir + filename, dst)
        return dst
    # except AttributeError as e:
    #     print('读取文件失败: ', e.args)
    #     return img_original
    # except Exception as e:
    #     print('图像矫正模块遇到未知错误: ', e.args)
    #     return img_original
    else:

        img_original = copy.deepcopy(img)
        # lines = get_border_by_canny(img.copy())
        # if compare_light(img):
        #     lines = get_border_by_canny(img.copy())
        # else:
        #     lines = get_border_gradient(img.copy())

        img, top, bottom, left, right = back_correct_skew(img)
        # print( top, bottom, left, right)
        t_l_point = find_cross_point(top, left)
        t_r_point = find_cross_point(top, right)
        b_l_point = find_cross_point(bottom, left)
        b_r_point = find_cross_point(bottom, right)
        # 用红色画出四个顶点
        for point in t_l_point, t_r_point, b_l_point, b_r_point:
            cv2.circle(img, point, 8, (0, 0, 255), 2)

        # 用蓝色画出四条边
        cv2.line(img, t_l_point, t_r_point, (255, 0, 0), 3)
        cv2.line(img, b_r_point, t_r_point, (255, 0, 0), 3)
        cv2.line(img, b_r_point, b_l_point, (255, 0, 0), 3)
        cv2.line(img, b_l_point, t_l_point, (255, 0, 0), 3)
        if is_debug == 1:
            plt.imshow(img)
            plt.show()
        # return  img
        # 透视变换
        width = 500  # 生成图的宽
        height = 316  # 生成图的高
        # 原图中的四个角点
        pts1 = np.float32([list(t_l_point), list(t_r_point), list(b_l_point), list(b_r_point)])
        # 变换后分别在左上、右上、左下、右下四个点
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        # 生成透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # 进行透视变换
        dst = cv2.warpPerspective(img, M, (width, height))
        if is_debug == 1:
            plt.imshow(dst)
            plt.show()
        # plt.imshow(dst)
        # plt.show()
        # 保存图片
        # cv2.imwrite(output_dir + filename, dst)
        return dst


# if __name__ == "__main__":
#     is_batch = 0
#     if is_batch == 0:
#         # path = "F:/idcard/problem_20190716/images/0123.jpg"
#         path = "F:/idcard/sfz/sfz_front/2aff5978-4473-412d-9927-c813de8242d9.jpeg"
#         # path = "F:/idcard/3(1)/3/images/65760311F8774D3B821B59240D35CB10.JPG"
#         img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
#         img = resize(img.copy(), width=500)
#         classfier = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
#         if len(img.shape) == 3:
#             grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
#         if len(faceRects) > 0:
#             max_face = faceRects[np.where(faceRects[:, 3] == faceRects[:, 3].max())]
#             # max_face = [[438, 187, 107, 108]]
#             img = correct_skew(img, 1, max_face[0])
#             # cv2.imwrite("2/output1/65E8225D40174C949AC06E6E82794C21.JPG", img)
#         else:
#             img = correct_skew(img, 0)
#     else:
#         # input_dir = "F:/idcard/problem_20190716/images/"
#         # output_dir = "F:/idcard/problem_20190716/output5/"
#
#         # input_dir = "F:/idcard/3(1)/3/images/"
#         # output_dir = "F:/idcard/3(1)/3/output4/"
#
#         input_dir = "F:/idcard/problem_20190716/images/"
#         output_dir = "F:/idcard/problem_20190716/output8/"
#
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         for filename in os.listdir(input_dir):
#             if len(filename.split(".")) < 2:
#                 continue
#             print(filename)
#             path = input_dir + filename
#             # 读取图片
#             img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
#             img = resize(img.copy(), width=500)
#             # img = cv2.resize(img, (imgWidth, imgHeight))
#             classfier = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
#             if len(img.shape) == 3:
#                 grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             else:
#                 gray = img
#             faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
#             # 判断是否是正面,大于0则检测到人脸,是正面
#             if len(faceRects) > 0:
#                 max_face = faceRects[np.where(faceRects[:, 3] == faceRects[:, 3].max())]
#                 dst = correct_skew(img, 1, max_face[0])
#                 # plt.imshow(dst)
#                 # plt.show()
#                 # print(output_dir + filename)
#                 cv2.imwrite(output_dir + filename, dst)
#             else:
#                 pass
#                 # dst = correct_skew(img, 0)
#                 # cv2.imwrite(output_dir + filename, dst)
