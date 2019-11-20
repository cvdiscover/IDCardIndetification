import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from foo.tools.back_correct_skew import *
from foo.tools.config import *


# 裁剪好的图像寻找信息位置
def find_information(result, img):
    """
    找到纠偏后图像中的（中华人民共和国）和（居民身份证）的位置
    :param result:裁剪后的图像
    :return:（中华人民共和国）和（居民身份证）的位置
    """

    result1, cut_h = image_select(result.copy())

    result = cv2.bilateralFilter(result1, 0, 100, 5)
    if is_debug == 1:
        cv2.imshow("bilateralFilter", result)
        cv2.waitKey(0)
        plt.hist(result.ravel(), 256, [0, 256])
        plt.show()

    # 阈值设置为80可以有效地去除大部分边框和国徽
    # ret, thresh = cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
    # res2 = cv2.adaptiveThreshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5)

    # thresh = cv2.adaptiveThreshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 256, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 80, 1)
    # ret, thresh = cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 135, 255, cv2.THRESH_BINARY)
    if is_debug == 1:
        cv2.imshow("thresh o", thresh)
        cv2.waitKey(0)
        plt.hist(thresh.ravel(), 256, [0, 256])
        plt.show()

    # thresh = cv2.bilateralFilter(result, 0, 100, 5)

    canny = cv2.Canny(thresh, 10, 150, 20)  # 50是最小阈值,150是最大阈值
    if is_debug == 1:
        cv2.imshow('canny1', canny)
        cv2.waitKey(0)

    # 开操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    canny = cv2.dilate(canny, kernelX)
    morphologyEx = open_demo(canny)
    # 闭操作
    # kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    # morphologyEx = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernelX)

    if is_debug == 1:
        cv2.imshow('morphologyEx1', morphologyEx)
        cv2.waitKey(0)

    # 膨胀腐蚀
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    Element = cv2.dilate(morphologyEx, kernelX)
    Element = cv2.dilate(Element, kernelY)
    Element = cv2.erode(Element, kernelX)
    Element = cv2.erode(Element, kernelY)

    if is_debug == 1:
        cv2.imshow('getStructuringElement', Element)
        cv2.waitKey(0)

    # 闭操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    morphologyEx = cv2.morphologyEx(Element, cv2.MORPH_CLOSE, kernelX)

    if is_debug == 1:
        cv2.imshow('morphologyEx2', morphologyEx)
        cv2.waitKey(0)

    _, contours, hair = cv2.findContours(morphologyEx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # over = cv2.drawContours(result, contours, -1, (255, 0, 0), 1)
    # cv2.imshow("counters", over)
    # cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    rectangle = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 矩形边框（boundingRect）
        if cv2.contourArea(c) < 2000:
            continue
        # if 2 <= w / h <= 5:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        # elif 0.5 <= w / h <= 1.5:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif 4 < w / h <= 20:
            rectangle.append((x, y, w, h))

    rectangle = sorted(rectangle, key=lambda a: a[2] * a[3], reverse=True)
    rec_len = len(rectangle)
    if rec_len > 2:
        for i in range(0, 2):
            x, y, w, h = rectangle[i]
            cv2.rectangle(img, (x, y + cut_h), (x + w, y + h + cut_h), (0, 255, 255), 2)
    else:
        for i in range(0, rec_len):
            x, y, w, h = rectangle[i]
            cv2.rectangle(img, (x, y + cut_h), (x + w, y + h + cut_h), (0, 255, 255), 2)

    return img


# 找到纠偏后图像中的（中华人民共和国）和（居民身份证）的位置
def find_information_bymark(result, img):
    """
    找到纠偏后图像中的（中华人民共和国）和（居民身份证）的位置
    :param img: 纠偏后的原图
    :param result:图像的角点图片
    :return:（中华人民共和国）和（居民身份证）的位置
    """

    # 闭操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
    morphologyEx = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernelX)

    if is_debug == 1:
        cv2.imshow('morphologyEx1', morphologyEx)
        cv2.waitKey(0)

    # 膨胀腐蚀
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    Element = cv2.dilate(morphologyEx, kernelX)
    Element = cv2.dilate(Element, kernelY)
    Element = cv2.erode(Element, kernelX)
    Element = cv2.erode(Element, kernelY)

    if is_debug == 1:
        cv2.imshow('getStructuringElement', Element)
        cv2.waitKey(0)

    # 闭操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphologyEx = cv2.morphologyEx(Element, cv2.MORPH_CLOSE, kernelX)

    if is_debug == 1:
        cv2.imshow('morphologyEx2', morphologyEx)
        cv2.waitKey(0)

    _, contours, hair = cv2.findContours(morphologyEx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # over = cv2.drawContours(result, contours, -1, (255, 0, 0), 1)
    # cv2.imshow("counters", over)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 矩形边框（boundingRect）
        if cv2.contourArea(c) < 200:
            continue
        if 2 <= w / h <= 5:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        elif 0.7 <= w / h <= 1.3:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif 5 < w / h <= 20:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    lower_hsv = np.array([156, 43, 46])
    upper_hsv = np.array([180, 255, 255])
    mask = cv2.inRange(img, lower_hsv, upper_hsv)
    if is_debug == 1:
        cv2.imshow("over", img)
        cv2.imshow("mask", mask)
    return img


# 开操作 腐蚀+膨胀 cv.MORPH_OPEN
def open_demo(image):  # 开操作 腐蚀+膨胀 cv.MORPH_OPEN
    print(image.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    if is_debug == 1:
         cv2.imshow("open_result", binary)
    return binary


# 闭操作 膨胀+腐蚀 cv.MORPH_MORPH_CLOSE
def close_demo(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    if is_debug == 1:
        cv2.imshow("close_result", binary)
    return binary


# 国徽识别
def flann_univariate_matching(img):
    """
    国徽识别
    :param img:经矫正处理后的图像
    :return:框出国徽的图像和框的四个点
    """

    # 确保至少有一定数目的良好匹配（计算单应性最少需要4个匹配），将其设定为10，在实际中可能会使用一个比10大的值
    MIN_MATCH_COUNT = 15
    # 首先加载两幅图（查询图像和训练图像）
    img1 = cv2.imread(guohui_direct, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # img3 = cv2.imread('C:/Users/Alexi/Desktop/idcard_info/sfz_back/b323f2a9-fbb9-4e99-9708-d792f779792a.jpeg')
    if is_debug == 1:
        plt.imshow(img2), plt.show()
    # 创建 SIFT 和 detect / compute
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN 匹配参数
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    matches = flann.knnMatch(des1, des2, k=2)
    # print(help(flann.knnMatch))
    # print(matches)

    # store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        # print('m', m)
        # print('n', n)
        if m.distance < 0.9 * n.distance:
            good.append(m)

    min_len = min(len(good), len(kp1), len(kp2))
    if len(good) > MIN_MATCH_COUNT:
        # 在原始图像和训练图像中发现关键点
        # 有些图像会报错：IndexError: list index out of range
        # 有些图像不会报错，
        # 为了使得该代码具有普适性，增加异常处理
        try:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, M)  # 保存包含目标区域的一个矩形框的4个坐标

            if dst is not None:
                img2 = cv2.polylines(img, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)  # 将矩形框在训练图像中画出
                img2 = cv2.circle(img2, (dst[0][0][0], dst[0][0][1]), 8, (255, 0, 0), -1)
                if is_debug == 1:
                    plt.imshow(img2, 'gray'), plt.show()
                    cv2.imshow("result", img2)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                return img2, np.int32(dst)
            else:
                print("dst is none")
                return None, None
        except IndexError as IE:
            print(IE)
    else:
        print("Not enough matches are found - %d%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return None, None


# 计算边界
def predict_edge(dst, img_w, img_h):
    """

    :param img_h: 图像的高
    :param img_w: 图像的宽
    :param dst: 国徽的位置
    :return: 返回预估的四个角点 顺序为左下 右下 右上 左上
    """
    # gh_points 顺序为左上 左下 右下 右上 以下称为1， 2， 3， 4
    gh_points = [(dst[0][0][0], dst[0][0][1]), (dst[1][0][0], dst[1][0][1]), (dst[2][0][0], dst[2][0][1]),
                 (dst[3][0][0], dst[3][0][1])]
    # pre_edge_points 顺序为左 上 右 下
    pre_edge_points = []
    pre_angle_points = []
    k_set = []
    b_set = []
    length = []
    for j in range(4):  # 四条边顺序分别是12，23，34，41 即kb分别为左 下 右 上 四条边的斜率和偏差
        # print(i % 4)
        # print((i + 1) % 4)
        length.append(math.sqrt((gh_points[(j + 1) % 4][0] - gh_points[j % 4][0]) ** 2
                                + (gh_points[(j + 1) % 4][1] - gh_points[j % 4][1]) ** 2))
        if gh_points[(j + 1) % 4][0] == gh_points[j % 4][0]:
            k = "max"
            b_set.append(gh_points[j % 4][0])
        else:
            k = (gh_points[(j + 1) % 4][1] - gh_points[j][1]) / (gh_points[(j + 1) % 4][0] - gh_points[j][0])
            b_set.append(gh_points[j][1] - k * gh_points[j][0])
        k_set.append(k)

    for j in range(1, 5):  # 预估四个点的位置相对国徽左上角点分别为左 下 右 上
        if j % 2 == 0:
            pre_x = gh_points[0][0] - (gh_points[1][0] - gh_points[0][0]) * proportion[j-1]
            pre_y = gh_points[0][1] - (gh_points[1][1] - gh_points[0][1]) * proportion[j-1]
        else:
            pre_x = gh_points[0][0] - (gh_points[3][0] - gh_points[0][0]) * proportion[j-1]
            pre_y = gh_points[0][1] - (gh_points[3][1] - gh_points[0][1]) * proportion[j-1]
        pre_edge_points.append((int(pre_x), int(pre_y)))
        if k_set[j - 1] is not "max":
            b_set[j - 1] = int(pre_y) - k_set[j - 1] * int(pre_x)
        else:
            b_set[j - 1] = int(pre_x)

    for i in range(4):
        angle_x, angle_y = extend_line_bykb(k_set[i % 4], b_set[i % 4], k_set[(i + 1) % 4], b_set[(i + 1) % 4])
        if angle_x < 0:
            angle_x = 0
        elif angle_x > img_w:
            angle_x = img_w
        if angle_y < 0:
            angle_y = 0
        elif angle_y > img_h:
            angle_y = img_h
        pre_angle_points.append((angle_x, angle_y))

    return pre_angle_points


# 输入两条直线的斜率和偏移量返回两直线交点
def extend_line_bykb(k1, b1, k2, b2):
    """
    输入两条直线的斜率和偏移量返回两直线交点
    :param k1: 两条直线的直线斜率
    :param b1: 两条直线的偏移量
    :param k2: 两条直线的直线斜率
    :param b2: 两条直线的偏移量
    :return: 两直线交点
    """
    if k1 == k2:
        return False
    if k1 is not None and k2 is not None:
        if k1 is not "max" or k2 is not "max":
            if k2 is "max":
                x = b2 * 1.0
                y = k1 * x * 1.0 + b1 * 1.0
            elif k1 is "max":
                x = b1 * 1.0
                y = k2 * x * 1.0 + b2 * 1.0
            else:
                x = (b2 - b1) * 1.0 / (k1 - k2)
                y = k1 * x * 1.0 + b1 * 1.0
            # print("x -----   y -----")
            # print(int(x), int(y))
            return int(x), int(y)
    else:
        print("no accross")
        return False


# 传入四个点进行透视变换
def perspective_transformation(points, flag, src):
    """
        传入四个点进行透视变换
        :param flag: fitline: 直线检测拟合结果 else：pre预估结果
        :param points: 包含四个点的点集
        :return: 两直线交点
        """
    if flag is "fitline":
        p1, p2, p3, p4 = points  # p1-右下  p2-左下  p3-右上  p4-左上
    else:
        p2, p1, p3, p4 = points  # p1-右下  p2-左下  p3-右上  p4-左上
    # 原图中书本的四个角点 左上、右上、左下、右下
    pts1 = np.float32([p4, p3, p2, p1])
    if p2[1] - p4[1] < p3[0] - p4[0]:
        height = 500
        width = int(height * 1.58)
    else:
        width = 500
        height = int(width * 1.58)
    # 变换后分别在左上、右上、左下、右下四个点
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv2.warpPerspective(src, M, (width, height))
    # plt.subplot(121), plt.imshow(src[:, :, ::-1]), plt.title('input')
    # plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
    # img[:, :, ::-1]是将BGR转化为RGB
    if is_debug == 1:
        plt.show()
        cv2.imshow("output", dst)
    return dst


# 计算点（x3, y3）到直线((x1, x2), (x2, y2))的距离
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
    if length != 0:
        distance = 2 * area / length
    else:
        distance = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    # print(area,length,distance)
    return distance


# 直线拟合找近似边缘
def fit_line(image, points, ori):
    """
    直线拟合找近似边缘
    :param ori: 裁剪后原图
    :param points: 预估的四个顶点 顺序为左下 右下 右上 左上
    :param image:处理后的图像
    :return: 直线拟合后的四个点
    """
    result = ori.copy()
    _, contours, hair = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    rect = cv2.boundingRect(cnts[0])
    mid_point = [rect[0]+rect[2]/2, rect[1]+rect[3]/2]
    if is_debug == 1:
        cv2.imshow('line_detection_demo', image)
        cv2.waitKey(0)

    binary = cv2.Canny(image, 10, 150, 20)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 360, 100, minLineLength=10, maxLineGap=2)
    edge = []
    for i in range(4):  # 这里的edge是一个4*n的数组 0 1 2 3 分别是右左下上的线段集
        edge.append([])
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            midx = (x1 + x2) / 2
            midy = (y1 + y2) / 2
            if x1 != x2:
                k = (y2 - y1) / (x2 - x1)
            else:
                k = 999
            if abs(k) > 1:

                if midx <= mid_point[0]:
                    d1 = cal_distance(points[3][0], points[3][1], points[0][0], points[0][1], x1, y1)
                    d2 = cal_distance(points[3][0], points[3][1], points[0][0], points[0][1], x2, y2)
                    if d1 < 30 and d2 < 30:
                        edge[1].append([x1, y1])
                        edge[1].append([x2, y2])
                else:
                    d1 = cal_distance(points[1][0], points[1][1], points[2][0], points[2][1], x1, y1)
                    d2 = cal_distance(points[1][0], points[1][1], points[2][0], points[2][1], x2, y2)
                    if d1 < 30 and d2 < 30:
                        edge[0].append([x1, y1])
                        edge[0].append([x2, y2])
                cv2.line(result, (x1, y1), (x2, y2), (255, 255, 0), 2)
            else:
                if midy <= mid_point[1]:
                    d1 = cal_distance(points[2][0], points[2][1], points[3][0], points[3][1], x1, y1)
                    d2 = cal_distance(points[2][0], points[2][1], points[3][0], points[3][1], x2, y2)
                    if d1 < 30 and d2 < 30:
                        edge[3].append([x1, y1])
                        edge[3].append([x2, y2])
                else:
                    d1 = cal_distance(points[1][0], points[1][1], points[0][0], points[0][1], x1, y1)
                    d2 = cal_distance(points[1][0], points[1][1], points[0][0], points[0][1], x2, y2)
                    if d1 < 30 and d2 < 30:
                        edge[2].append([x1, y1])
                        edge[2].append([x2, y2])
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if is_debug == 1:
        for i in range(4):
            cv2.imshow("fitline123", result)
            cv2.waitKey(0)

    if len(edge[0]) is 0 or len(edge[1]) is 0 or len(edge[2]) is 0 or len(edge[3]) is 0:
        print("未能拟合出四条边缘 改为使用预估位置")
        return None
        # min_outside(image)
    else:
        k_set = []
        b_set = []
        points = []
        for i in range(len(edge)):  # 0 1 2 3 分别是左右上下的线段集
            output = cv2.fitLine(np.array(edge[i]), cv2.DIST_L2, 0, 0.01, 0.01)
            k_set.append(output[1] / output[0])
            b_set.append(output[3] - (output[1] / output[0]) * output[2])
        for i in range(2, 4):
            for j in range(0, 2):
                points.append(extend_line_bykb(k_set[i], b_set[i], k_set[j], b_set[j]))

        return points


# 选取信息蒙版
def image_select(after_cut_image):
    """

    :param after_cut_image: 裁剪后视为准确的图像
    :return: 上版部分置为白色消除部分干扰
    """
    # mask = np.zeros(after_cut_image.shape[:2], dtype="uint8")
    H, W = after_cut_image.shape[:2]
    ymin, ymax, xmin, xmax = int(H / 2), 480, 0, W
    # ymin, ymax, xmin, xmax, ydmin, ydmax = 0, int(H/2), 0, W, 490, H
    # for i in range(W):
    #     for j in range(H):
    #         if ymax > j > ymin and xmax> i > xmin:  # 注意长宽与数组行列的对应关系
    #             after_cut_image[j - 1, i - 1] = 255
    #         elif ydmax > j > ydmin and xmax > i > xmin:
    #             after_cut_image[j - 1, i - 1] = 255
    after_cut_image = after_cut_image[ymin: ymax, xmin: xmax]
    if is_debug == 1:
        cv2.imshow("after_cut_image", after_cut_image)
        cv2.waitKey(0)

    return after_cut_image, int(H / 2)


# 基本操作
def basic_demo(image):
    """
        基本操作
        :param image: 输入图像
        :return: 双边滤波后的图像
        """
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(image, 0, 100, 15)
    # cv2.imshow("bi_demo", img)
    # cv2.waitKey(0)

    edges = cv2.Canny(img, 10, 150, 20)  # 50是最小阈值,150是最大阈值
    # cv2.imshow('Canny', edges)
    # cv2.waitKey(0)
    return edges


if __name__ == '__main__':

    # src = cv.resize(src, (500, 500), interpolation=cv.INTER_CUBIC)

    # src = cv.resize(src, (500, 500), interpolation=cv.INTER_CUBIC)
    input_dir = "C:/Users/Alexi/Desktop/idcard_info/sfz_back/"
    output_dir = "C:/Users/Alexi/Desktop/idcard_info/fit_point_info_backup/"

    is_batch = 0
    if is_batch:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for filename in os.listdir(input_dir):
            if len(filename.split(".")) < 2:
                continue
            print(filename)
            path = input_dir + filename
            # 读取图片
            src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            src = resize(src.copy(), width=500)
            img, text1, text2 = correct_image(src.copy())
            # img_mark = mark_corner_image(img)
            result, dst = flann_univariate_matching(img.copy())
            if result is not None and dst is not None:
                if dst[0][0][1] > dst[1][0][1]:
                    result, dst = flann_univariate_matching(cv2.flip(cv2.flip(img.copy(), 1), 0))
                if result is not None and dst is not None:
                    try:
                        (h, w) = img.copy().shape[:2]
                        pre_angle_points = predict_edge(dst, w, h)
                        after_basic_img = close_demo(basic_demo(result.copy()))

                        fit_points = fit_line(after_basic_img, pre_angle_points, result.copy())
                        if fit_points is None:
                            result = perspective_transformation(pre_angle_points, None, result.copy())
                        else:
                            result = perspective_transformation(fit_points, "fitline", result.copy())
                        marked_image = find_information(result, result.copy())
                        if marked_image is not None:
                            cv2.imencode('.jpg', marked_image)[1].tofile(str(output_dir + filename))
                    except Exception as E:
                        print(E)
    else:  # 22c3b8ef-8f11-4b71-8173-6d1bab663d2d  76fff000-fdfa-4bb4-b4b6-21fb999837cd
        fileName = "C:/Users/Alexi/Desktop/idcard_info/sfz_back/97499342-c9d5-4ade-b02f-60d4b73756f0.jpeg"  # perspective_transformation_
        src = cv2.imread(fileName)
        src = resize(src.copy(), width=500)

        img, text1, text2 = correct_image(src.copy())
        cv2.imshow("img", img)
        result, dst = flann_univariate_matching(img.copy())

        cv2.imshow("before", result)

        if result is not None and dst is not None:
            if dst[0][0][1] > dst[1][0][1]:
                result, dst = flann_univariate_matching(cv2.flip(cv2.flip(img.copy(), 1), 0))
            (h, w) = img.copy().shape[:2]
            pre_angle_points = predict_edge(dst, w, h)
            # try:
            # for i in range(4):
            #     result0 = cv2.circle(result.copy(), pre_angle_points[i], 8, (255, 0, 0), -1)
            #     cv2.imshow("result0", result0)
            #     cv2.waitKey(0)
            after_basic_img = close_demo(basic_demo(result.copy()))
            cv2.imshow("after_basic_img", after_basic_img)
            cv2.waitKey(0)
            fit_points = fit_line(after_basic_img, pre_angle_points, result.copy())
            # for i in range(4):
            #     result1 = cv2.circle(result.copy(), fit_points[i], 8, (255, 0, 0), -1)
            #     cv2.imshow("result1", result1)
            #     cv2.waitKey(0)
            if fit_points is None:
                result = perspective_transformation(pre_angle_points, None, result.copy())
            else:
                result = perspective_transformation(fit_points, "fitline", result.copy())

            marked_image = find_information(result, result.copy())
            cv2.imshow("marked_image", marked_image)
            cv2.waitKey(0)
            # except Exception as E:
            #     print(E)
            cv2.destroyAllWindows()
