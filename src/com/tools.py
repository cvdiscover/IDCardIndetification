import cv2
import matplotlib.pylab as plt
import numpy as np
import copy
import math


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


def project(img_binary, orientation=0, is_show = 0):
    """
    图片投影
    :param img_binary: 二值图片
    :param orientation: 投影方向 0：水平投影  1：垂直投影
    :return:水平方向角點
    """
    img_binary = np.array(img_binary)
    (h, w) = img_binary.shape  # 返回高和宽
    # print(h,w)#s输出高和宽
    thresh_copy = img_binary.copy()
    thresh_copy[:, :] = 255
    # 水平投影
    if orientation == 0:
        # horizontal_count = np.array([0 for z in range(0, h)])
        # # 每行白色像素个数
        # for j in range(0, h):
        #     horizontal_count[j] = np.count_nonzero(img_binary[j, :])
        horizontal_count = img_binary.sum(axis=1) / 255
        count = horizontal_count
        for j in range(0, h):  # 遍历每一行
            # for i in range(0, int(horizontal_count[j])):
            #     thresh_copy[j, i] = 0
            thresh_copy[j, 0:int(horizontal_count[j])] = 0
    else:
        # vertical_count = np.array([0 for z in range(0, w)])
        # # 每列白色像素个数
        # for j in range(0, w):
        #     vertical_count[j] = np.count_nonzero(img_binary[:, j])
        vertical_count = img_binary.sum(axis=0) / 255
        count = vertical_count
        for j in range(0, w):  # 遍历每一列
            # for i in range((h - int(vertical_count[j])), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            #     thresh_copy[i, j] = 0  # 涂黑
            thresh_copy[(h - int(vertical_count[j])): h, j] = 0
    return count


def check_idnumber(img):

    imgHeight, imgWidth = 316, 500
    id_y = int(imgHeight / 1.3)
    id_x = int(imgWidth / 3.06)
    id_addHeight = int(id_y + imgHeight / 6.70)
    id_addWidth = int(id_x + 322)
    id = img[id_y:id_addHeight, id_x:id_addWidth]
    img_gray = cv2.cvtColor(id, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.medianBlur(img_gray, 5)
    th, img_binary = cv2.threshold(img_gray_blur, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # plt.imshow(img_binary, cmap=plt.gray())
    # plt.show()
    horizonal_number = project(img_binary, 0)
    best_center = get_best_region(horizonal_number)
    if best_center > 40 or best_center < 12:
        return 1
    else:
        return 0


def get_best_region(count):
    """
    根据图片像素投影分割图片
    :param count: 投影值
    :return:
    """

    k = 20
    # 当前27个数据和
    sum = 0
    # 最大27个数据和
    best_score = 0
    # 当前27个数据
    ring_buffer = np.zeros(k)
    # 最佳偏移
    best_offset = 0

    for y_offset in range(len(count)):
        score = count[y_offset]
        sum += score
        buffer_index = y_offset % k
        ring_buffer[buffer_index] = score

        if (y_offset >= k - 1):
            if (sum > best_score):
                best_score = sum
                best_offset = y_offset - k + 1
            next_buffer_index = (y_offset + 1) % k
            sum -= ring_buffer[next_buffer_index]

    return (best_offset+  int(k / 2))


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


def line_length(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return length


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
