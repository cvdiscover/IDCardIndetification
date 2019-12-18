from src.config.config import *
from src.com.tools import *
import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np


# 根据长宽比比例进行大小调整
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):#插值方法选用基于局部像素的重采样
    dim = None
    (h, w) = image.shape[:2]  #image.shape[0]指图片长（垂直尺寸），image.shape[1]指图片宽（水平尺寸）
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)  #h为真实高度，height为设定高度0
        dim = (int(w * r), height)  #dim括号内的是按比例调整后的长和宽
    else:
        r = width / float(w)  #为真实宽度，width为设定宽度
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)  #interpolation为插值方法，缩写inter
    return resized


def cv_show(name, img):
    cv2.imshow(name,img)
    cv2.waitKey(0)


def check_location(img, region):
    """
    检测位置是否正确
    :param img: 原图
    :param region: 各信息位置
    :return:
    """

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
    elif region[7][0][1] + 1/2* region[7][0][3] > region[8][0][1] + region[8][0][3]:
        return 1
    else:
        return 0


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
    #
    # 身份证号码左上角垂直距离
    id = int(imgHeight / 1.3)
    # 相片右下角垂直位置
    photo = face_rect[0][1] + face_rect[0][3] + 60
    distance = abs(photo-id)+8

    #精准确定照片位置
    photo_x1 = face_rect[0][0] - 40
    photo_x2 = face_rect[0][0] + face_rect[0][2] + 40
    photo_y1 = face_rect[0][1] - 40
    photo_y2 = face_rect[0][1] + face_rect[0][3] + 60
    photo_cut = img[photo_y1:photo_y2, photo_x1:photo_x2]
    img=cv2.rectangle(img, (photo_x1, photo_y1), (photo_x2, photo_y2), (0, 0, 255), 2)
    if is_debug ==1:
        plt.imshow(img,cmap=plt.gray())
        plt.show()
    photo_region = get_photo_position(photo_cut)
    photo_region[0][0] += photo_x1
    photo_region[0][1] += photo_y1

    # 地址
    address_y = int(imgHeight / 2.11)+distance
    address_x = 87
    address_addHeight = int(address_y + imgHeight / 3.17)
    address_addWidth = photo_region[0][0]
    address = img[address_y:address_addHeight, address_x:address_addWidth]
    img = cv2.rectangle(img, (address_x, address_y), (address_addWidth, address_addHeight), (0, 0, 255), 2)
    try:
        address_region = get_regions(address, 1, 1, is_consider_color=0)
    except Exception as e:
        address_region = get_regions(address, 1, 1, is_consider_color=0)
    for i in range(len(address_region)):
        address_region[i][0] += address_x
        address_region[i][1] += address_y
    left_position = address_region[:, 0].min() - 15

    # 名字
    name_y = int(imgHeight / 10.8) - 10+distance
    name_x = left_position
    name_addHeight = int(name_y + 46) + 10
    name_addWidth = int(name_x + imgWidth / 4)
    name = img[name_y:name_addHeight, name_x:name_addWidth]
    img = cv2.rectangle(img, (name_x, name_y), (name_addWidth, name_addHeight), (0, 0, 255), 2)
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
    sex_y = int(imgHeight / 4.5)+distance
    sex_x = left_position
    sex_addHeight = int(sex_y + imgHeight / 7.71)
    sex_addWidth = int(sex_x + 45)
    sex = img[sex_y:sex_addHeight, sex_x:sex_addWidth]
    img = cv2.rectangle(img, (sex_x, sex_y), (sex_addWidth, sex_addHeight), (0, 0, 255), 2)
    try:
        sex_region = get_regions(sex, 1)
        if len(sex_region) == 0:
            sex_region = get_regions(sex, 1, is_consider_color=0)
    except Exception as e:
        sex_region = get_regions(sex, 1, is_consider_color=0)

    sex_region[0][0] += sex_x
    sex_region[0][1] += sex_y
    regions.append(sex_region)

    # 民族
    nationality_y = int(imgHeight / 4.5)+distance
    nationality_x = left_position + 90
    nationality_addHeight = int(nationality_y + imgHeight / 7.71)
    nationality_addWidth = int(nationality_x + 50)
    nationality = img[nationality_y:nationality_addHeight, nationality_x:nationality_addWidth]
    img = cv2.rectangle(img, (nationality_x, nationality_y), (nationality_addWidth, nationality_addHeight), (0, 0, 255), 2)
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
    birth_year_y = int(imgHeight / 2.84)+distance
    birth_year_x = left_position
    birth_year_addHeight = int(birth_year_y + imgHeight / 7.71)
    birth_year_addWidth = int(birth_year_x + 68)
    birth_year = img[birth_year_y:birth_year_addHeight, birth_year_x:birth_year_addWidth]
    img = cv2.rectangle(img, (birth_year_x, birth_year_y), (birth_year_addWidth, birth_year_addHeight), (0, 0, 255), 2)
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
    birth_month_y = int(imgHeight / 2.84)+distance
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
    birth_day_y = int(imgHeight / 2.84)+distance
    birth_day_x = left_position + 125
    birth_day_addHeight = int(birth_day_y + imgHeight / 7.71)
    birth_day_addWidth = int(birth_day_x + 37)
    birth_day = img[birth_day_y:birth_day_addHeight, birth_day_x:birth_day_addWidth]
    img = cv2.rectangle(img, (birth_day_x, birth_day_y), (birth_day_addWidth, birth_day_addHeight), (0, 0, 255), 2)
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
    id_y = int(imgHeight / 1.3)+distance
    id_x = int(imgWidth / 3.06)
    id_addHeight = int(id_y + imgHeight / 6.70)
    id_addWidth = int(id_x + 322)
    id = img[id_y:id_addHeight, id_x:id_addWidth]
    img = cv2.rectangle(img, (id_x, id_y), (id_addWidth, id_addHeight), (0, 0, 255), 2)
    try:
        id_region = get_regions(id, 1, is_consider_color = 0)
    except Exception as e:
        id_region = get_regions(id, 1, is_consider_color=0)
    id_region[0][0] += id_x
    id_region[0][1] += id_y
    regions.append(id_region)
    regions.append(photo_region)

    regions = complete_box(regions)
    img_copy = copy.deepcopy(img)
    for i in range(len(regions)):

        # rect[0],rect[1],rect[2],rect[3]  分别为矩形左上角点的横坐标，纵坐标，宽度，高度
        for j in range(len(regions[i])):
            rect = regions[i][j]
            x1, x2 = rect[0], rect[0] + rect[2]
            y1, y2 = rect[1], rect[1] + rect[3]
            box = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
            cv2.drawContours(img_copy, np.array([box]), 0, (0, 255, 0), 1)
    if is_debug == 1:
        plt.imshow(img_copy, cmap=plt.gray())
        plt.show()
    cv2.imencode('.jpg', img_copy)[1].tofile(str(save_name))
    return regions


def get_photo_position(img):
    """
    :param img:原图
    :return:返回身份证相片框
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (7, 7), 1)
    edges = cv2.Canny(img_gray_blur, 70, 20)  # Canny算子边缘检测
    _, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_contour = img_binary | edges

    erode_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_dilate = cv2.dilate(img_contour, dilate_elment, iterations=1)
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
    address_x = regions[6][:, 0].min()
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


# 获取框内文字信息
def get_regions(img, scale, is_address=0, is_name=0, is_date=0, is_front = 1, is_consider_color = 1):
    """
    对单块区域处理后，获取文文本位置
    :param is_name:
    :param img: 图片
    :param is_address: 是否是地址
    :return: 文本位置
    """
    img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    th, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 根据背景文字颜色为蓝色特点去除背景文字
    if is_consider_color == 1:
        if is_address == 1:
            img_binary[(np.where(img[:, :, 0] > img[:, :, 1] + 10) or np.where(img[:, :, 0] > img[:, :, 2] + 10))] = 0
        elif is_address == 0 and is_front == 1:
            img_binary[np.where(img[:, :, 0] > img[:, :, 1] + 10)] = 0
            img_binary[np.where(img[:, :, 0] > img[:, :, 2] + 10)] = 0

    if is_address:
        for i in reversed(range(img.shape[1] - 20, img.shape[1])):
            if cv2.countNonZero(img_binary[:, i]) > 25:
                img_binary[:, i] = 0

    #  进行形态学开运算
    img_open = morphology(img_binary, scale, is_date)

    # 获取文字区域位置
    text_region = find_word_regions(img_open, is_address, is_name=is_name, is_date=is_date)

    # 在原图片上绘制文字区域矩形和剪切文字区域
    for i in range(len(text_region)):

        # rect[0],rect[1],rect[2],rect[3]  分别为矩形左上角点的横坐标，纵坐标，宽度，高度
        rect = text_region[i]
        x1, x2 = rect[0], rect[0] + rect[2]
        y1, y2 = rect[1], rect[1] + rect[3]
        box = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
        cv2.drawContours(img, np.array([box]), 0, (0, 255, 0), 2)
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
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # 取面积最大的轮廓
    if is_address == 0:
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
    while i < len(rects):
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


def max_face_detect(faces, image):
    left = faces[0].left()
    top = faces[0].top()
    right = faces[0].right()
    bottom = faces[0].bottom()
    Wwidth = right - left
    Hheigt = top - bottom

    width = right - left
    high = top - bottom
    left2 = np.uint(left - 0.3*width)
    bottom2 = np.uint(bottom + 0.6*width)
    img = cv2.rectangle(image, (left2, bottom2 - 20), (left2 + 2 * width - 50, bottom2 + 2 * high), (0, 0, 255), 2)
    # cv_show('img',img)

    top2 = np.uint(bottom2 + 1.8 * high)
    right2 = np.uint(left2 + 1.6 * width)
    # 人像框范围
    max_face = image[top2:bottom2, left2:right2, :]
    # cv_show('face_img',max_face)
    return max_face, img, width, high, left2, bottom2, right2


# 开操作 腐蚀+膨胀 cv.MORPH_OPEN
def open_demo(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return binary


# 闭操作 膨胀+腐蚀 cv.MORPH_CLOSE
def close_demo(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("close_result", binary)
    return binary

# gamma函数处理
def gamma_trans(img, gamma):
    '''
    :param img:纠偏
    :param gamma:
    :return:
    '''
    # 建立映射表
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    # 颜色值为整数
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。
    return cv2.LUT(img, gamma_table)


def getpart_info(part):
    """
    :param part: 身份证信息框大体位置
    :return: 身份证信息轮廓
    """

    result = cv2.bilateralFilter(part.copy(), 0, 50, 5)
    # cv2.imshow("bilateralFilter", result)
    # cv2.waitKey(0)

    thresh = cv2.adaptiveThreshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 5)
    # cv2.imshow("thresh o", thresh)
    # cv2.waitKey(0)

    # 开操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    morphologyEx = cv2.dilate(thresh, kernelX)
    morphologyEx = open_demo(morphologyEx)

    # cv2.imshow('morphologyEx1', morphologyEx)
    # cv2.waitKey(0)

    # 膨胀腐蚀
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    Element = cv2.dilate(morphologyEx, kernelX)
    Element = cv2.dilate(Element, kernelY,iterations=2)
    Element = cv2.erode(Element, kernelX)
    Element = cv2.erode(Element, kernelY)

    # cv2.imshow('getStructuringElement', Element)
    # cv2.waitKey(0)

    _, contours, hair = cv2.findContours(Element, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def box_get_front_correction1(img,addX,addY ,imgHeight,imgWidth,faceLeft,name_y,sex_y,birth_year_y
                              ,name_addHeight,birth_year_addHeight):
    '''
    根据地址的坐标 定位其他信息的位置
    :param img: 输入图片
    :param addX: 地址的x坐标
    :param addY: 地址的y坐标
    :param imgHeight: 图片高度
    :param imgWidth: 图片宽度
    :param faceLeft: 人脸左边坐标
    :return: 文字的位置
    '''
    regions = []

    # 地址
    address_addHeight = int(addY + imgHeight / 2.15)
    address_addWidth = faceLeft
    address = img[addY:address_addHeight, addX:address_addWidth]
    img = cv2.rectangle(img, (addX, addY), (address_addWidth, address_addHeight), (0, 0, 255), 2)

    # 名字
    name_x = addX
    name_y = name_y # int(addY - imgHeight/1.65)
    # name_addHeight = int(name_y + 46)
    name_addHeight = int(name_addHeight + name_y)
    name_addWidth = int(name_x + imgWidth / 5)
    name = img[name_y:name_addHeight, name_x:name_addWidth]
    img = cv2.rectangle(img, (name_x, name_y), (name_addWidth, name_addHeight), (0, 0, 255), 2)

    # 民族
    nationality_x = addX + 150
    nationality_y = sex_y
    nationality_addHeight = int(nationality_y + imgHeight / 7.65)
    nationality_addWidth = int(nationality_x + 50)
    nationality = img[nationality_y:nationality_addHeight, nationality_x:nationality_addWidth]
    img = cv2.rectangle(img, (nationality_x, nationality_y), (nationality_addWidth, nationality_addHeight), (0, 0, 255),
                        2)

    # 性别
    sex_x= addX
    sex_y= sex_y # int(addY - imgHeight/2.65)
    sex_addHeight = int(sex_y + imgHeight / 7.71)
    sex_addWidth = int(sex_x + 45)
    sex = img[sex_y:sex_addHeight, sex_x:sex_addWidth]
    img = cv2.rectangle(img, (sex_x, sex_y), (sex_addWidth, sex_addHeight), (0, 0, 255), 2)

    birth_year_x = addX
    # int(addY - imgHeight/4.6)
    birth_year_y = birth_year_y
    # birth_year_addHeight = int(birth_year_y + imgHeight / 7.71)
    birth_year_addHeight = int(birth_year_addHeight + birth_year_y)
    birth_year_addWidth = int(birth_year_x + 85)
    birth_year = img[birth_year_y:birth_year_addHeight, birth_year_x:birth_year_addWidth]
    img = cv2.rectangle(img, (birth_year_x, birth_year_y), (birth_year_addWidth, birth_year_addHeight), (0, 0, 255), 2)

    #出生 月
    birth_month_x = birth_year_x + 123
    birth_month_y = birth_year_y
    # birth_month_addHeight = int(birth_month_y + imgHeight / 7.6)
    birth_month_addHeight = birth_year_addHeight
    birth_month_addWidth = int(birth_month_x + 40)
    birth_month = img[birth_month_y:birth_month_addHeight, birth_month_x:birth_month_addWidth]
    img = cv2.rectangle(img, (birth_month_x, birth_month_y), (birth_month_addWidth, birth_month_addHeight), (0, 0, 255),
                        2)

    #出生 日
    birth_day_x = birth_year_x + 190
    birth_day_y = birth_year_y
    # birth_day_addHeight = int(birth_day_y + imgHeight / 7.6)
    birth_day_addHeight = birth_year_addHeight
    birth_day_addWidth = int(birth_day_x + 43)
    birth_day = img[birth_day_y:birth_day_addHeight, birth_day_x:birth_day_addWidth]
    img = cv2.rectangle(img, (birth_day_x, birth_day_y), (birth_day_addWidth, birth_day_addHeight), (0, 0, 255), 2)


def box_get_front_correction2(img,addX,addY ,imgHeight,imgWidth,faceLeft,name_y,sex_y,birth_year_y,
                              name_addHeight,birth_year_addHeight):
    '''
    根据地址的坐标 定位其他信息的位置
    :param img: 输入图片
    :param addX: 地址的x坐标
    :param addY: 地址的y坐标
    :param imgHeight: 图片高度
    :param imgWidth: 图片宽度
    :param faceLeft: 人脸左边坐标
    :return: 文字的位置
    '''
    regions = []

    # 地址
    address_addHeight = int(addY + imgHeight / 2.3)
    address_addWidth = faceLeft
    address = img[addY:address_addHeight, addX:address_addWidth]
    img = cv2.rectangle(img, (addX, addY), (address_addWidth, address_addHeight), (0, 0, 255), 2)

    # 名字
    name_x = addX
    name_y = name_y # int(addY - imgHeight/1.65)
    # name_addHeight = int(name_y + 46) + 10

    name_addHeight = int(name_addHeight + name_y) + 10
    name_addWidth = int(name_x + imgWidth / 4)
    name = img[name_y:name_addHeight, name_x:name_addWidth]
    img = cv2.rectangle(img, (name_x, name_y), (name_addWidth, name_addHeight), (0, 0, 255), 2)

    # 性别
    sex_x= addX
    sex_y= sex_y # int(addY - imgHeight/2.65)
    sex_addHeight = int(sex_y + imgHeight / 7.71)
    sex_addWidth = int(sex_x + 45)
    sex = img[sex_y:sex_addHeight, sex_x:sex_addWidth]
    img = cv2.rectangle(img, (sex_x, sex_y), (sex_addWidth, sex_addHeight), (0, 0, 255), 2)

    # 出生 年
    birth_year_x = addX
    # int(addY - imgHeight/4.6)
    birth_year_y = birth_year_y
    # birth_year_addHeight = int(birth_year_y + imgHeight / 7.71)
    birth_year_addHeight = int(birth_year_addHeight + birth_year_y)
    birth_year_addWidth = int(birth_year_x + 85)
    birth_year = img[birth_year_y:birth_year_addHeight, birth_year_x:birth_year_addWidth]
    img = cv2.rectangle(img, (birth_year_x, birth_year_y), (birth_year_addWidth, birth_year_addHeight), (0, 0, 255), 2)

#对数组进行排序
def findSmallest(arr):
    # 将第一个元素的值作为最小值赋给smallest
    smallest = arr[0]
    smallest_index = 0  # 将第一个值的索引作为最小值的索引赋给smallest_index
    for i in range(1, len(arr)):
        if arr[i] < smallest:  # 对列表arr中的元素进行一一对比
            smallest = arr[i]
            smallest_index = i
    return smallest_index


def selectionSort(arr):
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)  # 一共要调用5次findSmallest
        newArr.append(arr.pop(smallest))  # 每一次都把findSmallest里面的最小值删除并存放在新的数组newArr中
    return newArr


# 投影
def projection(part_1,z2):
    '''

    :param part_1: 输入图像
    :param z2: y坐标
    :return: 框出图像
    '''
    pro_Count = list()
    result = cv2.bilateralFilter(part_1.copy(), 0, 50, 5)
    gray_part_1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    height, width = part_1.shape[:2]
    thresh = cv2.adaptiveThreshold(gray_part_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 5)

    # (_, thresh) = cv2.threshold(gray_part_1, 140, 255, cv2.THRESH_BINARY)
    # cv_show('thresh', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 形态学处理，定义矩形结构
    closed = cv2.erode(thresh, kernel, iterations=5)
    # cv_show('closed', closed)
    height, width = closed.shape[:2]
    # print(height, width)
    v = [0] * width
    z = [0] * height
    hfg = [[0 for col in range(2)] for row in range(height)]
    lfg = [[0 for col in range(2)] for row in range(width)]
    a = 0
    # 垂直投影
    # 统计并存储每一列的黑点数
    for x in range(0, width):
        for y in range(0, height):
            if closed[y][x] == 0:
                a = a + 1
            else:
                continue
        v[x] = a
        a = 0
    l = len(v)
    # print l
    # print width
    # 创建空白图片，绘制垂直投影图
    emptyImage = np.zeros((height, width, 3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for x in range(0, width):
        for y in range(0, v[x]):
            b = (255, 255, 255)
            emptyImage[y, x] = b
    # cv2.imshow('垂直', emptyImage)

    for p in range(0, 1):
        incol = 1
        start1 = 0
        j1 = 0
        z1 = hfg[p][0]
        # z2 = hfg[p][1]
        for i1 in range(0, width):
            if incol == 1 and v[i1] >= 20:  # 从空白区进入文字区
                start1 = i1  # 记录起始列分割点
                incol = 0
            elif (i1 - start1 > 3) and v[i1] < 20 and incol == 0:  # 从文字区进入空白区
                incol = 1
                lfg[j1][0] = start1 - 2  # 保存列分割位置
                lfg[j1][1] = i1 + 2
                l1 = start1 - 2
                l2 = i1 + 2
                j1 = j1 + 1
                cv2.rectangle(part_1, (l1, z1), (l2, z2), (255, 0, 0), 2)
                pro_Count.append(((l1, z1), (l2, z2)))
    # print(pro_Count)

    # cv_show('part_1', part_1)
    return pro_Count


# 调节图像的曝光度
def gamma(img):
    '''
    :param img:纠偏后的图像
    :return: 曝光处理过后的图像
    '''
    value_of_gamma = cv2.getTrackbarPos('Value of Gamma', 'demo')  # gamma取值
    value_of_gamma = value_of_gamma + 350.0
    value_of_gamma = value_of_gamma * 0.01  # 压缩gamma范围，以进行精细调整
    image_gamma_correct = gamma_trans(img, value_of_gamma)  # 2.5为gamma函数的指数值，大于1曝光度下降，大于0小于1曝光度增强
    # cv2.imshow("demo", image_gamma_correct)
    # cv2.waitKey(0)
    return image_gamma_correct


# 信息定位
def box_get_front_message(image, save_name, orig_img):
    '''
    :param image:纠偏完成的图
    :param save_name: 保存途径
    :param orig_img: 原图
    :return: 信息定位结果
    '''
    classfier = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_alt2.xml")
    image_copy = image.copy()
    detector = dlib.get_frontal_face_detector()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_cv = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    dets = detector(image, 2)  # 使用detector进行人脸检测 dets为返回的结果

    if len(dets) == 0:
        left = faces_cv[0][0]
        top = faces_cv[0][1]
        right = faces_cv[0][0]+faces_cv[0][2]
        bottom = faces_cv[0][1]+faces_cv[0][3]
        Wwidth = right - left
        Hheigt = top - bottom
    else:
        left = dets[0].left()
        top = dets[0].top()
        right = dets[0].right()
        bottom = dets[0].bottom()
        Wwidth=right - left
        Hheigt=top - bottom

    # 照片的位置（不怎么精确）
    width = right - left
    high = top - bottom
    left2 = np.uint(left - 0.3*width)
    bottom2 = np.uint(bottom + 0.6*width)

    img = cv2.rectangle(image, (left2,bottom2-20),(left2+2*width-50,bottom2+2*high),(0,0,255),2)

    top2 = np.uint(bottom2 + 1.8 * high)
    right2 = np.uint(left2 + 1.6 * width)
    face_img = image[top2:bottom2, left2:right2, :]

    face_gray = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
    ret = cv2.threshold(face_gray,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)[1]
    # cv_show('ret',ret)
    erode = cv2.erode(ret,(9,9),iterations=3)
    dilate = cv2.dilate(erode,(9,9),iterations=3)

    '''
    找身份证号码的位置
    '''
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
    sqlKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))

    tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,sqlKernel)

    gradX = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
    gradX = np.absolute(gradX)
    (minVal,maxVal) = (np.min(gradX),np.max(gradX))
    gradX = (255*((gradX-minVal)/(maxVal-minVal)))
    gradX = gradX.astype('uint8')

    gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,sqlKernel)

    thresh=cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,rectKernel,iterations=1)

    threshCnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts=threshCnts

    cur_img=img.copy()
    cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
    # cv_show('cur_img',cur_img)

    locs=[]
    # 遍历轮廓
    for (i, c) in enumerate(cnts):
        (x, y, w, h)=cv2.boundingRect(c)
        ar = w / float(h)
        # 选择合适的区域，根据实际任务来，这里的基本上都是四个数字一组
        if ar > 5 and ar < 40:
            if w > 350 and w < 600:
                # 把符合的留下来
                locs.append((x, y, w, h))

    locs=sorted(locs,key=lambda x: x[0])
    # print('locs', locs)
    # output=[]
    #
    if len(locs) != 0:
        for (i,(gX,gY,gW,gH)) in enumerate(locs):
            # initialize the list of group digits
            cv2.rectangle(img,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),2)
        Xwidth = gW + 10
        Yheight = gH + 10
        long = 2 * Xwidth
    else:
        Xwidth = 250
        long = 2 * Xwidth
        gH = 25
    result_img=img.copy()
    '''
     框出信息区域
    '''
    X1 = 2*width
    Y1 = 2*high
    left3 = left2-(long-X1)

    if left3 > 0:
        left3 = left3
        top2 = bottom2 + 2 * high - 100
        point1 = (left3, bottom2)
        point2 = (left2, top2)
    else:
        left3 = 50
        top2 = bottom2 + 2 * high - 100
        point1 = (left3, bottom2)
        point2 = (left2, top2)

    if top2 < 0:
        top2 = 20
    img_copy = img.copy()
    cv2.rectangle(img, point1, point2, (0, 0, 255), 2)

    part = image_copy[top2:bottom2,left3:left2+15]
    # cv_show('part',part)

    retCnts = getpart_info(part)

    retCnts = sorted(retCnts, key=cv2.contourArea, reverse=True)
    h = retCnts[0]
    i = 0
    (x, y, w, h) = cv2.boundingRect(h)
    x = x + left3
    y = y + top2
    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 255), 2)

    cut = x - 5
    part_in=image_copy[top2:y+100,cut:left2+15]
    # cv_show('part_in',part_in)

    X = x-5
    Y = y-5
    conCnts = getpart_info(part_in)
    countY = []
    keyCount = list()

    for c in conCnts:
        (x, y, w, h) = cv2. boundingRect(c)
        x = x + cut
        y = y + top2
        if abs(x-cut) < 30 and w > 20 and y > 20:
            countY.append(y)
            # 由5加到了10
            cv2.rectangle(img_copy, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
            w = w + 8
            keyCount.append((y, h, w))
    # print('len(keyCount)', len(keyCount))
    if len(keyCount) < 5:
        # 曝光度处理
        image_copy = gamma(image_copy)

        part = image_copy[top2:bottom2, left3:left2 + 15]
        # cv_show('part', part)

        retCnts = getpart_info(part)

        retCnts = sorted(retCnts, key=cv2.contourArea, reverse=True)
        h = retCnts[0]
        i = 0
        (x, y, w, h) = cv2.boundingRect(h)
        x = x + left3
        y = y + top2
        cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 255), 2)

        cut = x - 5
        part_in = image_copy[top2:y + 100, cut:left2 + 15]
        # cv_show('part_in', part_in)

        X = x - 5
        Y = y - 5
        conCnts = getpart_info(part_in)
        countY = []
        keyCount = list()

        for c in conCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            x = x + cut
            y = y + top2
            if abs(x - cut) < 30 and w > 20 and y > 20:
                countY.append(y)
                # 由5加到了10
                cv2.rectangle(img_copy, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
                w = w + 8
                keyCount.append((y, h, w))
                # cv_show('img_copy', img_copy)
                countY = selectionSort(countY)
                keyCount = selectionSort(keyCount)
                # print('keycount = ', keyCount)
    else:
        # cv_show('img_copy',img_copy)
        countY = selectionSort(countY)
        keyCount = selectionSort(keyCount)
        # print('keycount = ',keyCount)

    try:
        # 投影
        part_1 = image[keyCount[1][0]:keyCount[1][0] + keyCount[1][1], X + keyCount[2][2] + 7:left2]
        # cv_show('part_1', part_1)
        pro_count_1 = projection(part_1, keyCount[1][0] + keyCount[1][1])
        len_count = len(pro_count_1)
        # print(len_count)
        if len_count == 4:
            (x1, y1), (x2, y2) = pro_count_1[3]
            cv2.rectangle(result_img, (x1 + X + keyCount[2][2] + 7, y1 + keyCount[1][0]), (x2 + X + keyCount[2][2] + 7, y2),
                      (0, 0, 255), 2)
        else:
            (x1, y1), (x2, y2) = pro_count_1[2]
            cv2.rectangle(result_img, (x1 + X + keyCount[2][2] + 7, y1 + keyCount[1][0]), (x2 + X + keyCount[2][2] + 7, y2),
                 (0, 0, 255), 2)
        # cv_show('result_img',result_img)

        # cv_show('res', res)

        part_2 = image[keyCount[2][0]:keyCount[2][0] + keyCount[2][1], X + keyCount[2][2] + 10:left2]
        # cv_show('part_2', part_2)
        pro_count_2 = projection(part_2, keyCount[2][0] + keyCount[2][1])
        (x1, y1), (x2, y2) = pro_count_2[0]

        num_1 = []
        num_2 = []
        temp = len(pro_count_2)
        for i in range(temp):
            (x1, y1), (x2, y2) = pro_count_2[i]
            num_1.append(x1)
            num_2.append(x2)

        # print('num = ', num_1)
        # print('num = ', num_2)
        # 年的x坐标
        year_x1 = num_1[0]
        year_x2 = num_2[0]
        # 根据年的坐标估算月的坐标
        # print(year_x1)
        month_x = year_x1 + 70
        month = []

        for i in range(len(num_1)):
            if abs(num_1[i] - month_x) <= 20:
                month.append(abs(num_1[i] - month_x))
                temp = i
                if i > len(num_1):
                    break

        month_x = min(month)+month_x
        # print('month_x',month_x)
        # print('num_2[temp]',num_2[temp])
        cv2.rectangle(result_img, (year_x2 +(X + keyCount[2][2] + 7), y1 + keyCount[2][0]), (month_x+(X + keyCount[2][2] + 7) , y2), (0, 0, 255), 2)
        cv2.rectangle(result_img, (num_2[temp] + (X + keyCount[2][2] + 7) + 5, y1 + keyCount[2][0]),
                          (num_2[temp] + (X + keyCount[2][2] + 7) + 50, y2), (0, 0, 255), 2)

        temp = 1
        # print(keyCount[0][1],keyCount[2][1])
        for i in range(1, len(countY)):
            temp += 1
            # print(temp)
        if temp == 4 or temp == 5:
            box_get_front_correction2(result_img, X, Y, 8 * gH, long, left2, keyCount[0][0] , keyCount[1][0]
                                          , keyCount[2][0],keyCount[0][1],keyCount[2][1])
        elif temp > 6:
            for i in temp:
                if keyCount[i + 1][0] - (keyCount[i][0] + keyCount[i][1]) < 5:
                    keyCount[i + 1][0] = keyCount[i + 2][0]
            box_get_front_correction2(result_img, X, Y, 8 * gH, long, left2, keyCount[1][0], keyCount[2][0]
                                          , keyCount[3][0],keyCount[0][1],keyCount[2][1])
        else:
            box_get_front_correction2(result_img, X, Y, 8 * gH, long, left2, keyCount[0][0], keyCount[1][0]
                                          , keyCount[1][0] + 60,keyCount[0][1],keyCount[2][1])

        result_img = np.hstack((orig_img, result_img))
        # cv_show('result_img', result_img)
        cv2.imencode('.jpg', result_img)[1].tofile(str(save_name))
    except Exception as e:
        temp = 1
        for i in range(1,len(countY)):
            temp += 1
        if temp == 4 or temp == 5:
            box_get_front_correction1(result_img, X , Y ,8 * gH, long, left2,countY[0] ,countY[1]
                                          ,countY[2],keyCount[0][1],keyCount[2][1])
        elif temp == 6:
            box_get_front_correction1(result_img, X, Y, 8 * gH, long, left2, countY[1], countY[2]
                                         ,countY[3],keyCount[1][1],keyCount[2][1])
        else:
            box_get_front_correction1(result_img, X , Y ,8 * gH, long, left2,countY[0],countY[1]
                                         ,countY[1]+60,keyCount[0][1],keyCount[2][1])
        result_img = np.hstack((orig_img, result_img))
        # cv_show('result_img',result_img)
        cv2.imencode('.jpg', result_img)[1].tofile(str(save_name))


