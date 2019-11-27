import dlib
from src.back_correct_skew import back_correct_skew
from src.config.config import *
from src.com.tools import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib
from PIL import Image
import copy


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  ##定义


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

    # interpo;ation为插值方法，这里选用的是
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # 0,1,2,3分别对应左上，右上，右下和左下坐标
    # 计算左上和右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算左下和右上
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 根据四个关键点进行透视变换
def four_point_transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值（图形可能是个多边形），多边形取最大的差值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # 取W1与W2中最大的
    maxWidth = max(int(widthA), int(widthB))

    # 同理得到
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置，dst为变换后的点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        # 作-1更精确
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵M
    # 由二维变成三维，然后再转换为二维，转换成矩形，opencv会自动将坐标点进行计算，得出变换矩阵M
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    # 返回变换后结果
    return warped


def perspective_transformation(points, src):
    """
        传入四个点进行透视变换
        :param flag: fitline: 直线检测拟合结果 else：pre预估结果
        :param points: 包含四个点的点集
        :return: 两直线交点
        """
    p1, p2, p3, p4 = points  # p1-右下  p2-左下  p3-右上  p4-左上
    # 原图中书本的四个角点 左上、右上、左下、右下
    pts1 = np.float32([p1, p2, p3, p4])
    if p3[1] - p1[1] < p2[0] - p1[0]:
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

    return dst


# 基于两直线的交点计算
def find_cross_point(line1, line2):
    """
    计算两直线交点
    :param line1: 直线一两端点坐标
    :param line2: 直线二两端点坐标
    :return: 交点坐标
    """
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if x2 == x1:
        return False

    # 计算k1,由于点均为整数，需要进行浮点数转化
    k1 = (y2 - y1) * 1.0 / (x2 - x1)

    # 整型转浮点型是关键
    b1 = y1 * 1.0 - x1 * k1 * 1.0

    # L2直线斜率不存在操作
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        # 斜率存在操作
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        if k1 == k2:
            return False
        # k1X+b1 = k2X+b2(令y1=y2求交点)
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return int(x), int(y)


def shape_to_np(shape, dtype="int"):
    # 创建68*2
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 遍历每一个关键点
    # 得到坐标
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# 旋转+人脸检测
def face_detect_rotation(img):
    """
    :param img: 旋转过后的原图
    :return: 人脸的数据点；灰度图；人脸检测后的图像；图像的长宽比
    """
    # 加载人脸检测
    detector = dlib.get_frontal_face_detector()
    # 加载5个关键点的定位
    predictor = dlib.shape_predictor('../data/shape_predictor_5_face_landmarks.dat')

    orig = img.copy()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测，检测出来人脸框,2表示采样次数，得到人脸检测的大框Z
    rects = detector(img1, 2)

    # 自定义旋转
    # 通过人脸检测来进行图像的旋转处理，从而进行人脸检测
    # 对人脸框进行关键点的定位，人脸关键点相对于框的位置
    shape = predictor(img1, rects[0])
    # 转换成ndarray格式，转换成坐标值
    shape = shape_to_np(shape)
    ratio = img.shape[1] / 500.0

    img2 = copy.deepcopy(img)
    img4 = copy.deepcopy(img)
    img5 = copy.deepcopy(img)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(img1, 2)
    plt.figure()
    ax = plt.subplot(111)
    ax.imshow(img4)

    # 5点法检测人脸，左眼、右眼和唇上口#检测眼睛个唇上口的位置
    # 'ro'表示red，'o'类型的线条
    plt.scatter(shape[0:5, 0], shape[0:5, 1], color='r', marker='o', s=5)
    for i in np.arange(5):
        plt.text(shape[i, 0] - 8, shape[i, 1] - 8, i)
    # plt.show()
    print('人脸检测完毕')

    # 读取图片，并且做初始化旋转操作#读取图片+人脸识别
    shape = predictor(img1, rects[0])
    shape = shape_to_np(shape)
    x1 = shape[2][0]
    y1 = shape[2][1]
    x2 = shape[0][0]
    y2 = shape[0][1]
    return shape, img1, img2, img4, img5, ratio


# 计算图片旋转角度
def cal_rotation_angle(img):
    """
    计算旋转角度
    :param img: 图片
    :return:旋转角度
    """
    from src.back_correct_skew import mark_corner_image
    img_mark = mark_corner_image(img, point_size = 3)
    # 投影计算角度，对图片进行纠正
    from src.com.tools import project

    project_image = np.zeros((180, img_mark.shape[0]))

    img_mark_im = Image.fromarray(img_mark)
    for i in range(180):

        im_rotate = img_mark_im.rotate(i)
        project_image[i] = project(im_rotate, orientation=0)

    max_index = np.where(project_image[:, :] == project_image.max())
    rotate_angle = max_index[0][0]

    return rotate_angle


# 人脸检测（判断正反面）
def face_detect(img):
    """
    :param img:原图
    :return: 纠偏后的图像
    """
    # 加载人脸检测
    detector = dlib.get_frontal_face_detector()
    classfier = cv2.CascadeClassifier("..data//haarcascades/haarcascade_frontalface_alt2.xml")

    from src.back_correct_skew import cal_rotation_angle
    angle = cal_rotation_angle(img.copy())

    # 将填充颜色改为（100,100,100）
    img1 = Image.fromarray(np.array(img))
    im2 = img1.copy().convert('RGBA')
    rot = im2.rotate(angle, expand=True)
    fff = Image.new('RGBA', rot.size, (100,) * 4)
    out = Image.composite(rot, fff, rot)
    img_rotation = out.convert(img1.mode)

    img_rotation = Image.fromarray(np.array(img_rotation))

    # 使用dlib人脸检测模型
    faces_dlib = detector(np.array(img_rotation), 1)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(np.array(img_rotation), cv2.COLOR_BGR2GRAY)
    else:
        gray = np.array(img_rotation)
    faces_cv = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    is_front = 1
    if len(faces_dlib) == 0 or len(faces_cv) == 0:
        img_rotation = img_rotation.rotate(180, expand=True)
        faces_dlib = detector(np.array(img_rotation), 1)

        if len(img.shape) == 3:
            grey = cv2.cvtColor(np.array(img_rotation), cv2.COLOR_BGR2GRAY)
        else:
            grey = np.array(img_rotation)

        # 使用opencv人脸检测模型
        faces_cv = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faces_dlib) == 0 or len(faces_cv) == 0:
            is_front = 0
    # plt.show()
    if is_front:
        return np.array(img_rotation)
    else:
        return img


def predict_rect(gray, scr, shape):
    """
    :param gray: 灰度图
    :param scr: 原图
    :param shape: 人脸检测返回的数据点
    :return: 矩形的左上角坐标，矩形的长和宽
    """
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqlKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype('uint8')

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel, iterations=2)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqlKernel)
    threshCnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = threshCnts

    cur_img = scr.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 2)

    locs = []

    # 遍历轮廓
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        mid = (shape[1][1] + shape[3][1]) / 2.0

        # 选择合适的区域，根据实际任务来，这里的基本上都是四个数字一组
        if ar > 11 and ar < 40 and y >= shape[4][1]+1.5*(shape[4][1]-mid) and w >= 120:

            # 把符合的留下来
            locs.append((x, y, w, h))

            gX = locs[0][0]
            gY = locs[0][1]
            gW = locs[0][2]
            gH = locs[0][3]
            cv2.rectangle(scr, (gX - 3, gY - 3), (gX + gW + 3, gY + gH + 3), (0, 0, 255), 2)
            detect_id = []
            detect_id.append([gX - 3, gY - 3, gX + gW + 3, gY + gH + 3])

    #预估grabcut的rect值
    if len(locs) == 0:
        leng = abs(11.5 * (shape[0][0] - shape[2][0]))
    else:
        leng = abs(1.92 * (detect_id[0][2] - detect_id[0][0]))
    mid = (shape[1][1] + shape[3][1]) / 2.0
    wid = abs(14 * (mid - shape[4][1]))

    rt_x = shape[4][0] * 1.0 + (leng / 4.0)
    rt_y = shape[4][1] * 1.0 - (wid / 2.0)
    rb_x = rt_x
    rb_y = rt_y + wid
    lt_x = rt_x - leng
    lt_y = rt_y
    lb_x = rt_x - leng
    lb_y = rt_y + wid  # 计算出rect的四个点
    return lt_x, lt_y, leng, wid


# 寻找最大轮廓
def find_max_contour(after_grabcut, img):
    """
    :param after_grabcut: grabcut处理之后的图像
    :param img: 原图
    :return: 最大轮廓图
    """
    cnts = cv2.findContours(after_grabcut, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    # 轮廓排序
    # 获取面积最大的一个轮廓
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    img2 = img.copy()
    img2[:, :] = 0
    # 获取并绘制轮廓
    cv2.drawContours(img2, cnts, -1, (255, 255, 255), 1)
    return img2


def line_detect(after_contours, scr):
    """
    :param after_contours: 轮廓
    :param scr: 原图
    :return: 对轮廓进行直线检测后的直线集
    """
    # 对轮廓检测后的直线进行直线检测
    minLineLength = 140
    maxLineGap = 7
    lines = cv2.HoughLinesP(after_contours, 1, np.pi / 360, 10, minLineLength, maxLineGap)
    l1 = lines[:, 0, :]

    # lines[:,0,:]将直线压缩成二维图像，找出两个端点(降低维度)
    for x1, y1, x2, y2 in l1:
        cv2.line(scr, (x1, y1), (x2, y2), (0, 255, 0), 1)
    plt.imshow(scr, cmap=plt.gray())
    return l1


# 限制直线检测后的直线数量
def line_area(l1, lt_x, rt_x, lt_y, lb_y):
    """
    :param l1:轮廓检测后的直线
    :param lt_x:直线左上端点横坐标
    :param rt_x:直线右上端点横坐标
    :param lt_y:直线左上端点纵坐标
    :param lb_y:直线左下端点纵坐标
    :return:水平直线和垂直直线
    """
    levelline = []
    vertline = []

    for i in range(0, len(l1)):
        if l1[i][1] == l1[i][3] or abs(l1[i][1] - l1[i][3]) <= 3:
            levelline.append(l1[i])
        if l1[i][0] == l1[i][2] or abs(l1[i][0] - l1[i][2]) <= 3:
            vertline.append(l1[i])
        i += 1
        if i > len(l1) - 1:
            break

    # 水平线和接近水平的线
    # 垂直线和接近垂直的线
    # 将水平与垂直线分割出来

    # 水平线限制区
    l_limit = []
    for i in range(0, len(levelline)):
        if levelline[i][1] >= lt_y + 6 and levelline[i][1] <= lb_y + 12 \
                and levelline[i][0] >= lt_x - 7 and levelline[i][0] <= rt_x + 6:
            l_limit.append(levelline[i])
        i += 1
        if i > len(levelline) - 1:
            break

    # 垂直线限制区
    v_limit = []
    for j in range(0, len(vertline)):
        if vertline[j][1] >= lt_y + 6 and vertline[j][1] <= lb_y + 12 \
                and vertline[j][0] >= lt_x - 7 and vertline[j][0] <= rt_x + 6:
            v_limit.append(vertline[j])
        j += 1
        if j > len(vertline) - 1:
            break
    return l_limit, v_limit


# 划分上下左右直线
def line_group(l_limit, v_limit):
    """
    :param l_limit: 水平直线
    :param v_limit: 垂直直线
    :return: 上、下、左、右直线集
    """
    # 上部的水平线
    # 下部的水平线
    top_level = []
    bottom_level = []
    for n in range(0, len(l_limit) - 1):
        if abs(l_limit[n][1] - l_limit[n + 1][1]) >= 100:
            if l_limit[n][1] > l_limit[n + 1][1]:
                top_level.append(l_limit[n + 1])
                bottom_level.append(l_limit[n])
            if l_limit[n][1] < l_limit[n + 1][1]:
                top_level.append(l_limit[n])
                bottom_level.append(l_limit[n + 1])
        n += 1
        if n > len(l_limit) - 1:
            break

    # 左侧的垂直线
    # 右侧的垂直线
    left_vert = []
    right_vert = []
    for m in range(0, len(v_limit) - 1):
        if abs(v_limit[m][0] - v_limit[m + 1][0]) >= 130:
            if v_limit[m][0] > v_limit[m + 1][0]:
                left_vert.append(v_limit[m + 1])
                right_vert.append(v_limit[m])
            if v_limit[m][0] < v_limit[m + 1][0]:
                left_vert.append(v_limit[m])
                right_vert.append(v_limit[m + 1])
        m += 1
        if m > len(v_limit) - 1:
            break

    return top_level, bottom_level, left_vert, right_vert


# 进行直线拟合
def fitline(lines):
    """
    :param lines:直线集合
    :return:最终拟合直线的斜率和截距
    """
    loc = []
    for line in lines:
        x1, y1, x2, y2 = line
        loc.append([x1, y1])
        loc.append([x2, y2])

    loc = np.array(loc)
    output = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)
    if output[1] != 1:
        k = (output[1] / output[0])
        b = (output[3] - k*output[2])
    else:
        k = 57.29
        b = (output[3] - 57.29*output[2])
    return k, b


def classify_lines(l_limit, v_limit):
    """
    :param l_limit:水平直线集
    :param v_limit: 垂直直线集
    :return: 上、下、左、右直线
    """
    top_level, bottom_level, left_vert, right_vert = line_group(l_limit, v_limit)

    # 上一部分水平线拟合而成的斜率和b值
    topline = fitline(top_level)

    # 下一部分水平线拟合而成的斜率和b值
    bottomline = fitline(bottom_level)

    # 左侧垂直线的拟合而成的斜率和b值
    leftline = fitline(left_vert)

    # 右侧垂直线的拟合而成的斜率和b值
    rightline = fitline(right_vert)
    return topline, bottomline, leftline, rightline


def distance(lines1, lines2):
    """
    :param lines1: 直线1
    :param lines2: 直线2
    :return: 两直线斜率k之间的距离，两直线截距之间的距离
    """
    k_distance = 0.5*(abs(lines1[0][0]-lines2[0][0])+abs(lines1[1][0]-lines2[1][0]))
    b_distance = 0.5*(abs(lines1[2][1]-lines2[2][1])+abs(lines1[3][1]-lines2[3][1]))
    return k_distance, b_distance


# 在四个方案直线上选择最佳的
def find_line_in_four(lines1, lines2, lines3, lines4):
    '''
    由直线的k,b值的接近程度来筛选最佳的直线
    lines1-lines4表示4条需要比较的直线集
    tan(3°)=0.05
    |b1 - b2| <= 30
    distance_ab[0]: k_distance
    distance_ab[1]: b_distance
    直线的优先级是lines2 > lines1 > lines4 >lines3
    只比较
    '''

    distance_12 = distance(lines1, lines2)
    distance_13 = distance(lines1, lines3)
    distance_14 = distance(lines1, lines4)
    distance_23 = distance(lines2, lines3)
    distance_24 = distance(lines2, lines4)
    distance_34 = distance(lines3, lines4)
    if distance_12 == min(distance_12,distance_13,distance_14,distance_23,distance_24,distance_34):
        return lines1
    elif distance_13 == min(distance_12,distance_13,distance_14,distance_23,distance_24,distance_34):
        return lines1
    elif distance_14 == min(distance_12,distance_13,distance_14,distance_23,distance_24,distance_34):
        return lines1
    elif distance_23 == min(distance_12,distance_13,distance_14,distance_23,distance_24,distance_34):
        return lines2
    elif distance_24 == min(distance_12, distance_13, distance_14, distance_23, distance_24, distance_34):
        return lines2
    else:
        return lines3


# 在三个方案直线上选择最佳(lines1->lines3优先级由大到小)
def find_line_in_three(lines1,lines2,lines3):
    """
    :param lines1: 直线1
    :param lines2: 直线2
    :param lines3: 直线3
    :return: 最优直线
    """
    distance_12 = distance(lines1, lines2)
    distance_13 = distance(lines1, lines3)
    distance_23 = distance(lines2, lines3)
    if distance_12 == min(distance_12, distance_13, distance_23):
        return lines1
    if distance_13 == min(distance_12, distance_13, distance_23):
        return lines1
    else:
        return lines2


# 在两个方案上选择最佳
def find_line_in_two(lines1, lines2):
    return lines1


# lines1为二值化，lines2为canny，lines3为grabcut1，lines4为grabcut
def select_best_lines(lines1, lines2, lines3, lines4):
    """
    :param lines1: 直线1
    :param lines2: 直线2
    :param lines3: 直线3
    :param lines4: 直线4
    :return: 最佳直线集
    """
    best_lines = []

    # lines3不存在
    if len(lines1) == 4 and len(lines2) == 4 and len(lines3) == 4 and len(lines4):
        line = find_line_in_four(lines1, lines2, lines3, lines4)
        best_lines.append(line)

    # lines1不存在
    if len(lines1) != 4 and len(lines2) == 4 and len(lines3) == 4 and len(lines4) == 4:
        line = find_line_in_three(lines2, lines3, lines4)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) == 4 and len(lines4) == 4:
        line = find_line_in_two(lines3,lines4)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) == 4:
        line = lines4
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) == 4 and len(lines4) != 4:
        line = lines3
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) == 4 and len(lines3) != 4 and len(lines4) == 4:
        line = find_line_in_two(lines2, lines4)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) == 4:
        line = lines4
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) == 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines2
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) == 4 and len(lines3) == 4 and len(lines4) != 4:
        line = find_line_in_two(lines2, lines3)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) == 4 and len(lines4) != 4:
        line = lines3
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) == 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines2
        best_lines.append(line)

    # lines2不存在
    if len(lines2) != 4 and len(lines1) == 4 and len(lines3) == 4 and len(lines4) == 4:
        line = find_line_in_three(lines1, lines3, lines4)
        best_lines.append(line)
    if len(lines2) != 4 and len(lines1) != 4 and len(lines3) == 4 and len(lines4) == 4:
        line = find_line_in_two(lines3,lines4)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) == 4:
        line = lines4
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) == 4 and len(lines4) != 4:
        line = lines3
        best_lines.append(line)
    if len(lines2) != 4 and len(lines1) == 4 and len(lines3) != 4 and len(lines4) == 4:
        line = find_line_in_two(lines1, lines4)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) == 4:
        line = lines4
        best_lines.append(line)
    if len(lines1) == 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines1
        best_lines.append(line)
    if len(lines1) == 4 and len(lines2) != 4 and len(lines3) == 4 and len(lines4) != 4:
        line = find_line_in_two(lines1, lines3)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) == 4 and len(lines4) != 4:
        line = lines3
        best_lines.append(line)
    if len(lines1) == 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines1
        best_lines.append(line)

    # lines3不存在
    if len(lines3) != 4 and len(lines1) == 4 and len(lines2) == 4 and len(lines4) == 4:
        line = find_line_in_three(lines1, lines2, lines4)
        best_lines.append(line)
    if len(lines3) != 4 and len(lines1) != 4 and len(lines2) == 4 and len(lines4) == 4:
        line = find_line_in_two(lines2,lines4)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) == 4:
        line = lines4
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) == 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines2
        best_lines.append(line)
    if len(lines2) != 4 and len(lines1) == 4 and len(lines3) != 4 and len(lines4) == 4:
        line = find_line_in_two(lines1, lines4)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) == 4:
        line = lines4
        best_lines.append(line)
    if len(lines1) == 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines1
        best_lines.append(line)
    if len(lines1) == 4 and len(lines2) == 4 and len(lines3) != 4 and len(lines4) != 4:
        line = find_line_in_two(lines1, lines2)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) == 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines2
        best_lines.append(line)
    if len(lines1) == 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines1
        best_lines.append(line)

    # lines4不存在
    if len(lines4) != 4 and len(lines1) == 4 and len(lines2) == 4 and len(lines3) == 4:
        line = find_line_in_three(lines1, lines2, lines3)
        best_lines.append(line)
    if len(lines2) == 4 and len(lines1) != 4 and len(lines3) == 4 and len(lines4) != 4:
        line = find_line_in_two(lines2,lines3)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) == 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines2
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) == 4 and len(lines4) != 4:
        line = lines3
        best_lines.append(line)
    if len(lines2) != 4 and len(lines1) == 4 and len(lines3) == 4 and len(lines4) != 4:
        line = find_line_in_two(lines1, lines3)
        best_lines.append(line)
    if len(lines1) == 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines1
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) != 4 and len(lines3) == 4 and len(lines4) != 4:
        line = lines3
        best_lines.append(line)
    if len(lines1) == 4 and len(lines2) == 4 and len(lines3) != 4 and len(lines4) != 4:
        line = find_line_in_two(lines1, lines2)
        best_lines.append(line)
    if len(lines1) != 4 and len(lines2) == 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines2
        best_lines.append(line)
    if len(lines1) == 4 and len(lines2) != 4 and len(lines3) != 4 and len(lines4) != 4:
        line = lines1
        best_lines.append(line)
    return best_lines


# 由拟合直线的斜率k和b计算四条直线的四个交点
def find_cross_point1(topline, bottomline, leftline, rightline):
    """
    :param topline: 顶部直线
    :param bottomline: 下部分直线
    :param leftline: 左边直线
    :param rightline: 右边直线
    :return: 四条直线的交点
    """
    # 通过斜率计算交点
    x1 = (topline[1] - leftline[1]) * 1.0 / (leftline[0] - topline[0])
    y1 = leftline[0] * x1 * 1.0 + leftline[1] * 1.0
    x2 = (topline[1] - rightline[1]) * 1.0 / (rightline[0] - topline[0])
    y2 = topline[0] * x2 * 1.0 + topline[1] * 1.0
    x3 = (bottomline[1] - leftline[1]) * 1.0 / (leftline[0] - bottomline[0])
    y3 = leftline[0] * x3 * 1.0 + leftline[1] * 1.0
    x4 = (bottomline[1] - rightline[1]) * 1.0 / (rightline[0] - bottomline[0])
    y4 = bottomline[0] * x4 * 1.0 + bottomline[1] * 1.0
    t_l_point = ([int(x1), int(y1)])
    t_r_point = ([int(x2), int(y2)])
    b_l_point = ([int(x3), int(y3)])
    b_r_point = ([int(x4), int(y4)])
    return t_l_point, t_r_point, b_l_point, b_r_point


# grabcut图像切割
def grabcut_correct(img3, lt_x, lt_y, leng, wid):
    """
    :param img3: 人脸检测后的图像（旋转过后）
    :param lt_x: 预测边框的左上角横坐标
    :param lt_y: 预测边框的左上角纵坐标
    :param leng: 预测边框的长度
    :param wid: 预测边框的宽度
    :return: grabcut分割后进行轮廓检测后，再进行直线检测后的直线集
    """
    # grabcut前景分割算法
    # 创建了一个与加载图像同形状的掩膜
    # 创建了以0为填充对象的前景和背景模型
    mask = np.zeros(img3.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 标识出想要隔离对象的矩形，前景与背景要基于这个初始矩形留下的区域来决定
    rect = (int(lt_x)-5, int(lt_y)+10, int(leng), int(wid))

    # 进行GrabCut算法，并且用一个矩形进int(lt_y)行初始化模型,5表示算法的迭代次数
    cv2.grabCut(img3, mask, rect, bgdModel, fgdModel, 7, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_binary = mask2 * 255
    img = img3 * mask2[:, :, np.newaxis]
    rectangle = cv2.rectangle(img.copy(), (int(lt_x)-5, int(lt_y) +10), (int(lt_x+leng)-5, int(lt_y+wid) + 10), (0, 255, 0), 2)
    dilate = cv2.dilate(img_binary, (3, 3), iterations=1)

    # 寻找最大轮廓
    img2 = find_max_contour(dilate, img_binary)

    # 对最大轮廓进行直线检测
    l1 = line_detect(img2, img3)

    return l1


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
    return distance


def test_get_id(img):
    get_id_by_binary(img.copy())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)


def get_id_by_binary(img, face_rect):
    img = cv2.pyrMeanShiftFiltering(img, 10, 50)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #  膨胀图像
    erode_elment_x = int(face_rect[2] / 90 * 20)
    erode_elment = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_elment_x, 3))
    img_dilate = cv2.dilate(img_binary, erode_elment, iterations=1)
    if is_show_id_binary:
        plt.imshow(img_dilate, cmap=plt.gray())
        plt.show()
    id_rect = [0, 0, 0, 0]
    # 1. 查找轮廓

    binary,contours,hierarchy= cv2.findContours(img_dilate, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
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

    corners = cv2.goodFeaturesToTrack(gray, 500, 0.06, 1)

    corners = np.int0(corners)
    corners_list = []
    for i in corners:
        x, y = i.ravel()
        corners_list.append([x, y])
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.circle(img_mark, (x, y), 3, (0, 0, 255), -1)

    # 显示原图和处理后的图像
    img_mark_gray = img_mark[:, :, 2]  # cv2.cvtColor(img_mark, cv2.COLOR_BGR2GRAY)
    img_mark_gray = cv2.medianBlur(img_mark_gray, 7)

    if max_face[2] > 70:
        elment = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
    else:
        elment = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    img_mark_gray_dilate = cv2.dilate(img_mark_gray, elment, iterations=1)
    _, contours, _ = cv2.findContours(img_mark_gray_dilate, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

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

            continue
        if -45 < rect[2] <= 0:
            regions_1.append(rect)
        else:
            regions_2.append(rect)
        regions.append(rect)
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
    _, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    _, contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按面积排序

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
    return lines


def predict_location(land_mark, face, id_rect):
    """
    根据人眼睛鼻子眉心位置，估算身份证边界位置
    :param land_mark:左眼右眼眉心鼻子位置
    :param face:人脸位置左上角点坐标及高宽
    :return:
        估算的4边位置
    """
    id_width = max(id_rect[1][0], id_rect[1][1])
    if id_width < 2.2 * face[2] or id_width > 5 * face[2]:
        id_width = int(3.1 * face[2])

    horizontal_x_distance = land_mark[1][0] - land_mark[0][0]
    horizontal_y_distance = land_mark[1][1] - land_mark[0][1]

    eye_id_distance_y = int(id_rect[0][1]) - int((land_mark[1][1] + land_mark[0][1]) / 2 - 2 * horizontal_y_distance)

    if eye_id_distance_y < 1.1 * face[2]:
        eye_id_distance_y = int(1.72 * face[2])

    #  上线
    top_line_pre = [land_mark[0][0] - 5 * horizontal_x_distance, land_mark[0][1] - 5 * horizontal_y_distance,
                    land_mark[1][0] + horizontal_x_distance, land_mark[1][1] + horizontal_y_distance]
    top_line = [top_line_pre[0], top_line_pre[1] + eye_id_distance_y, top_line_pre[2],
                top_line_pre[3] + eye_id_distance_y]
    top_line = [top_line[0], top_line[1] - int(0.97 * id_width), top_line[2],
                top_line[3] - int(0.97 * id_width)]
    for i in range(4):
        if top_line[i] < 0:
            top_line[i] = 0

    # 下线
    bottom_line_pre = [land_mark[0][0] - 5 * horizontal_x_distance, land_mark[0][1] - 5 * horizontal_y_distance,
                       land_mark[1][0] + horizontal_x_distance, land_mark[1][1] + horizontal_y_distance]
    bottom_line = [bottom_line_pre[0], bottom_line_pre[1] + eye_id_distance_y, bottom_line_pre[2],
                   bottom_line_pre[3] + eye_id_distance_y]
    bottom_line = [bottom_line[0], bottom_line[1] + int(0.12 * id_width), bottom_line[2],
                   bottom_line[3] + int(0.12 * id_width)]
    vertical_x_distance = land_mark[3][0] - land_mark[2][0]
    vertical_y_distance = land_mark[3][1] - land_mark[2][1]

    # 左线
    eye_id_distance_x = int((land_mark[2][0] + land_mark[3][0]) / 2 ) - int(id_rect[0][0])
    if eye_id_distance_x > 1.5 * face[2]:
        eye_id_distance_x = int(0.82 * face[2])
    left_line_pre = [land_mark[2][0] - 2 * vertical_x_distance, land_mark[2][1] - 2 * vertical_y_distance,
                     land_mark[3][0] + 2 * vertical_x_distance, land_mark[3][1] + 2 * vertical_y_distance]
    left_line = [left_line_pre[0] - eye_id_distance_x, left_line_pre[1], left_line_pre[2] - eye_id_distance_x,
                 left_line_pre[3]]
    left_line = [left_line[0] - int(1.07 * id_width), left_line[1], left_line[2] - int(1.07 * id_width),
                 left_line[3]]

    for i in range(4):
        if left_line[i] < 0:
            left_line[i] = 0
    # 右线
    right_line_pre = [land_mark[2][0] - 2 * vertical_x_distance, land_mark[2][1] - 2 * vertical_y_distance,
                      land_mark[3][0] + 2 * vertical_x_distance, land_mark[3][1] + 2 * vertical_y_distance]
    right_line = [right_line_pre[0] - eye_id_distance_x, right_line_pre[1], right_line_pre[2] - eye_id_distance_x,
                  right_line_pre[3]]
    right_line = [right_line[0] + int(0.68 * id_width), right_line[1], right_line[2] + int(0.68 * id_width),
                  right_line[3]]

    top_line = [top_line[0], (top_line[1] + top_line[3]) // 2, top_line[2], (top_line[1] + top_line[3]) // 2]
    bottom_line = [bottom_line[0], (bottom_line[1] + bottom_line[3]) // 2, bottom_line[2],
                   (bottom_line[1] + bottom_line[3]) // 2]
    left_line = [(left_line[0] + left_line[2]) // 2, top_line[1], (left_line[0] + left_line[2]) // 2, left_line[3]]
    right_line = [(right_line[0] + right_line[2]) // 2, right_line[1], (right_line[0] + right_line[2]) // 2,
                  right_line[3]]
    predict_border_lines = [top_line, bottom_line, left_line, right_line]
    return predict_border_lines


def get_border_by_canny(img, is_filter=0):
    """
    使用canny算子计算图像的梯度，并进行hough直线检测
    :param img:
    :return:
    """

    if is_filter:
        img = cv2.pyrMeanShiftFiltering(img, 10, 50)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny算子边缘检测
    edges = cv2.Canny(img_gray, 70, 30)
    elment = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, elment, iterations=1)

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

    # 转回uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst = cv2.medianBlur(dst, 3)
    _, dst_binary = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)
    if is_debug == 1:
        plt.imshow(dst_binary, cmap=plt.gray())
        plt.show()

    # 查找轮廓
    _, contours, hierarchy = cv2.findContours(dst_binary.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    # 按面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 将图片涂黑
    fill = cv2.rectangle(dst_binary.copy(), (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)

    # 将最大轮廓涂白
    fill = cv2.drawContours(fill.copy(), contours, 0, (255, 255, 255), -1)

    x = cv2.Sobel(fill, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(fill, cv2.CV_16S, 0, 1, ksize=3)

    # 转回uint8
    absX = cv2.convertScaleAbs(x)
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

    # 查找轮廓
    _, contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 将图片涂黑
    fill = cv2.rectangle(img_binary.copy(), (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)

    # 将最大轮廓涂白
    fill = cv2.drawContours(fill.copy(), contours, 0, (255, 255, 255), -1)

    x = cv2.Sobel(fill, cv2.CV_16S, 1, 0, ksize=3)
    y = cv2.Sobel(fill, cv2.CV_16S, 0, 1, ksize=3)

    # 转回uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    # 对边缘进行膨胀
    elment = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, elment, iterations=1)

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
    cv2.grabCut(img, mask, rect_copy, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_show = img * mask2[:, :, np.newaxis]

    img_binary = mask2 * 255

    # 查找轮廓
    _, contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 将图片涂黑
    fill = cv2.rectangle(img_binary.copy(), (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)

    # 将最大轮廓涂白
    fill = cv2.drawContours(fill.copy(), contours, 0, (255, 255, 255), -1)
    if is_debug == 1:
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
    top_mean = img_hsv[0:3, :, 2].mean()
    bottom_mean = img_hsv[-3:h + 1, :, 2].mean()
    left_mean = img_hsv[:, 0:3, 2].mean()
    right_mean = img_hsv[:, -3:w + 1, 2].mean()
    border_mean = (top_mean + bottom_mean + left_mean + right_mean) / 4
    center_mean = img_hsv[int(h / 2) - 20:int(h / 2) + 20, int(w / 2) - 20:int(w / 2) + 20, 2].mean()
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
    while i < len(lines_2d):
        line = lines_2d[i]
        merge_line = []
        j = 0
        for h_line in lines_2d:
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
            if k_d_c < 0.2 and y_d_c < y_d:
                top_line = top_current
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
        for i in range(start + 1, len(bottom_lines)):
            bottom_current = bottom_lines[i]
            k_bottom_current = abs((bottom_current[3] - bottom_current[1]) / (bottom_current[2] - bottom_current[0]))
            y_current = (bottom_current[1] + bottom_current[3]) / 2
            k_d_c = abs(k_bottom_predict - k_bottom_current)
            y_d_c = y_predict - y_current
            if k_d_c < 0.2 and abs(y_d_c) < abs(y_d) and y_d_c < 10:
                bottom_line = bottom_current
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

    if orientation:
        line12_distance = abs((line1[0] + line1[2]) / 2 - (line2[0] + line2[2]) / 2)
        line13_distance = abs((line1[0] + line1[2]) / 2 - (line3[0] + line3[2]) / 2)
        line23_distance = abs((line2[0] + line2[2]) / 2 - (line3[0] + line3[2]) / 2)
    else:
        line12_distance = abs((line1[1] + line1[3]) / 2 - (line2[1] + line2[3]) / 2)
        line13_distance = abs((line1[1] + line1[3]) / 2 - (line3[1] + line3[3]) / 2)
        line23_distance = abs((line2[1] + line2[3]) / 2 - (line3[1] + line3[3]) / 2)
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

    # 直线line1到line2的距离
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
    :param lines2: 通过最大轮廓得到的边界线f
    :return: 选择的最佳边界线
    """
    best_lines = []

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
            if k_d_c < 0.2 and y_d_c < y_d:
                top_line = top_current
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
    return best_lines


def filter_and_classify_lines(img, lines, img2, face_rect):
    """
    过滤并将直线分类（上、下、左、右）
    :param lines: 直线
    :param img: 原图
    :param img2: 调试标记图片
    :return: 分类后的直线
    """

    # 创建水平和垂直线list
    horizontal, vertical = [], []
    lines_2d_original = lines[:, 0, :]
    lines_2d = []
    for line in lines_2d_original:
        x1, y1, x2, y2 = line
        lines_2d.append([x1, y1, x2, y2])

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
    box = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
    cv2.drawContours(img2, np.array([box]), 0, (0, 255, 0), 2)

    for line in horizontal:
        x1, y1, x2, y2 = line
        if (y1 + y2) / 2 < face_top:
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

    if is_show_point_lines == 1:
        plt.imshow(img)
        plt.show()

    return np.array([t_l_point, t_r_point, b_l_point, b_r_point])


def test_point(img, lines1, lines2, lines3):
    if len(lines1) == 4:
        draw_point_lines(img.copy(), lines1)
    if len(lines2) == 4:
        draw_point_lines(img.copy(), lines2)
    if len(lines3) == 4:
        draw_point_lines(img.copy(), lines3)


# 根据直线的距离比较最佳直线的正面纠偏
def front_correct_skew(img):
    """
    正面纠偏
    :param img: 图像
    :return: 纠偏后的图像
    """

    img2 = copy.deepcopy(img)
    img3 = copy.deepcopy(img)

    # 人脸及人脸特征点检测

    detector = dlib.get_frontal_face_detector()
    # C:/Users/Alexi/Desktop/IDCard_Identification/src/shape_predictor_68_face_landmarks.dat
    predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")
    faces = detector(img, 1)
    face_rect = faces[0]
    max_face = [face_rect.left(), face_rect.top(), face_rect.right() - face_rect.left(),
                face_rect.bottom() - face_rect.top()]
    shape = predictor(img, faces[0])

    land_mark = []
    land_mark.append((shape.part(36).x, shape.part(36).y))
    land_mark.append((shape.part(45).x, shape.part(45).y))
    land_mark.append((shape.part(27).x, shape.part(27).y))
    land_mark.append((shape.part(57).x, shape.part(57).y))

    cv2.circle(img2, land_mark[0], 8, (0, 0, 255), 2)
    cv2.circle(img2, land_mark[1], 8, (0, 0, 255), 2)
    cv2.circle(img2, land_mark[2], 8, (0, 0, 255), 2)
    cv2.circle(img2, land_mark[3], 8, (0, 0, 255), 2)

    if is_debug == 1:
        plt.imshow(img2)
        plt.show()

    id_rect = get_id_by_binary(img.copy(), max_face)
    predict_border_lines = predict_location(land_mark, max_face, id_rect)
    if is_show_predict_lines:
        for line in predict_border_lines:
            x1, y1, x2, y2 = line
            cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 5)
        if is_debug ==1:
            plt.imshow(img2)
            plt.show()

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
        best_lines_by_contour = select_border_lines_by_max_contour_1(top_lines, bottom_lines, left_lines, right_lines, predict_border_lines, img.copy())
    except Exception as e:
        best_lines_by_contour = []
    try:
        lines_by_grabcut = get_border_by_grabcut(img.copy(), copy.deepcopy(predict_border_lines))
        top_lines, bottom_lines, left_lines, right_lines = filter_and_classify_lines(img.copy(), lines_by_grabcut, img2.copy(),
                                                                                     face_rect)
        best_lines_by_grabcut = select_border_lines_by_max_contour_1(top_lines, bottom_lines, left_lines, right_lines, predict_border_lines, img.copy())
    except Exception as e:
        best_lines_by_grabcut = []

    best_lines = select_best_border(id_rect, best_lines_by_canny, best_lines_by_contour, best_lines_by_grabcut)

    top = best_lines[0]
    bottom = best_lines[1]
    left = best_lines[2]
    right = best_lines[3]

    t_l_point = find_cross_point(top, left)
    t_r_point = find_cross_point(top, right)
    b_l_point = find_cross_point(bottom, left)
    b_r_point = find_cross_point(bottom, right)
    return t_l_point, t_r_point, b_l_point, b_r_point, img2


# 根据斜率k和b选择最佳直线的正面纠偏
def front_correct_skew1(img):
    """
    正面纠偏
    :param img: 图像
    :return: 纠偏后的图像
    """

    img2 = copy.deepcopy(img)
    img3 = copy.deepcopy(img)
    # 人脸及人脸特征点检测

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(
        "../data/shape_predictor_68_face_landmarks.dat")
    faces = detector(img, 1)
    face_rect = faces[0]
    max_face = [face_rect.left(), face_rect.top(), face_rect.right() - face_rect.left(),
                face_rect.bottom() - face_rect.top()]
    shape = predictor(img, faces[0])

    land_mark = []
    land_mark.append((shape.part(36).x, shape.part(36).y))
    land_mark.append((shape.part(45).x, shape.part(45).y))
    land_mark.append((shape.part(27).x, shape.part(27).y))
    land_mark.append((shape.part(57).x, shape.part(57).y))

    cv2.circle(img2, land_mark[0], 8, (0, 0, 255), 2)
    cv2.circle(img2, land_mark[1], 8, (0, 0, 255), 2)
    cv2.circle(img2, land_mark[2], 8, (0, 0, 255), 2)
    cv2.circle(img2, land_mark[3], 8, (0, 0, 255), 2)

    if is_debug == 1:
        plt.imshow(img2)
        plt.show()

    id_rect = get_id_by_binary(img.copy(), max_face)
    predict_border_lines = predict_location(land_mark, max_face, id_rect)
    if is_show_predict_lines:
        for line in predict_border_lines:
            x1, y1, x2, y2 = line
            cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # 通过canny算子计算梯度，并检测直线
    try:
        lines_by_canny = get_border_by_canny(img.copy())
        if len(lines_by_canny) > 100:
            lines_by_canny = get_border_by_canny(img.copy(), 1)
        top_lines, bottom_lines, left_lines, right_lines = filter_and_classify_lines(img.copy(), lines_by_canny,
                                                                                     img2.copy(),
                                                                                     face_rect)
        canny_lines = []
        top_line = fitline(top_lines)
        bottom_line = fitline(bottom_lines)
        left_line = fitline(left_lines)
        right_line = fitline(right_lines)

        # 添加k,b值
        canny_lines.append([top_line[0], top_line[1]])
        canny_lines.append([bottom_line[0], bottom_line[1]])
        canny_lines.append([left_line[0], left_line[1]])
        canny_lines.append([right_line[0], right_line[1]])

    except Exception as e:
        canny_lines = []

    # 获取二值化后的最大轮廓的边界线
    try:
        lines_by_contour = get_border_by_binary_max_contour(img.copy())
        top_lines, bottom_lines, left_lines, right_lines = filter_and_classify_lines(img.copy(), lines_by_contour,
                                                                                     img2.copy(),
                                                                                     face_rect)
        contour_lines = []
        top_line = fitline(top_lines)
        bottom_line = fitline(bottom_lines)
        left_line = fitline(left_lines)
        right_line = fitline(right_lines)
        # 添加k,b值
        contour_lines.append([top_line[0], top_line[1]])
        contour_lines.append([bottom_line[0], bottom_line[1]])
        contour_lines.append([left_line[0], left_line[1]])
        contour_lines.append([right_line[0], right_line[1]])
    except Exception as e:
        contour_lines = []

    try:
        lines_by_grabcut = get_border_by_grabcut(img.copy(), copy.deepcopy(predict_border_lines))
        top_lines, bottom_lines, left_lines, right_lines = filter_and_classify_lines(img.copy(), lines_by_grabcut,
                                                                                     img2.copy(),
                                                                                     face_rect)
        grabcut_lines = []
        top_line = fitline(top_lines)
        bottom_line = fitline(bottom_lines)
        left_line = fitline(left_lines)
        right_line = fitline(right_lines)

        # 添加k,b值
        grabcut_lines.append([top_line[0],top_line[1]])
        grabcut_lines.append([bottom_line[0], bottom_line[1]])
        grabcut_lines.append([left_line[0], left_line[1]])
        grabcut_lines.append([right_line[0], right_line[1]])
    except Exception as e:
        grabcut_lines = []

    try:
        shape, img1, img3, img4, img5, ratio = face_detect_rotation(img)
        lt_x, lt_y, leng, wid = predict_rect(img1, img3, shape)
        l1 = grabcut_correct(img3, lt_x, lt_y, leng, wid)
        l_limit, v_limit = line_area(l1, lt_x, lt_x + leng, lt_y, lt_y + wid)
        top_line1, bottom_line1, left_line1, right_line1 = classify_lines(l_limit, v_limit)

        grabcut1_lines = []
        grabcut1_lines.append([top_line1[0], top_line1[1]])
        grabcut1_lines.append([bottom_line1[0], bottom_line1[1]])
        grabcut1_lines.append([left_line1[0], left_line1[1]])
        grabcut1_lines.append([right_line1[0], right_line1[1]])
    except Exception as e:
        grabcut1_lines = []

    best_lines = select_best_border(canny_lines, contour_lines, grabcut_lines, grabcut1_lines)

    if is_debug == 1:
        plt.imshow(img2)
        plt.show()
    # 按直线中点位置排序

    return best_lines, img2

# 正反面纠偏
def correct_skew(img, is_front, max_face=[0, 0, 0, 0]):
    """
        检测最大轮廓并进行透视变换和裁剪
        默认大小1400x900 （身份证比例
        :param save_path: 存储路径, 处理后的图像保存在指定路径, 文件名和源文件相同
        :param show_process: 显示处理过程
        :param pic_path: 原图路径
        :return:正面或者反面纠偏后透视变换后的图像
        """

    if is_front == 1:
        img_original = copy.deepcopy(img)

        try:
            t_l_point, t_r_point, b_l_point, b_r_point, img2 = front_correct_skew(img)
        except Exception as e:
            shape, img1, img2, img4, img5, ratio = face_detect_rotation(img)
            lt_x, lt_y, leng, wid = predict_rect(img1, img2, shape)
            l1 = grabcut_correct(img2, lt_x, lt_y, leng, wid)
            l_limit, v_limit = line_area(l1, lt_x, lt_x + leng, lt_y, lt_y + wid)
            top_line, bottom_line, left_line, right_line = classify_lines(l_limit, v_limit)
            t_l_point, t_r_point, b_l_point, b_r_point = find_cross_point1(top_line, bottom_line, left_line, right_line)

        # 用红色画出四个顶点
        for point in t_l_point, t_r_point, b_l_point, b_r_point:
            cv2.circle(img, (point[0],point[1]), 8, (0, 0, 255), 2)

        # 用蓝色画出四条边
        cv2.line(img, (t_l_point[0],t_l_point[1]), (t_r_point[0],t_r_point[1]), (255, 0, 0), 3)
        cv2.line(img, (b_r_point[0],b_r_point[1]), (t_r_point[0],t_r_point[1]), (255, 0, 0), 3)
        cv2.line(img, (b_r_point[0],b_r_point[1]), (b_l_point[0],b_l_point[1]), (255, 0, 0), 3)
        cv2.line(img, (b_l_point[0],b_l_point[1]), (t_l_point[0],t_l_point[1]), (255, 0, 0), 3)
        if is_debug == 1:
            plt.imshow(img)
            plt.show()
            cv2.imwrite("img_test1.jpg", img)

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
        if is_debug == 1:
            plt.imshow(dst)
            plt.show()
        return dst

    else:
        img_original = copy.deepcopy(img)

        img, top, bottom, left, right = back_correct_skew(img)
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
        # 保存图片
        return dst
