# 利用GrabCut将前景与背景分割
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import dlib
import matplotlib.patches as mpatches
from PIL import Image
import copy

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  ##定义

# 按比例进行大小的调整
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    ''' 插值方法选用基于局部像素的重采样
     image.shape[0]指图片长（垂直尺寸），image.shape[1]指图片宽（水平尺寸）
     h为真实高度，height为设定高度0
     dim括号内的是按比例调整后的长和宽
     为真实宽度，width为设定宽度
     4行2列的零矩阵，四个坐标点'''
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
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
    maxWidth = max(int(widthA), int(widthB))  # 取W1与W2中最大的
    # 同理得到
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 变换后对应坐标位置，dst为变换后的点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")  # 作-1更精确
    # 计算变换矩阵M
    M = cv2.getPerspectiveTransform(rect, dst)  # 由二维变成三维，然后再转换为二维，转换成矩形，opencv会自动将坐标点进行计算，得出变换矩阵M
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
    # plt.subplot(121), plt.imshow(src[:, :, ::-1]), plt.title('input')
    # plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
    # img[:, :, ::-1]是将BGR转化为RGB

    return dst

# 基于两直线的交点计算
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
        x = (b2 - b1) * 1.0 / (k1 - k2)  # k1X+b1 = k2X+b2(令y1=y2求交点)
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
    # 加载人脸检测
    detector = dlib.get_frontal_face_detector()
    # 加载5个关键点的定位
    predictor = dlib.shape_predictor('C:/Users/Administrator/PycharmProjects/IDCardIndetification/foo/tools/shape_predictor_5_face_landmarks.dat')

    orig = img.copy()
    # img = resize(orig, width=500)
    print(img.shape)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测，检测出来人脸框,2表示采样次数，得到人脸检测的大框Z
    rects = detector(img1, 2)

    # 自定义旋转
    # 通过人脸检测来进行图像的旋转处理，从而进行人脸检测
    # for i in range(90, 360, 90):
    #     if len(rects) == 0:
    #         img1 = Image.fromarray(np.array(img))
    #         rot = img1.rotate(90, expand=True)
    #         img_rotation = Image.fromarray(np.array(rot))
    #         img = np.array(img_rotation)
    #         img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         rects = detector(img1, 2)
    #         if len(rects) != 0:
    #             break
    #         else:
    #             continue
    shape = predictor(img1, rects[0])  # 对人脸框进行关键点的定位，人脸关键点相对于框的位置
    shape = shape_to_np(shape)  # 转换成ndarray格式，转换成坐标值
    # img = resize(img, width=500)
    ratio = img.shape[1] / 500.0

    img2 = copy.deepcopy(img)
    img4 = copy.deepcopy(img)
    img5 = copy.deepcopy(img)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(img1, 2)
    plt.figure()
    ax = plt.subplot(111)
    ax.imshow(img4)
    # plt.axis('off')
    plt.scatter(shape[0:5, 0], shape[0:5, 1], color='r', marker='o', s=5)  # 'ro'表示red，'o'类型的线条
    for i in np.arange(5):
        plt.text(shape[i, 0] - 8, shape[i, 1] - 8, i)
    # plt.show()  # 5点法检测人脸，左眼、右眼和唇上口#检测眼睛个唇上口的位置
    print('人脸检测完毕')  # 人脸识别
    #读取图片，并且做初始化旋转操作#读取图片+人脸识别

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
    from foo.tools.back_correct_skew import mark_corner_image
    img_mark = mark_corner_image(img, point_size = 3)
    # 投影计算角度，对图片进行纠正
    from foo.tools.tools import project

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
    # 加载人脸检测
    detector = dlib.get_frontal_face_detector()
    classfier = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_alt2.xml")

    from foo.tools.back_correct_skew import cal_rotation_angle
    angle = cal_rotation_angle(img.copy())

    # 将填充颜色改为（100,100,100）
    img1 = Image.fromarray(np.array(img))
    im2 = img1.copy().convert('RGBA')
    rot = im2.rotate(angle, expand=True)
    fff = Image.new('RGBA', rot.size, (100,) * 4)
    out = Image.composite(rot, fff, rot)
    img_rotation = out.convert(img1.mode)

    img_rotation = Image.fromarray(np.array(img_rotation))

    # img_rotation = Image.fromarray(np.array(img))
    # 使用dlib人脸检测模型
    faces_dlib = detector(np.array(img_rotation), 1)
    # print(faces_dlib[0])
    #
    # box = [[318, 474], [318, 367], [426, 367], [426, 474]]
    # # cv2.drawContours(img, np.array([box]), 0, (0, 255, 0), 2)
    # plt.imshow(img, cmap=plt.gray())
    # plt.show()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(np.array(img_rotation), cv2.COLOR_BGR2GRAY)
    else:
        gray = np.array(img_rotation)
    # plt.imshow(gray, cmap=plt.gray())
    # plt.show()
    # 使用opencv人脸检测模型
    faces_cv = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    # print(faces_cv)
    # plt.imshow(img_rotation, cmap=plt.gray())
    # plt.show()
    is_front = 1
    # print(len(faces_dlib) , len(faces_cv) )
    if len(faces_dlib) == 0 or len(faces_cv) == 0:
        img_rotation = img_rotation.rotate(180, expand=True)
        faces_dlib = detector(np.array(img_rotation), 1)

        if len(img.shape) == 3:
            grey = cv2.cvtColor(np.array(img_rotation), cv2.COLOR_BGR2GRAY)
        else:
            grey = np.array(img_rotation)

        # 使用opencv人脸检测模型
        faces_cv = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        #print(len(faces_dlib),len(faces_cv))
        if len(faces_dlib) == 0 or len(faces_cv) == 0:
            is_front = 0

    # print(len(faces_dlib), len(faces_cv))
    # plt.imshow(img_rotation, cmap=plt.gray())
    # plt.show()
    if is_front:
        return np.array(img_rotation)
    else:
        return img


def predict_rect(gray, scr, shape):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqlKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    # cv_show('tophat', tophat)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype('uint8')
    # cv_show('gradX', gradX)

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel, iterations=2)
    # cv_show('gradX', gradX)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv_show('thresh', thresh)

    thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqlKernel)
    # cv_show('thresh', thresh)

    threshCnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = threshCnts

    cur_img = scr.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 2)
    # cv_show('img', cur_img)

    locs = []
    # 遍历轮廓
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
            # print(w, h)
        ar = w / float(h)
        mid = (shape[1][1] + shape[3][1]) / 2.0

            # 选择合适的区域，根据实际任务来，这里的基本上都是四个数字一组
        if ar > 11 and ar < 40 and y >= shape[4][1]+1.5*(shape[4][1]-mid) and w >= 120  :
                # 把符合的留下来
            locs.append((x, y, w, h))
            # print('locs', locs)

            gX = locs[0][0]
            gY = locs[0][1]
            gW = locs[0][2]
            gH = locs[0][3]
                # id_le = []
                # if len(locs) != 0:
                #     for i in range(0, len(locs)):
                #         id_le.append(locs[i][2] - locs[i][0])
                #         i += 1
                #         if i > len(locs):
                #             break
                #     # print('id_le', id_le)
                #     # 身份证位置长度筛选
                #     # 返回最大长度框的索引值
                #     location = id_le.index(max(id_le))
                #     # print('location', location)
                #     max_rect = locs[location]
                #     print('max_rect', max_rect)
                #
                #     if max_rect[2] >= 550 or max_rect[2] <= 140:
                #         max_rect = 0
                #     else:
                #         gX = max_rect[0]
                #         gY = max_rect[1]
                #         gW = max_rect[2]
                #         gH = max_rect[3]
            cv2.rectangle(scr, (gX - 3, gY - 3), (gX + gW + 3, gY + gH + 3), (0, 0, 255), 2)
            detect_id = []
            detect_id.append([gX - 3, gY - 3, gX + gW + 3, gY + gH + 3])
            # cv_show('rectangle', scr)

    #预估grabcut的rect值
    if len(locs) == 0:
        leng = abs(11.5 * (shape[0][0] - shape[2][0]))
    else:
        leng = abs(1.92 * (detect_id[0][2] - detect_id[0][0]))
    mid = (shape[1][1] + shape[3][1]) / 2.0
    wid = abs(14 * (mid - shape[4][1]))
    print('leng', leng)
    print('wid', wid)  # 计算长度和宽度

    rt_x = shape[4][0] * 1.0 + (leng / 4.0)
    rt_y = shape[4][1] * 1.0 - (wid / 2.0)
    rb_x = rt_x
    rb_y = rt_y + wid
    lt_x = rt_x - leng
    lt_y = rt_y
    lb_x = rt_x - leng
    lb_y = rt_y + wid  # 计算出rect的四个点
    return lt_x, lt_y, leng, wid

# 投影变换
# def projection(img_binary, orientation=0, is_show = 0):
#     """
#     图片投影
#     :param img_binary: 二值图片
#     :param orientation: 投影方向 0：水平投影  1：垂直投影
#     :return:
#     """
#     img_binary = np.array(img_binary)
#     (h, w) = img_binary.shape  # 返回高和宽
#     # print(h,w)#s输出高和宽
#     thresh_copy = img_binary.copy()
#     thresh_copy[:, :] = 255
#     # 水平投影
#     if orientation == 0:
#         # horizontal_count = np.array([0 for z in range(0, h)])
#         # # 每行白色像素个数
#         # for j in range(0, h):
#         #     horizontal_count[j] = np.count_nonzero(img_binary[j, :])
#         horizontal_count = img_binary.sum(axis=1) / 255
#         count = horizontal_count
#         for j in range(0, h):  # 遍历每一行
#             # for i in range(0, int(horizontal_count[j])):
#             #     thresh_copy[j, i] = 0
#             thresh_copy[j, 0:int(horizontal_count[j])] = 0
#         if is_show:
#             plt.imshow(thresh_copy, cmap=plt.gray())
#             plt.show()
#     else:
#         # vertical_count = np.array([0 for z in range(0, w)])
#         # # 每列白色像素个数
#         # for j in range(0, w):
#         #     vertical_count[j] = np.count_nonzero(img_binary[:, j])
#         vertical_count = img_binary.sum(axis=0) / 255
#         count = vertical_count
#         for j in range(0, w):  # 遍历每一列
#             # for i in range((h - int(vertical_count[j])), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
#             #     thresh_copy[i, j] = 0  # 涂黑
#             thresh_copy[(h - int(vertical_count[j])): h, j] = 0
#         if is_show:
#             plt.imshow(thresh_copy, cmap=plt.gray())
#             plt.show()
#     return count
#
# # 提取峰值范围
# def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
#     """
#     提取波峰
#     :param array_vals: 投影数组
#     :param minimun_val: 波峰最小值
#     :param minimun_range: 波峰最小跨度
#     :return:
#     """
#     start_i = None
#     end_i = None
#     peek_ranges = []
#     for i, val in enumerate(array_vals):
#         if val >= minimun_val and start_i is None:
#             start_i = i
#         elif val >= minimun_val and start_i is not None:
#             pass
#         elif val < minimun_val and start_i is not None:
#             end_i = i
#             if end_i - start_i >= minimun_range:
#                 peek_ranges.append((start_i, end_i))
#             start_i = None
#             end_i = None
#         elif val < minimun_val and start_i is None:
#             pass
#         else:
#             print(val , minimun_val , start_i )
#             raise ValueError("cannot parse this case...")
#     return peek_ranges
#
# # 标记角点位置
# def mark_corner_image(img,point_size = 3):
#     """
#     标记角点位置
#     :param img: 图片
#     :param point_size: 标记角点大小
#     :return:
#     """
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_mark = gray.copy()
#     img_mark[:, :] = 0
#     gray = np.float32(gray)
#
#     # 输入图像必须是 float32 ,最后一个参数在 0.04 到 0.05 之间
#     corners = cv2.goodFeaturesToTrack(gray, 500, 0.06, 1)
#
#     corners = np.int0(corners)
#     # print(corners)
#     corners_list = []
#     for i in corners:
#         x, y = i.ravel()
#         # if y < 329 + 1.5 * 74 or y > 329 + 2.5 * 74:
#         #     continue
#         corners_list.append([x, y])
#         #cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
#         cv2.circle(img_mark, (x, y), point_size, (255, 255, 255), -1)
#
#
#     return img_mark
#
# # 计算旋转角度
# def cal_rotation_angle(img):
#     """
#     计算旋转角度
#     :param img: 图片
#     :return:旋转角度
#     """
#     img_mark = mark_corner_image(img,point_size = 3)
#     # 投影计算角度，对图片进行纠正
#     project_image = np.zeros((180, img_mark.shape[0]))
#
#     img_mark_im = Image.fromarray(img_mark)
#     for i in range(180):
#         im_rotate = img_mark_im.rotate(i)
#         project_image[i] = projection(im_rotate, orientation=0)
#
#     max_index = np.where(project_image[:, :] == project_image.max())
#     rotate_angle = max_index[0][0]
#     vertical_peek_ranges2d = []
#     return rotate_angle
#
# # 纠正图片
# def correct_image(img):
#     """
#     纠正图片
#     :param img: 图片
#     :return: 纠正后的图片、（中华人民共和国位置）、（有效日期位置）
#     """
#     angle = cal_rotation_angle(img)
#     img_im = Image.fromarray(img)
#     img_correction = np.array(img_im.rotate(angle, expand=True))
#     # 双边均值滤波
#     # img_correction = cv2.pyrMeanShiftFiltering(img_correction, 10, 50)
#     # img_correction = np.array(img_correction)
#     img_mark = mark_corner_image(img_correction)
#
#     # print("radon time:",datetime.now() - start_time)
#     horizontal_sum = projection(img_mark, orientation=0, is_show=1)
#     peek_ranges = extract_peek_ranges_from_array(horizontal_sum)
#     print(peek_ranges)
#     # img_correct = correct_image(img)
#     img_mblur = cv2.medianBlur(img_mark, 1)
#
#
#     elment = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 2))
#     img_dilate = cv2.dilate(img_mblur, elment, iterations=2)
#
#
#     contours, _ = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     rects = []
#     rects_m = []
#     img_filter = img_dilate.copy()
#     img_filter[:, :] = 0
#     for i in range(len(contours)):
#         cnt = contours[i]
#         # 计算该轮廓的面积
#         area = cv2.contourArea(cnt)
#         # 面积小的都筛选掉
#         if (area < 800):
#             continue
#
#         # 找到最小的矩形，该矩形可能有方向
#         rect = cv2.boundingRect(cnt)
#         if rect[2] < 150 or rect[3] < 10:
#             continue
#         rect_m = cv2.minAreaRect(cnt)
#         #print(rect_m)
#         rects.append(rect)
#         rects_m.append(rect_m)
#
#         # cv2.drawContours(img_filter, cnt, 0, (255, 255, 255), 1)
#     rects = np.array(rects)
#     text1_index = np.where(rects[:, 1]==rects[:, 1].min())
#     text1 = rects_m[text1_index[0][0]]
#
#     text2_index = np.where(rects[:, 1] == rects[:, 1].max())
#     text2 = rects_m[text2_index[0][0]]
#
#     rects = np.array(rects)
#     # plt.imshow(img_dilate, cmap=plt.gray())
#     # plt.show()
#     y_mean = rects[:, 1].mean()
#     max_rect = rects[np.where(rects[:, 2] == rects[:, 2].max())]
#     return np.array(img_correction), text1, text2

# 身份证位置检测

# def id_number_detect(img3,img1):
#
#     return locs, max_rect, gX, gY, gW, gH

# 限定直线检测的区域

# 寻找最大轮廓
def find_max_contour(after_grabcut, img):
    cnts = cv2.findContours(after_grabcut, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    # 轮廓排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]  # 获取面积最大的一个轮廓
    img2 = img.copy()
    img2[:, :] = 0
    cv2.drawContours(img2, cnts, -1, (255, 255, 255), 1)  # 获取并绘制轮廓
    # cv_show('outline', img2)
    return img2


def line_detect(after_contours, scr):
    # 对轮廓检测后的直线进行直线检测
    minLineLength = 140
    maxLineGap = 7
    lines = cv2.HoughLinesP(after_contours, 1, np.pi / 360, 10, minLineLength, maxLineGap)  # 笔记记载了
    l1 = lines[:, 0, :]
    for x1, y1, x2, y2 in l1:  # lines[:,0,:]将直线压缩成二维图像，找出两个端点(降低维度)
        cv2.line(scr, (x1, y1), (x2, y2), (0, 255, 0), 1)
    plt.imshow(scr, cmap=plt.gray())
    # plt.show()
    return l1

# 限制直线检测后的直线数量
def line_area(l1, lt_x, rt_x, lt_y, lb_y):
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
    print('levelline', levelline)
    print('vertline', vertline)

    # 水平线限制区
    l_limit = []
    for i in range(0, len(levelline)):
        if levelline[i][1] >= lt_y + 6 and levelline[i][1] <= lb_y + 12 and levelline[i][0] >= lt_x - 7 and levelline[i][0] <= rt_x + 6:
            l_limit.append(levelline[i])
        i += 1
        if i > len(levelline) - 1:
            break
    print('l_limit', l_limit)

    # 垂直线限制区
    v_limit = []
    for j in range(0, len(vertline)):
        if vertline[j][1] >= lt_y + 6 and vertline[j][1] <= lb_y + 12 and vertline[j][0] >= lt_x - 7 and vertline[j][0] <= rt_x + 6:
            v_limit.append(vertline[j])
        j += 1
        if j > len(vertline) - 1:
            break
    print('v_limit', v_limit)
    return l_limit, v_limit

# 划分上下左右直线
def line_group(l_limit, v_limit):
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

    print('top_level', top_level)  # 显示出上部分的水平线
    print('bottom_level', bottom_level)  # 显示出下部分的水平线
    print('left_vert', left_vert)  # 显示出左侧的垂直线
    print('right_vert', right_vert)  # 显示出右侧的垂直线
    return top_level, bottom_level, left_vert, right_vert

# 进行直线拟合
def fitline(lines):
    loc = []
    for line in lines:
        x1,y1,x2,y2 = line
        loc.append([x1,y1])
        loc.append([x2,y2])

    loc = np.array(loc)
    output = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)
    if output[1] != 1:
        k = (output[1] / output[0])
        b = (output[3] - k*output[2])
    else:
        k = 57.29
        b = (output[3] - 57.29*output[2])
    return k,b


def classify_lines(l_limit, v_limit):
    top_level, bottom_level, left_vert, right_vert = line_group(l_limit,v_limit)

    topline = fitline(top_level)#上一部分水平线拟合而成的斜率和b值
    bottomline = fitline(bottom_level)#下一部分水平线拟合而成的斜率和b值
    leftline = fitline(left_vert)#左侧垂直线的拟合而成的斜率和b值
    rightline = fitline(right_vert)#右侧垂直线的拟合而成的斜率和b值
    return topline,bottomline,leftline,rightline
#
def distance(lines1 ,lines2):
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

    # grabcut前景分割算法
    # 创建了一个与加载图像同形状的掩膜
    # 创建了以0为填充对象的前景和背景模型
    mask = np.zeros(img3.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # 标识出想要隔离对象的矩形，前景与背景要基于这个初始矩形留下的区域来决定
    rect = (int(lt_x)-5, int(lt_y)+10, int(leng), int(wid))
    print('rect', rect)
    # 进行GrabCut算法，并且用一个矩形进int(lt_y)行初始化模型,5表示算法的迭代次数
    cv2.grabCut(img3, mask, rect, bgdModel, fgdModel, 7, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_binary = mask2 * 255
    img = img3 * mask2[:, :, np.newaxis]
    rectangle = cv2.rectangle(img.copy(), (int(lt_x)-5, int(lt_y) +10), (int(lt_x+leng)-5, int(lt_y+wid) + 10), (0, 255, 0),2)
    # cv_show('rectangle', rectangle)
    # canny = cv2.Canny(mask3,70,30)
    dilate = cv2.dilate(img_binary, (3, 3), iterations=1)
    # cv_show('dilate', dilate)

    # 寻找最大轮廓
    img2 = find_max_contour(dilate, img_binary)

    # 对最大轮廓进行直线检测
    l1 = line_detect(img2, img3)

    return l1




# if __name__ == "__main__":
    # input_dir = "C:/Users/Administrator/PycharmProjects/cread_ocr/test1/"
    # output_dir = "C:/Users/Administrator/PycharmProjects/cread_ocr/result3/"
    # is_batch = 1
    # if is_batch:
    #     input_dir = input_dir
    #     output_dir = output_dir
    #     batch_process(input_dir, output_dir)  # 批量处理
    # else:
    #     img_name = img_name
    #     path = path_without_img_name + img_name
    #     save_name = "../output/" + img_name.split(".")[0] + ".jpg"
    #     single_process(path, save_name)  # 单张调试




    #
    # img = cv2.imdecode(np.fromfile(name, dtype=np.uint8), -1)
    # # 返回人脸检测后的点集，旋转后的图像和图片的比率
    # shape, img1, img3, img4, img5, ratio = face_detect_rotation(img)
    # # locs为检测到的所有框，max_rect为检测到的长度最大的框（可能为身份证号），gX, gY, gW, gH为最大框的范围
    # lt_x, lt_y, leng, wid = predict_rect(img1, img3, shape)
    #
    # rectangle1 = cv2.rectangle(img4.copy(), (int(x1), int(y1)), (int(x4), int(y4)), (0, 255, 0), 2)
    # cv_show('rectangle1', rectangle1)
    #
    # # 透视变换
    # warped = perspective_transformation(p.reshape(4, 2) * ratio, img5)
    # warped1 = resize(warped, width=500)
    # cv_show("scanned", warped1)  # 透视变换
    # cv2.imencode('.jpg', warped1, )[1].tofile(savename)
    #
    #
    # for filename in os.listdir("C:/Users/Administrator/PycharmProjects/cread_ocr/test3/"):
    #     print(filename)
    #     grabcut_correct("C:/Users/Administrator/PycharmProjects/cread_ocr/test3/" + filename, "C:/Users/Administrator/PycharmProjects/cread_ocr/result_3/" + filename)








    # #
    # for filename in os.listdir("f ailed1/"):
    #     print(filename)
    #     grabcut("failed1/"+filename,"result/"+filename)
    #
    # for filename in os.listdir("id_dingwei/"):
    #     print(filename)
    #     grabcut("id_dingwei/"+filename,"result4/"+filename)

    # for filename in os.listdir("finished/"):
    #     print(filename)
    #     grabcut("finished/" + filename, "result4/" + filename)

