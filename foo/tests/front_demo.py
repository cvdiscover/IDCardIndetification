# 利用GrabCut将前景与背景分割
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import dlib
import math
from PIL import Image
import copy


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  ##定义


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


def face_detect_rotation(img):
    # 加载人脸检测
    detector = dlib.get_frontal_face_detector()
    # 加载5个关键点的定位
    predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
    orig = img.copy()
    img = resize(orig, height=450)
    print(img.shape)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测，检测出来人脸框,2表示采样次数，得到人脸检测的大框Z
    rects = detector(img1, 2)

    # 通过人脸检测来进行图像的旋转处理，从而进行人脸检测
    for i in range(90, 360, 90):
        if len(rects) == 0:
            img1 = Image.fromarray(np.array(img))
            rot = img1.rotate(90, expand=True)
            img_rotation = Image.fromarray(np.array(rot))
            img = np.array(img_rotation)
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(img1, 2)
            if len(rects) != 0:
                break
            else:
                continue
    shape = predictor(img1, rects[0])  # 对人脸框进行关键点的定位，人脸关键点相对于框的位置
    shape = shape_to_np(shape)  # 转换成ndarray格式，转换成坐标值
    img = resize(img, height=450)
    ratio = img.shape[0] / 450.0

    img3 = img.copy()
    img4 = img.copy()
    img5 = img.copy()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(img1, 2)
    plt.figure()
    ax = plt.subplot(111)
    ax.imshow(img4)
    # plt.axis('off')
    plt.scatter(shape[0:5, 0], shape[0:5, 1], color='r', marker='o', s=5)  # 'ro'表示red，'o'类型的线条
    for i in np.arange(5):
        plt.text(shape[i, 0] - 8, shape[i, 1] - 8, i)
    plt.show()  # 5点法检测人脸，左眼、右眼和唇上口#检测眼睛个唇上口的位置
    print('人脸检测完毕')  # 人脸识别
    # 读取图片，并且做初始化旋转操作#读取图片+人脸识别

    shape = predictor(img1, rects[0])
    shape = shape_to_np(shape)
    x1 = shape[2][0]
    y1 = shape[2][1]
    x2 = shape[0][0]
    y2 = shape[0][1]
    return shape, img1, img3, img4, img5, ratio


# 身份证位置检测
def id_number_detect(img3, img1):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    sqlKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    tophat = cv2.morphologyEx(img1, cv2.MORPH_TOPHAT, rectKernel)
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

    cur_img = img3.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 2)
    cv_show('img', cur_img)

    locs = []
    # 遍历轮廓
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        # print(w, h)
        ar = w / float(h)

        # 选择合适的区域，根据实际任务来，这里的基本上都是四个数字一组
        if ar > 11 and ar < 40:
            # if (w > 150 and w < 600):  # and (h>10 and h<20):
            # 把符合的留下来
            locs.append((x, y, w, h))

    id_le = []
    if len(locs) != 0:
        for i in range(0, len(locs)):
            id_le.append(locs[i][2] - locs[i][0])
            i += 1
            if i > len(locs):
                break
        # print('id_le', id_le)
        # 身份证位置长度筛选
        # 返回最大长度框的索引值
        location = id_le.index(max(id_le))
        # print('location', location)
        max_rect = locs[location]
        # print('le', max_rect)

        if max_rect[2] >= 550 or max_rect[2] <= 120:
            max_rect = 0
        else:
            gX = max_rect[0]
            gY = max_rect[1]
            gW = max_rect[2]
            gH = max_rect[3]
    return locs, max_rect, gX, gY, gW, gH


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
        if levelline[i][1] >= lt_y + 6 and levelline[i][1] <= lb_y + 12 and levelline[i][0] >= lt_x - 7 and \
                levelline[i][0] <= rt_x + 6:
            l_limit.append(levelline[i])
        i += 1
        if i > len(levelline) - 1:
            break
    print('l_limit', l_limit)

    # 垂直线限制区
    v_limit = []
    for j in range(0, len(vertline)):
        if vertline[j][1] >= lt_y + 6 and vertline[j][1] <= lb_y + 12 and vertline[j][0] >= lt_x - 7 and vertline[j][
            0] <= rt_x + 6:
            v_limit.append(vertline[j])
        j += 1
        if j > len(vertline) - 1:
            break
    print('v_limit', v_limit)
    return l_limit, v_limit


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


def fitline(lines):
    loc = []
    for line in lines:
        x1, y1, x2, y2 = line
        loc.append([x1, y1])
        loc.append([x2, y2])

    loc = np.array(loc)
    output = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)
    if output[1] != 1:
        k = (output[1] / output[0])
        b = (output[3] - k * output[2])
    else:
        k = 57.29
        b = (output[3] - 57.29 * output[2])
    return k, b


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
    point = []
    point.append([int(x1), int(y1)])
    point.append([int(x2), int(y2)])
    point.append([int(x3), int(y3)])
    point.append([int(x4), int(y4)])
    p = np.array(point)
    return x1, y1, x4, y4, p


def grabcut_correct(name, savename):
    img = cv2.imdecode(np.fromfile(name, dtype=np.uint8), -1)
    # 返回人脸检测后的点集，旋转后的图像和图片的比率
    shape, img1, img3, img4, img5, ratio = face_detect_rotation(img)
    # locs为检测到的所有框，max_rect为检测到的长度最大的框（可能为身份证号），gX, gY, gW, gH为最大框的范围
    locs, max_rect, gX, gY, gW, gH = id_number_detect(img3, img1)

    detect_id = []
    cv2.rectangle(img3, (gX - 3, gY - 3), (gX + gW + 3, gY + gH + 3), (0, 0, 255), 2)
    detect_id.append([gX - 3, gY - 3, gX + gW + 3, gY + gH + 3])
    cv_show('rectangle', img3)
    # 预估grabcut的rect值
    if len(locs) == 0 or max_rect == 0:
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

    # grabcut前景分割算法
    # 创建了一个与加载图像同形状的掩膜
    # 创建了以0为填充对象的前景和背景模型
    mask = np.zeros(img3.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # 标识出想要隔离对象的矩形，前景与背景要基于这个初始矩形留下的区域来决定
    rect = (int(lt_x) - 5, int(lt_y) + 10, int(leng), int(wid))
    print('rect', rect)
    # 进行GrabCut算法，并且用一个矩形进int(lt_y)行初始化模型,5表示算法的迭代次数
    cv2.grabCut(img3, mask, rect, bgdModel, fgdModel, 7, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask3 = mask2 * 255
    img = img3 * mask2[:, :, np.newaxis]

    rectangle = cv2.rectangle(img.copy(), (int(lt_x) - 5, int(lt_y) + 10), (int(rb_x) - 5, int(rb_y) + 10), (0, 255, 0),
                              2)
    cv_show('rectangle', rectangle)
    dilate = cv2.dilate(mask3, (3, 3), iterations=1)
    cv_show('dilate', dilate)

    cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    # 轮廓排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]  # 获取面积最大的一个轮廓
    img2 = mask3.copy()
    img2[:, :] = 0
    cv2.drawContours(img2, cnts, -1, (255, 255, 255), 1)  # 获取并绘制轮廓
    cv_show('outline', img2)

    # 对轮廓检测后的直线进行直线检测
    minLineLength = 140
    maxLineGap = 7
    lines = cv2.HoughLinesP(img2, 1, np.pi / 360, 10, minLineLength, maxLineGap)  # 笔记记载了
    l1 = lines[:, 0, :]
    for x1, y1, x2, y2 in l1:  # lines[:,0,:]将直线压缩成二维图像，找出两个端点(降低维度)
        cv2.line(img3, (x1, y1), (x2, y2), (0, 255, 0), 1)
    plt.figure()
    ax = plt.subplot(111)
    ax = plt.imshow(img3)
    plt.show()

    # 区域筛选后的水平线和垂直线
    l_limit, v_limit = line_area(l1, lt_x, rt_x, lt_y, lb_y)

    # 对区域内的直线进行上下左右的分类
    top_level, bottom_level, left_vert, right_vert = line_group(l_limit, v_limit)

    topline = fitline(top_level)  # 上一部分水平线拟合而成的斜率和b值
    bottomline = fitline(bottom_level)  # 下一部分水平线拟合而成的斜率和b值
    leftline = fitline(left_vert)  # 左侧垂直线的拟合而成的斜率和b值
    rightline = fitline(right_vert)  # 右侧垂直线的拟合而成的斜率和b值

    # 由斜率k和b计算上下左右拟合直线的交点
    x1, y1, x4, y4, p = find_cross_point1(topline, bottomline, leftline, rightline)

    rectangle1 = cv2.rectangle(img4.copy(), (int(x1), int(y1)), (int(x4), int(y4)), (0, 255, 0), 2)
    cv_show('rectangle1', rectangle1)

    # 透视变换
    warped = four_point_transform(img5, p.reshape(4, 2) * ratio)
    warped1 = resize(warped, height=450)
    cv_show("scanned", warped1)  # 透视变换
    cv2.imencode('.jpg', warped1, )[1].tofile(savename)


if __name__ == "__main__":
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

    for filename in os.listdir("test2/"):
        print(filename)
        grabcut_correct("test2/" + filename, "front_cut_image/" + filename)
    # #
    # for filename in os.listdir("failed1/"):
    #     print(filename)
    #     grabcut("failed1/"+filename,"result/"+filename)
    #
    # for filename in os.listdir("id_dingwei/"):
    #     print(filename)
    #     grabcut("id_dingwei/"+filename,"result4/"+filename)

    # for filename in os.listdir("finished/"):
    #     print(filename)
    #     grabcut("finished/" + filename, "result4/" + filename)

