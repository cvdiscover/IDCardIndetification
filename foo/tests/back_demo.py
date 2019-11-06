import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# fileName = "../sfz_back/perspective_transformation_3.jpeg"  # perspective_transformation_
# src = cv2.imread(fileName)
# x, y = src.shape[0:2]
# src = cv2.resize(src, (480, int(480 * x / y)))


def basic_demo(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(image, 0, 100, 15)
    # cv2.imshow("bi_demo", img)
    # cv2.waitKey(0)

    edges = cv2.Canny(img, 10, 150, 20)  # 50是最小阈值,150是最大阈值
    # cv2.imshow('Canny', edges)
    # cv2.waitKey(0)
    return img


def open_demo(image):  # 开操作 腐蚀+膨胀 cv.MORPH_OPEN
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("threshold: %s" % ret)
    # cv2.imshow("threshold", binary)
    # cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("open_result", binary)
    # cv2.waitKey(0)
    # find_contours(binary)
    return binary


def line_detection_demo(image):
    cv2.imshow('line_detection_demo', image)
    cv2.waitKey(0)
    result = src.copy()
    binary = cv2.Canny(image, 10, 150, 20)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=200)
    edge_vertical = []
    edge_standard = []
    if lines is not None:
        for i, line in enumerate(lines[:4]):
            print(line[0])
            x1, y1, x2, y2 = line[0]
            k = (y2 - y1)/(x2 - x1)
            if abs(k) > 1:
                edge_vertical.append(line)
                cv2.line(result, (x1, y1), (x2, y2), (255, 255, 0), 2)
            else:
                edge_standard.append(line)
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if len(edge_vertical) < 2 or len(edge_standard) < 2:
        print("未检测到四条边缘 改为最小外接矩形")
        min_outside(image)
    else:
        points = []
        # def pot(s):
        #     return s[1]
        edge_vertical = sorted(edge_vertical, key=lambda a: (a[0][0]+a[0][2])/2, reverse=True)
        print(" print(edge_vertical) ")
        print(edge_vertical)   # 此处将垂直的线按x轴进行排序区分左右 排前面的是靠右的
        edge_standard = sorted(edge_standard, key=lambda a: (a[0][1]+a[0][3])/2, reverse=True)
        print(" print(edge_standard) ")
        print(edge_standard)   # 此处将水平的线按y轴进行排序区分上下 排前面的是靠下的
        for i, eds in enumerate(edge_standard):
            for j, edv in enumerate(edge_vertical):
                points.append(extend_line_byline(eds[i-1], edv[j-1]))
        cv2.imshow('result', result)
        print(points)
        # points = sorted(points, key=lambda a: a[0], reverse=True)
        # points = sorted(points, key=lambda a: a[1], reverse=True)
        # print("after")
        # print(points)
        perspective_transformation(points)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fit_line(image, src):
    contours, hair = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    rect = cv2.boundingRect(cnts[0])
    mid_point = [rect[0]+rect[2]/2, rect[1]+rect[3]/2]
    # cv2.imshow('line_detection_demo', image)
    cv2.waitKey(0)
    result = src.copy()
    binary = cv2.Canny(image, 10, 150, 20)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=2)
    edge = []
    for i in range(4):  # 这里的edge是一个4*n的数组 0 1 2 3 分别是右左下上的线段集
        edge.append([])
    # edge_vertical_left = []
    # edge_vertical_right = []
    # edge_standard_upper = []
    # dge_standard_down = []
    if lines is not None:
        for i, line in enumerate(lines):
            print(line[0])
            x1, y1, x2, y2 = line[0]
            midx = (x1 + x2) / 2
            midy = (y1 + y2) / 2
            if x1 != x2:
                k = (y2 - y1) / (x2 - x1)
            else:
                k = 999
            if abs(k) > 1:
                if midx <= mid_point[0]:
                    edge[1].append([x1, y1])
                    edge[1].append([x2, y2])
                    # edge_vertical_left.append([x1, y1])
                    # edge_vertical_left.append([x2, y2])
                else:
                    edge[0].append([x1, y1])
                    edge[0].append([x2, y2])
                    # edge_vertical_right.append([x1, y1])
                    # edge_vertical_right.append([x2, y2])
                cv2.line(result, (x1, y1), (x2, y2), (255, 255, 0), 2)
            else:
                if midy <= mid_point[1]:
                    edge[3].append([x1, y1])
                    edge[3].append([x2, y2])
                    # edge_standard_upper.append([x1, y1])
                    # edge_standard_upper.append([x2, y2])
                else:
                    edge[2].append([x1, y1])
                    edge[2].append([x2, y2])
                    # dge_standard_down.append([x1, y1])
                    # dge_standard_down.append([x2, y2])
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    # cv2.imshow("fitline1", result)
    if len(edge[0]) is 0 or len(edge[1]) is 0 or len(edge[2]) is 0 or len(edge[3]) is 0:
        print("未能拟合出四条边缘 改为最小外接矩形")

        return min_outside(image, src)
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

        print("---------------------")
        print(points)

        return perspective_transformation(points, src)
        # output = cv2.fitLine(np.array(edge[1]), cv2.DIST_L2, 0, 0.01, 0.01)
        # k = output[1] / output[0]
        # b = output[3] - k * output[2]
        # print("---------------------")
        # print(k, b)
        # result = cv2.line(src.copy(), (0, b), (500, 500*k+b), (0, 0, 255), 2)
        # cv2.imshow("fitline2", result)


def min_outside(image, src):
    contours, hair = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    m = float(0)
    # i = 0
    # a, b, c, d = int(0), int(0), int(0), int(0)
    result = 0
    points = []
    for c in contours:

        #  矩形边框（boundingRect）
        # x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #  minAreaRect - min Area Rect 最小区域矩形
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)  # 这里的box是一个list 包含了四个顶点
        box = np.int0(box)
        # x, y, w, h = np.int0(box[0]), np.int0(box[1]), np.int0(box[2]), np.int0(box[3])
        image = cv2.drawContours(src, [box], 0, (0, 0, 255), 3)
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        if x1 < 0:
            x1 = 0
        elif y1 < 0:
            y1 = 0
        if width > 0 and hight > 0:
            if m < cv2.contourArea(c):
                m = cv2.contourArea(c)
                result = src[y1:y1 + hight, x1:x1 + width]
                # cv2.imshow("result", result)
                points = cal_point(x1, y1, width, hight)
                # cv2.waitKey(0)
                print(y1, y1 + hight, x1, x1 + width)

    return perspective_transformation(points, src)


def cal_point(x, y, w, h):
    """
    根据左上角点和宽高计算四个点坐标
    :param x: 左上角x
    :param y: 左上角y
    :param w: 宽
    :param h: 高
    :return: 四个点的点集
    """
    points = []
    # p1-右下  p2-左下  p3-右上  p4-左上
    p1 = (x + w, y + h)
    p2 = (x, y + h)
    p3 = (x + w, y)
    p4 = (x, y)
    points.append(p1)
    points.append(p2)
    points.append(p3)
    points.append(p4)
    return points


def perspective_transformation(points, src):
    """
        传入四个点进行透视变换
        :param points: 包含四个点的点集
        :return: 两直线交点
        """
    # box = np.int0(points)
    # image = cv2.drawContours(src.copy(), [pts1], 0, (0, 0, 255), 3)
    # cv2.imshow("box_image", image)
    print("-------------------------------------")
    p1, p2, p3, p4 = points # p1-右下  p2-左下  p3-右上  p4-左上
    print(p1, p2, p3, p4)
    # 原图中书本的四个角点 左上、右上、左下、右下
    pts1 = np.float32([p4, p3, p2, p1])
    print("-------------------------------------AAA")
    print(p4[0] - p1[0], p2[1] - p1[1])
    if p2[1] - p4[1] < p3[0] - p4[0]:
        height = 500
        width = int(height * 1.58)
        print(height, width)
    else:
        width = 500
        height = int(width * 1.58)
        print(height, width)
    # 变换后分别在左上、右上、左下、右下四个点
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行透视变换
    dst = cv2.warpPerspective(src, M, (width, height))
    plt.subplot(121), plt.imshow(src[:, :, ::-1]), plt.title('input')
    plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
    # img[:, :, ::-1]是将BGR转化为RGB
    plt.show()
    # cv2.imshow("output", dst)
    return dst
    # get_information(dst)


def extend_line_byline(line1, line2):
    """
    输入两条直线返回两直线交点
    :param line1: 直线1
    :param line2: 直线2
    :return: 两直线交点
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
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
    if k2 is None:
        x = x3
    else:
        if k1 == k2:
            return False
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    print("x -----   y -----")
    print(int(x), int(y))
    return int(x), int(y)


def extend_line_bykb(k1, b1, k2, b2):
    """
    输入两条直线的斜率和偏移量返回两直线交点
    :param k1: 两条直线的直线斜率
    :param b1: 两条直线的偏移量
    :param k2: 两条直线的直线斜率
    :param b2: 两条直线的偏移量
    :return: 两直线交点
    """
    if k1 is not None and k2 is not None:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    else:
        return False
    y = k1 * x * 1.0 + b1 * 1.0
    print("x -----   y -----")
    print(int(x), int(y))
    return int(x), int(y)


def find_contours(image):
    contours, w1 = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flag = 0
    docCnt = None
    if len(contours) > 0:
        # 按轮廓大小降序排列
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in cnts:
            # 近似轮廓
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 如果我们的近似轮廓有四个点，则确定找到了纸
            if len(approx) == 4:
                docCnt = approx
                flag = 1
                break
    print(docCnt)
    points = []
    for i in range(len(docCnt)):
        points.append(docCnt[i][0])
    if flag == 1:
        perspective_transformation(points)
    else:
        return False


def get_information(result):
    result = cv2.bilateralFilter(result, 0, 100, 5)

    # 阈值设置为80可以有效地去除大部分边框和国徽
    ret, thresh = cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # ret, thresh = cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("thresh o", thresh)
    cv2.waitKey(0)

    plt.hist(thresh.ravel(), 256, [0, 256])
    plt.show()

    canny = cv2.Canny(thresh, 10, 150, 20)  # 50是最小阈值,150是最大阈值
    cv2.imshow('canny1', canny)
    cv2.waitKey(0)

    # 闭操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
    morphologyEx = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernelX)
    # 预览效果
    cv2.imshow('morphologyEx1', morphologyEx)
    cv2.waitKey(0)

    # 膨胀腐蚀
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
    Element = cv2.dilate(morphologyEx, kernelX)
    Element = cv2.dilate(Element, kernelY)
    Element = cv2.erode(Element, kernelX)
    Element = cv2.erode(Element, kernelY)

    # 预览效果
    cv2.imshow('getStructuringElement', Element)
    cv2.waitKey(0)

    # 闭操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphologyEx = cv2.morphologyEx(Element, cv2.MORPH_CLOSE, kernelX)
    # 预览效果
    cv2.imshow('morphologyEx2', morphologyEx)
    cv2.waitKey(0)

    contours, hair = cv2.findContours(morphologyEx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # over = cv2.drawContours(result, contours, -1, (255, 0, 0), 1)
    # cv2.imshow("counters", over)

    for c in contours:

        # 矩形边框（boundingRect）
        if cv2.contourArea(c) < 150:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("over", result)


if __name__ == '__main__':
    # src = cv.resize(src, (500, 500), interpolation=cv.INTER_CUBIC)
    input_dir = "C:/Users/Alexi/Desktop/idcard_info/sfz_back/"
    output_dir = "C:/Users/Alexi/Desktop/idcard_info/sfz_back_result/"

    is_batch = 1
    if is_batch:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for filename in os.listdir(input_dir):
            if len(filename.split(".")) < 2:
                continue
            print(filename)
            path = input_dir + filename
            # 读取图片
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            x, y = img.shape[0:2]
            img = cv2.resize(img, (480, int(480 * x / y)))
            # img = resize(img.copy(), width=500)
            # correct_image(img)
            image = open_demo(basic_demo(img))
            result = fit_line(image, img)
            if result is not None:
                cv2.imencode('.jpg', result)[1].tofile(str(output_dir + filename))


    # cv2.imshow("src", src)
    # image = open_demo(basic_demo(src))
    #
    # # if find_contours(image):
    # fit_line(image)
    # # line_detection_demo(image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
