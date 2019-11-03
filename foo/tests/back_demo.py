import cv2
import numpy as np
import matplotlib.pyplot as plt

fileName = "C:/Users/Alexi/Desktop/idcard_info/sfz_back/11c84a94-023f-48e1-8ca9-babf7ff69327.jpeg"
src = cv2.imread(fileName)
x, y = src.shape[0:2]
src = cv2.resize(src, (480, int(480 * x / y)))


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
            points.append(extend_line(eds[i-1], edv[j-1]))
    cv2.imshow('result', result)
    perspective_transformation(points)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perspective_transformation(points):
    p1, p2, p3, p4 = points # p1-右下  p2-左下  p3-右上  p4-左上
    # 原图中书本的四个角点 左上、右上、左下、右下
    pts1 = np.float32([p4, p3, p2, p1])
    if p2[1] - p4[1] < p3[0] - p4[0]:
        height = 500
        # width = 500
        width = int(height * 1.58)
        print(height, width)
    else:
        # height = 500
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
    cv2.imshow("output", dst)


def extend_line(line1, line2):

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
    if k2 == None:
        x = x3
    else:
        if k1 == k2:
            return False
        x = (b2 - b1) * 1.0 / (k1 - k2)
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
    if flag == 1:
        perspective_transformation(docCnt)


if __name__ == '__main__':
    # src = cv.resize(src, (500, 500), interpolation=cv.INTER_CUBIC)

    cv2.imshow("src", src)
    image = open_demo(basic_demo(src))
    line_detection_demo(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()