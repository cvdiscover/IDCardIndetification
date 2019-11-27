from src.back_correct_skew import *
from src.config.config import *
from src.com.tools import *


def box_get_back(img, save_name, imgHeight, imgWidth):
    """
        获取反面文字位置
        :param img: 图片
        :param imgHeight: 图片高度
        :param imgWidth: 图片宽度
        :param index: 图片序号
        :return: 文字切片

        """
    regions = []
    # 签发机关
    issuing_authority_height = int(imgHeight / 1.5)
    issuing_authority_width = int(imgWidth / 2.5)
    issuing_authority_addHeight = int(imgHeight / 1.5 + imgHeight / 6)
    issuing_authority_addWidth = int(imgWidth / 2.5 + imgWidth / 2)
    img_issuing_authority = img[issuing_authority_height:issuing_authority_addHeight,
                            issuing_authority_width:issuing_authority_addWidth]
    # plt.imshow(img, cmap=plt.gray())
    # plt.show()
    scale = int(img.shape[1] / 500)
    # print(scale, img.shape)
    issuing_authority_region = get_regions(img_issuing_authority, scale, is_front = 0)
    issuing_authority_region[0][0] += issuing_authority_width
    issuing_authority_region[0][1] += issuing_authority_height
    regions.append(issuing_authority_region)
    # 有效日期
    effective_date_height = int(imgHeight / 1.25)
    effective_date_width = int(imgWidth / 2.5)
    effective_date_addHeight = int(imgHeight / 1.25 + imgHeight / 6)
    effective_date_addWidth = int(imgWidth / 2.5 + imgWidth / 2)
    img_effective_date = img[effective_date_height:effective_date_addHeight,
                         effective_date_width:effective_date_addWidth]
    # plt.imshow(img_effective_date, cmap=plt.gray())
    # plt.show()
    scale = int(img.shape[1] / 500)
    # print(scale, img.shape)
    effective_date_region = get_regions(img_effective_date, scale, is_front = 0)
    effective_date_region[0][0] += effective_date_width
    effective_date_region[0][1] += effective_date_height
    regions.append(effective_date_region)

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
            cv2.drawContours(img_copy, np.array([box]), 0, (0, 255, 0), 2)
    if is_debug == 1:
        plt.imshow(img_copy, cmap=plt.gray())
        plt.show()
    # plt.imshow(img_copy, cmap=plt.gray())
    # plt.show()
    cv2.imencode('.jpg', img_copy)[1].tofile(str(save_name))


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


# 通过国徽位置预估边界拟合直线完成
def pre_fitline_get_back(src, save_name):
    img, text1, text2 = correct_image(src.copy())
    result, dst = flann_univariate_matching(img.copy())
    if result is not None and dst is not None:
        if dst[0][0][1] > dst[1][0][1]:
            result, dst = flann_univariate_matching(cv2.flip(cv2.flip(img.copy(), 1), 0))
        if result is not None and dst is not None:
            if len_judge(result, dst):
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
                        cv2.imencode('.jpg', marked_image)[1].tofile(str(save_name))
                    return True
                except Exception as E:
                    print(E)
            else:
                print(save_name + "标记点不准确")
                return False


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

    thresh = cv2.adaptiveThreshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    if is_debug == 1:
        cv2.imshow("thresh o", thresh)
        cv2.waitKey(0)
        plt.hist(thresh.ravel(), 256, [0, 256])
        plt.show()

    if len(contours) > 150:
        thresh = cv2.bilateralFilter(result, 0, 80, 5)

    canny = cv2.Canny(thresh, 10, 150, 20)  # 50是最小阈值,150是最大阈值
    if is_debug == 1:
        cv2.imshow('canny1', canny)
        cv2.waitKey(0)

    # 开操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    canny = cv2.dilate(canny, kernelX)
    morphologyEx = open_demo(canny)

    # 闭操作
    # kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    # morphologyEx = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernelX)

    if is_debug == 1:
        cv2.imshow('morphologyEx1', morphologyEx)
        cv2.waitKey(0)

    # 膨胀腐蚀
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    Element = cv2.dilate(morphologyEx, kernelX)
    Element = cv2.dilate(Element, kernelY)
    Element = cv2.erode(Element, kernelX)
    Element = cv2.erode(Element, kernelY)

    if is_debug == 1:
        cv2.imshow('getStructuringElement', Element)
        cv2.waitKey(0)

    # 闭操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    morphologyEx = cv2.morphologyEx(Element, cv2.MORPH_CLOSE, kernelX)

    if is_debug == 1:
        cv2.imshow('morphologyEx2', morphologyEx)
        cv2.waitKey(0)

    _, contours, hair = cv2.findContours(morphologyEx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangle = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 矩形边框（boundingRect）
        if cv2.contourArea(c) < 2000:
            continue
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

    contours, hair = cv2.findContours(morphologyEx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
def open_demo(image):
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

    # store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
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

    # 四条边顺序分别是12，23，34，41 即kb分别为左 下 右 上 四条边的斜率和偏差
    for j in range(4):
        length.append(math.sqrt((gh_points[(j + 1) % 4][0] - gh_points[j % 4][0]) ** 2
                                + (gh_points[(j + 1) % 4][1] - gh_points[j % 4][1]) ** 2))
        if gh_points[(j + 1) % 4][0] == gh_points[j % 4][0]:
            k = "max"
            b_set.append(gh_points[j % 4][0])
        else:
            k = (gh_points[(j + 1) % 4][1] - gh_points[j][1]) / (gh_points[(j + 1) % 4][0] - gh_points[j][0])
            b_set.append(gh_points[j][1] - k * gh_points[j][0])
        k_set.append(k)

    # 预估四个点的位置相对国徽左上角点分别为左 下 右 上
    for j in range(1, 5):
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
        # p1-右下  p2-左下  p3-右上  p4-左上
        p1, p2, p3, p4 = points
    else:
        # p1-右下  p2-左下  p3-右上  p4-左上
        p2, p1, p3, p4 = points
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
    return distance


# 直线拟合找近似边缘
def fit_line(image, fit_points, ori):
    """
    直线拟合找近似边缘
    :param fit_points: 预估的四个顶点 顺序为左下 右下 右上 左上
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

    # 这里的edge是一个4*n的数组 0 1 2 3 分别是右左下上的线段集
    edge = []
    for i in range(4):
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
                    d1 = cal_distance(fit_points[3][0], fit_points[3][1], fit_points[0][0], fit_points[0][1], x1, y1)
                    d2 = cal_distance(fit_points[3][0], fit_points[3][1], fit_points[0][0], fit_points[0][1], x2, y2)
                    if d1 < 30 and d2 < 30:
                        edge[1].append([x1, y1])
                        edge[1].append([x2, y2])
                else:
                    d1 = cal_distance(fit_points[1][0], fit_points[1][1], fit_points[2][0], fit_points[2][1], x1, y1)
                    d2 = cal_distance(fit_points[1][0], fit_points[1][1], fit_points[2][0], fit_points[2][1], x2, y2)
                    if d1 < 30 and d2 < 30:
                        edge[0].append([x1, y1])
                        edge[0].append([x2, y2])
                cv2.line(result, (x1, y1), (x2, y2), (255, 255, 0), 2)
            else:
                if midy <= mid_point[1]:
                    d1 = cal_distance(fit_points[2][0], fit_points[2][1], fit_points[3][0], fit_points[3][1], x1, y1)
                    d2 = cal_distance(fit_points[2][0], fit_points[2][1], fit_points[3][0], fit_points[3][1], x2, y2)
                    if d1 < 30 and d2 < 30:
                        edge[3].append([x1, y1])
                        edge[3].append([x2, y2])
                else:
                    d1 = cal_distance(fit_points[1][0], fit_points[1][1], fit_points[0][0], fit_points[0][1], x1, y1)
                    d2 = cal_distance(fit_points[1][0], fit_points[1][1], fit_points[0][0], fit_points[0][1], x2, y2)
                    if d1 < 30 and d2 < 30:
                        edge[2].append([x1, y1])
                        edge[2].append([x2, y2])
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if is_debug == 1:
        for i in range(4):
            print(edge[i])
        cv2.imshow("fitline123", result)
        cv2.waitKey(0)

    if len(edge[0]) is 0 or len(edge[1]) is 0 or len(edge[2]) is 0 or len(edge[3]) is 0:
        print("未能拟合出四条边缘 改为使用预估位置")
        return None
        # min_outside(image)
    else:
        k_set = []
        b_set = []
        fit_points = []

        # 0 1 2 3 分别是左右上下的线段集
        for i in range(len(edge)):
            output = cv2.fitLine(np.array(edge[i]), cv2.DIST_L2, 0, 0.01, 0.01)
            k_set.append(output[1] / output[0])
            b_set.append(output[3] - (output[1] / output[0]) * output[2])
        for i in range(2, 4):
            for j in range(0, 2):
                fit_points.append(extend_line_bykb(k_set[i], b_set[i], k_set[j], b_set[j]))

        return fit_points


# 选取信息蒙版
def image_select(after_cut_image):
    """
    :param after_cut_image: 裁剪后视为准确的图像
    :return: 上版部分置为白色消除部分干扰
    """
    H, W = after_cut_image.shape[:2]
    ymin, ymax, xmin, xmax = int(H / 2), 480, 0, W
    after_cut_image = after_cut_image[ymin: ymax, xmin: xmax]
    if is_debug == 1:
        cv2.imshow("after_cut_image", after_cut_image)
        cv2.waitKey(0)

    return after_cut_image, int(H / 2)


# 通过求dst四个点组成矩形的四条边长度从而进行国徽检测算法的适用性判断
def len_judge(img, dst):
    """
    通过求dst四个点组成矩形的四条边长度从而进行国徽检测算法的适用性判断
    :param img: 图像
    :param dst: 定位国徽矩形的四个点坐标，dst[0][0][0]-左上角点宽，dst[0][0][1]-左上角点高，
                                      dst[1][0][0]-左下角点宽，dst[1][0][1]-左下角点宽
                                     2-右下角点，       3-右上角点
    :return: 适用返回True, 不适用返回False
    """
    len_max = (max(img.shape[:2]) * 129 / 636) * 1.3
    len_min = (max(img.shape[:2]) * 129 / 636) * 0.3

    for i in range(3):
        dst_len = math.sqrt(
            (dst[(i + 1) % 4][0][0] - dst[i % 4][0][0]) ** 2 + (dst[(i + 1) % 4][0][1] - dst[i % 4][0][1]) ** 2)
        if dst_len > len_max or dst_len < len_min:
            return False
    return True


