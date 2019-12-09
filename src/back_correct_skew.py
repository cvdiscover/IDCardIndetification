from PIL import Image
from src.com.tools import *
from src.config.config import *


def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
    """
    提取波峰
    :param array_vals: 投影数组
    :param minimun_val: 波峰最小值
    :param minimun_range: 波峰最小跨度
    :return:
    """
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val >= minimun_val and start_i is None:
            start_i = i
        elif val >= minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges


def median_split_ranges(peek_ranges):
    new_peek_ranges = []
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    for i, peek_range in enumerate(peek_ranges):
        num_char = int(round(widthes[i]/median_w, 0))
        if num_char > 1:
            char_w = float(widthes[i] / num_char)
            for i in range(num_char):
                start_point = peek_range[0] + int(i * char_w)
                end_point = peek_range[0] + int((i + 1) * char_w)
                new_peek_ranges.append((start_point, end_point))
        else:
            new_peek_ranges.append(peek_range)
    return new_peek_ranges


def cal_rotation_angle(img):
    """
    计算旋转角度
    :param img: 图片
    :return:旋转角度
    """

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


def mark_corner_image(img, point_size = 3):
    """
    标记角点位置
    :param img: 图片
    :param point_size: 标记角点大小
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_mark = gray.copy()
    img_mark[:, :] = 0
    gray = np.float32(gray)

    # 输入图像必须是 float32 ,最后一个参数在 0.04 到 0.05 之间
    corners = cv2.goodFeaturesToTrack(gray, 500, 0.06, 1)

    corners = np.int0(corners)
    corners_list = []
    for i in corners:
        x, y = i.ravel()
        corners_list.append([x, y])
        cv2.circle(img_mark, (x, y), point_size, (255, 255, 255), -1)

    return img_mark


def correct_image(img):
    """
    纠正图片
    :param img: 图片
    :return: 纠正后的图片、（中华人民共和国位置）、（有效日期位置）
    """

    angle = cal_rotation_angle(img)
    if 77 < angle < 103:
        angle = 90
        img_im = Image.fromarray(img)
        img_correction = np.array(img_im.rotate(angle, expand=True))
        if is_debug == 1:
            print('correct：90°')
            cv2.imshow("img_correction", img_correction)
            cv2.waitKey(0)
    elif angle < 30 or angle > 150:
        angle = 0
        print('Already closed correct')
    else:
        print('Already correct')

    img_im = Image.fromarray(img)
    img_correction = np.array(img_im.rotate(angle, expand=True))
    img_mark = mark_corner_image(img_correction)

    from src.com.tools import project

    horizontal_sum = project(img_mark, orientation=0, is_show=1)
    i = 0
    sum = 0
    for hor in horizontal_sum:
        if hor > 0:
            i += 1
            sum += hor
    minimun_val = int(sum/i/1.5)

    peek_ranges = extract_peek_ranges_from_array(horizontal_sum, minimun_val)

    img_mblur = cv2.medianBlur(img_mark, 1)

    elment = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 2))
    img_dilate = cv2.dilate(img_mblur, elment, iterations=2)

    _, contours, _ = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    rects_m = []
    img_filter = img_dilate.copy()
    img_filter[:, :] = 0
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        if (area < 800):
            continue

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.boundingRect(cnt)
        if rect[2] < 150 or rect[3] < 10:
            continue
        rect_m = cv2.minAreaRect(cnt)
        rects.append(rect)
        rects_m.append(rect_m)

    rects = np.array(rects)
    text1_index = np.where(rects[:, 1]==rects[:, 1].min())
    text1 = rects_m[text1_index[0][0]]

    text2_index = np.where(rects[:, 1] == rects[:, 1].max())
    text2 = rects_m[text2_index[0][0]]

    rects = np.array(rects)
    y_mean = rects[:, 1].mean()
    max_rect = rects[np.where(rects[:, 2] == rects[:, 2].max())]

    return np.array(img_correction), text1, text2


def get_lines(img):
    """
    检测图片中直线位置
    :param img: 图片
    :return: 水平线位置、竖直线位置
    """
    img2 = img.copy()
    from src.front_correct_skew import get_border_by_canny
    lines = get_border_by_canny(img)

    # 创建水平和垂直线list
    horizontal, vertical = [], []
    lines_2d_original = lines[:, 0, :]
    lines_2d = []
    for line in lines_2d_original:
        x1, y1, x2, y2 = line
        lines_2d.append([x1, y1, x2, y2])

    from src.front_correct_skew import merge_lines
    lines_2d = merge_lines(lines_2d)
    for line in lines_2d:
        x1, y1, x2, y2 = line
        if abs(x1 - x2) > abs(y1 - y2) and math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 100:
            horizontal.append([x1, y1, x2, y2])
        elif abs(x1 - x2) < abs(y1 - y2) and math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 50:
            vertical.append([x1, y1, x2, y2])

    return horizontal, vertical


def back_correct_skew(img):
    """
    身份证国徽面纠偏
    :param img: 图片
    :return: 纠偏后的图片、边界线位置
    """
    img, text1, text2 = correct_image(img)
    h, w = img.shape[:2]
    horizontal, vertical = get_lines(img)
    # 顶部、底部、左、右边界线位置
    top_line = bottom_line = left_line = right_line = []
    # 计算横线平均倾斜角
    mean__horizontal_angle = (text1[2] + text2[2]) / 2

    prediction_location = [int(text1[0][1]), int(text2[0][1]),  int(text2[0][0] -  0.9 * max(text2[1][0], text2[1][1])), int(text2[0][0] + 0.9 * max(text2[1][0], text2[1][1]))]
    # 计算横线倾斜角度
    from src.front_correct_skew import cal_distance

    d_t_min = h
    d_b_min = h

    for line in horizontal:
        x1, y1, x2, y2 = line
        k = (y1 - y2) / (x1 - x2)
        h = math.atan(k)
        a = math.degrees(h)
        # 获取与估计位置较近的水平线
        if abs(a - text1[2]) < 10 or abs(a - (90 + text1[2])) < 10:
            d = cal_distance(x1, y1, x2, y2, text1[0][0], text1[0][1] - 50)
            if d < 50 and (y1 +y2) / 2 < text1[0][1] - 15 and d < d_t_min:
                d_t_min = d
                top_line = line
        if abs(a - text2[2]) < 10 or abs(a - (90 + text2[2])) < 10:
            d = cal_distance(x1, y1, x2, y2, text2[0][0], text2[0][1] + 20)
            if d < 70 and (y1 +y2) / 2 > text2[0][1] + 15 and d < d_b_min:
                d_b_min = d
                bottom_line = line

    # 计算竖线平均倾斜角
    predict__vertical_angle = 90 - abs(mean__horizontal_angle)
    # 获取与估计位置较近的水平线
    d_l_min = w
    d_r_min = w
    for line in vertical:
        x1, y1, x2, y2 = line
        # 计算竖线倾斜角度
        if x1 - x2 == 0:
            a = 90
        else:
            k = (y1 - y2) / (x1 - x2)
            h = math.atan(k)
            a = math.degrees(h)
        if abs(a -  predict__vertical_angle) < 10 or abs(a -  (90 + predict__vertical_angle)):
            d = cal_distance(x1, y1, x2, y2, text2[0][0] -  0.9 * max(text2[1][0], text2[1][1]), text2[0][1])
            if d < 70 and d < d_l_min:
                d_l_min = d
                left_line = line

            d = cal_distance(x1, y1, x2, y2, text2[0][0] + 0.9 * max(text2[1][0], text2[1][1]), text2[0][1])
            if d < 70 and d < d_r_min:
                d_r_min = d
                right_line = line
    return np.array(img), top_line, bottom_line, left_line, right_line