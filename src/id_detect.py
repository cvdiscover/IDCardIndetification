# -*- coding: UTF-8 -*-
import sys
import os
from PIL import Image

from src.front_correct_skew import correct_skew, resize
from src.idcard_back_detection import pre_fitline_get_back, box_get_back
from src.idcard_front_detection import *
from src.config.config import *

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# 加载人脸检测模型
classfier = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_alt2.xml")
detector = dlib.get_frontal_face_detector()


def batch_process(input_dir="images/", output_dir="output/"):
    """
    批量处理图片
    :param front_output_dir: 正面定位后输出文件夹
    :param back_output_dir: 反面定位后输出文件夹
    :param input_dir: 输入文件夹
    :return:
    """
    if input_dir[-1] != "/":
        input_dir += "/"
    if output_dir[-1] != "/":
        output_dir += "/"
    # if back_output_dir[-1] != "/":
    #     back_output_dir += "/"
    front_output_dir = output_dir + "front/"
    back_output_dir = output_dir + "back/"
    if not os.path.exists(front_output_dir):
        os.makedirs(front_output_dir)
    if not os.path.exists(back_output_dir):
        os.makedirs(back_output_dir)
    for filename in os.listdir(input_dir):
        if len(filename.split(".")) < 2 or (filename.split(".")[-1] != "jpg" and filename.split(".")[-1] != "jpeg" and filename.split(".")[-1] != "png"):
            continue

        print(filename)
        path = input_dir + filename
        front_save_name = front_output_dir + filename
        back_save_name = back_output_dir + filename
        try:
            single_process(path, front_save_name, back_save_name)
        except Exception as e:
            print(e)


def single_process(path, front_save_name, back_save_name):
    """
    单张调试
    :param back_save_name: 反面存储地址
    :param front_save_name: 正面存储地址
    :param path:文件地址
    :return:
    """

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    orig = img.copy()
    img = resize(orig, width=500)
    img = face_detect(img)
    faces = detector(img, 1)

    if len(img.shape) == 3:
        grey = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    else:
        grey = np.array(img)

    # 使用opencv人脸检测模型

    # 人脸框位置
    faces_cv = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faces) > 0 and len(faces_cv) > 0:
        face_rect = faces[0]
        faceRects = np.array([[face_rect.left(), face_rect.top(), face_rect.right() - face_rect.left(),
                    face_rect.bottom() - face_rect.top()]])
    else:
        faceRects = np.array([])
    # 判断是否是正面,大于0则检测到人脸,是正面
    # print(len(faceRects))

    imgWidth = 500
    imgHeight = 316

    is_need_correct_skew = 0
    try:
        if len(faceRects) > 0:
            max_face = faceRects[np.where(faceRects[:, 3] == faceRects[:, 3].max())]
            regions = box_get_front_correction(
                copy.deepcopy(img), front_save_name, imgHeight, imgWidth, max_face)
            is_need_correct_skew = check_location(img, regions)
        else:
            img = resize(orig.copy(), width=500)
            if not pre_fitline_get_back(img.copy(), back_save_name):
                is_need_correct_skew = 1

    except Exception as e:
        print("初次定位出错，需要进行纠偏！")
        is_need_correct_skew = 1

    # 需要进行纠偏，纠偏完了后进行信息的定位
    if is_need_correct_skew == 1:
        if len(faceRects) > 0:
            # # 根据人脸与整体图片大小的比例判断是否需要纠正
            img = correct_skew(img, 1, max_face)
        else:
            img = correct_skew(img, 0)

        imgWidth = 800
        imgHeight = 506
        img = cv2.resize(img, (imgWidth, imgHeight))

        if len(faceRects) > 0:
            faces = detector(img, 1)
            # plt.imshow(img,cmap=plt.gray())
            # plt.show()
            if len(faces) == 0:
                return
            try:
                box_get_front_message(img, front_save_name)
            except Exception as e:
                print("正面定位失败！")
        else:
            try:
                box_get_back(img, back_save_name, 316, 500)
            except Exception as e:
                print("反面定位失败！")
                pass


def face_detect(img):

    # angle范围在-90->90度，纠正-90->90度的偏差
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

    # 使用opencv人脸检测模型
    faces_cv = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

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
        if len(faces_dlib) == 0 or len(faces_cv) == 0:
            is_front = 0
    if is_front:
        return np.array(img_rotation)
    else:
        return img


if __name__ == "__main__":

    # 0-单张处理； 1-批量处理
    is_batch = 0

    # 批量处理
    if is_batch:
        batch_process(input_dir, output_dir)

    # 单张处理
    else:
        img_name = img_name
        path = path_without_img_name + img_name
        if not os.path.exists(path_without_img_name):
            os.makedirs(path_without_img_name)
        save_name = single_output_dir + img_name.split(".")[0]+".jpg"
        single_process(path, save_name, save_name)
