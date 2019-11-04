# -*- coding: UTF-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import matplotlib.pylab as plt
import dlib
from PIL import Image
from datetime import datetime
from foo.tools.front_correct_skew import correct_skew, resize
from foo.idcard_back_detection import *
from foo.idcard_front_detection import *

# 加载人脸检测模型
classfier = cv2.CascadeClassifier("C:/Users/Alexi/Desktop/IDCard_Identification/haarcascades/haarcascade_frontalface_alt2.xml")
detector = dlib.get_frontal_face_detector()


def batch_process(input_dir="images/", output_dir="output/"):
    """
    批量处理图片
    :param input_dir: 输入文件夹
    :param output_dir: 定位后输出文件夹
    :return:
    """
    if input_dir[-1] != "/":
        input_dir += "/"
    if output_dir[-1] != "/":
        output_dir += "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # is_existed = os.listdir("D:/datasets/project_datasets/idcard/sfz_front_output4")
    for filename in os.listdir(input_dir):
        if len(filename.split(".")) < 2 or (filename.split(".")[-1] != "jpg" and filename.split(".")[-1] != "jpeg" and filename.split(".")[-1] != "png"):
            continue

        print(filename)
        # if filename in is_existed:
        #     continue
        path = input_dir + filename
        save_name = output_dir + filename
        try:
            single_process(path, save_name)
        except Exception as e:
            print(e)


def single_process(path, save_name):
    """
    单张调试
    :return:
    """
    # path = "images/28.jpg"
    start_time = datetime.now()
    # path = "F:/idcard/2/2/images/C370DA4B883D4F12A607938F6B390FBE.JPG"

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img = resize(img.copy(), width=500)
    img = face_detect(img)
    start_time = datetime.now()
    faces = detector(img, 1)
    if len(img.shape) == 3:
        grey = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    else:
        gray = np.array(img)
    # 使用opencv人脸检测模型

    faces_cv = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faces) > 0 and len(faces_cv) > 0:
        face_rect = faces[0]
        faceRects = np.array([[face_rect.left(), face_rect.top(), face_rect.right() - face_rect.left(),
                    face_rect.bottom() - face_rect.top()]])
    else:
        faceRects = np.array([])
    # 判断是否是正面,大于0则检测到人脸,是正面
    # print(len(faceRects))
    facedetect_time = datetime.now()

    imgWidth = 500
    imgHeight = 316
    # img = cv2.resize(img, (imgWidth, imgHeight))
    is_need_correct_skew = 0
    try:
        if len(faceRects) > 0:
            # regions = box_get_front(
            #     copy.deepcopy(img), save_name, imgHeight, imgWidth)
            max_face = faceRects[np.where(faceRects[:, 3] == faceRects[:, 3].max())]
            regions = box_get_front_correction(
                copy.deepcopy(img), save_name, imgHeight, imgWidth, max_face)
            is_need_correct_skew = check_location(img, regions)
        else:
            return
            # box_get_back(img, save_name, imgHeight, imgWidth)
            # is_need_correct_skew = 1
    except Exception as e:
        print("初次定位出错，需要进行纠偏！")
        is_need_correct_skew = 1
    #return

    if is_need_correct_skew == 1:
        if len(faceRects) > 0:

            # max_face = faceRects[np.where(faceRects[:, 3] == faceRects[:, 3].max())]
            # # 根据人脸与整体图片大小的比例判断是否需要纠正
            # if  280/ 900 > max_face_h/img.shape[0]:
            #print(max_face)
            img = correct_skew(img, 1, max_face)
        else:
            img = correct_skew(img, 0)

        # plt.imshow(img,cmap=plt.gray())
        # plt.show()
        correct_skew_time = datetime.now()
        imgWidth = 500
        imgHeight = 316
        img = cv2.resize(img, (imgWidth, imgHeight))

        if len(faceRects) > 0:
            faces = detector(img, 1)
            # plt.imshow(img,cmap=plt.gray())
            # plt.show()
            if len(faces) == 0:
                return

            face_rect = faces[0]

            faceRects = np.array([[face_rect.left(), face_rect.top(), face_rect.right() - face_rect.left(),
                                   face_rect.bottom() - face_rect.top()]])
            max_face = faceRects[np.where(faceRects[:, 3] == faceRects[:, 3].max())]
            try:
                box_get_front_correction(copy.deepcopy(img), save_name, imgHeight, imgWidth, max_face)
            except Exception as e:
                print("正面定位失败！")
        else:
            try:
                box_get_back(img, save_name, imgHeight, imgWidth)
            except Exception as e:
                print("反面定位失败！")
                pass

        locate_time = datetime.now()
        print(locate_time - correct_skew_time, correct_skew_time - facedetect_time, facedetect_time - start_time)
        # box_get_back(img, save_name, imgHeight, imgWidth)


def face_detect(img):

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
            gray = np.array(img_rotation)

        # 使用opencv人脸检测模型
        faces_cv = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        #print(len(faces_dlib),len(faces_cv))
        if len(faces_dlib) == 0 or len(faces_cv) ==0:
            is_front = 0

    # print(len(faces_dlib), len(faces_cv))
    # plt.imshow(img_rotation, cmap=plt.gray())
    # plt.show()
    if is_front:
        return np.array(img_rotation)
    else:
        return img


if __name__ == "__main__":
    is_debug = 0
    is_batch = 0
    if is_batch:
        input_dir = "C:/Users/Alexi/Desktop/idcard_info/sfz_front"
        output_dir = "C:/Users/Alexi/Desktop/idcard_info/sfz_result"
        batch_process(input_dir, output_dir)  # 批量处理
    else:

        img_name = "2f7677bf-691a-4c96-9151-37657e27cf9d.jpeg"
        # img_name = "0d4dc06f-c59c-429b-9034-748e9b00da84.jpeg"
        path = "C:/Users/Alexi/Desktop/idcard_info/sfz_front/"+ img_name
        # img_name = "28"
        # path = "D:/datasets/project_datasets/idcard/images/" + img_name + ".jpg"
        #"1bcad970-a536-41bb-885b-aaed1ba48e38.jpeg"
        save_name = "C:/Users/Alexi/Desktop/IDCard_Identification/output/"+img_name.split(".")[0]+".jpg"
        single_process(path, save_name)  # 单张调试
