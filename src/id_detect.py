# -*- coding: UTF-8 -*-
import sys
import os
from PIL import Image

from src.front_correct_skew import *
from src.idcard_back_detection import *
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
    :param output_dir: 输出地址
    :param input_dir: 输入文件夹
    :param output_dir: 定位后输出文件夹
    :return:
    """
    if input_dir[-1] != "/":
        input_dir += "/"
    if output_dir[-1] != "/":
        output_dir += "/"
    front_output_dir = output_dir + "front/"
    back_output_dir = output_dir + "back/"
    if not os.path.exists(front_output_dir):
        os.makedirs(front_output_dir)
    if not os.path.exists(back_output_dir):
        os.makedirs(back_output_dir)
    for filename in os.listdir(input_dir):
        if len(filename.split(".")) < 2 or (filename.split(".")[-1] != "jpg" and filename.split(".")[-1] != "jpeg" and filename.split(".")[-1] != "png"):
            continue
        print("#############################\n")
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
    origimg = copy.deepcopy(img)
    orig_img = resize(origimg, height = 506)

    # 人脸检测
    img = face_detect(origimg)  # 改为origimg
    faces = detector(img, 2)
    if len(img.shape) == 3:
        grey = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    else:
        grey = np.array(img)

    # 国徽检测
    ret = national_emblem_judgement(img.copy())[0]
    if ret is True:
        img = resize(orig.copy(), width=500)
        ret = pre_fitline_get_back(img.copy(), back_save_name)
        if len(faces) == 0 and ret is False:
            img = correct_skew(img.copy(), 0)
            marked_image = find_information(img)
            if marked_image is None:
                ret = False
                print("反面定位失败1！")
                # box_get_back(img, back_save_name, 316, 500)
            else:
                cv2.imencode('.jpg', marked_image)[1].tofile(str(back_save_name))
                return
        else:
            ret = False
            print("反面定位失败2!")
    else:
        try:
            if len(faces) == 0:
                img = correct_skew(img.copy(), 0)
                marked_image = find_information(img)
                if marked_image is None:
                    ret = False
                    print("反面定位失败3！")
                    # box_get_back(img, back_save_name, 316, 500)
                else:
                    cv2.imencode('.jpg', marked_image)[1].tofile(str(back_save_name))
                    return
            else:
                ret = False
                print("反面定位失败4!")
        except Exception as e:
            ret = False
            print("反面定位失败5！")

    if ret is False:
        faces_cv = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faces) > 0 or len(faces_cv) > 0:
            try:
                face_rect = faces[0]
                faceRects = np.array([[face_rect.left(), face_rect.top(), face_rect.right() - face_rect.left(),
                            face_rect.bottom() - face_rect.top()]])
                print('faceRects', faceRects)
            # 这里报错指的是dlib未检测到人脸，但是detectMultiScale分类器检测到了
            except Exception as e:
                face_rect = faces_cv[0]
                print('face_rect', face_rect)
                faceRects = np.array([[face_rect[0], face_rect[1], face_rect[2], face_rect[3]]])
        else:
            try:
                img = rotation_img(orig_img)
                img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                rects = detector(img1, 2)
                face_rect = rects[0]
                faceRects = np.array([[face_rect.left(), face_rect.top(), face_rect.right() - face_rect.left(),
                                       face_rect.bottom() - face_rect.top()]])
            except Exception as e:
                faceRects = np.array([])
                print('初次人脸检测失败')

        print('len(faceRects)',len(faceRects))

        imgWidth = 500
        imgHeight = 316

        is_need_correct_skew = 0
        try:
            if len(faceRects) > 0:
                max_face = faceRects[np.where(faceRects[:, 3] == faceRects[:, 3].max())]
                regions = box_get_front_correction(
                    copy.deepcopy(img), front_save_name, imgHeight, imgWidth, max_face)
                is_need_correct_skew = check_location(img, regions)
        except Exception as e:
            print("初次定位出错，需要进行纠偏！")
            is_need_correct_skew = 1

        # 需要进行纠偏，纠偏完了后进行信息的定位
        if is_need_correct_skew == 1:
            # 根据人脸与整体图片大小的比例判断是否需要纠正
            img = correct_skew(img, 1, max_face)
            print('len(img.shape)', len(img.shape))
        # 纠偏失败(纠偏失败返回的是灰度图，成功返回的是纠偏完后的图)

            if len(img.shape) == 2:
                img = front_correct_skew_after_failed(orig_img)
                imgWidth1 = 800
                imgHeight2 = 506
                img = cv2.resize(img, (imgWidth1, imgHeight2))

                faces = detector(img, 2)
                faces1 = classfier.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                if is_debug == 1:
                    plt.imshow(img,cmap=plt.gray())
                    plt.show()

                if len(faces) == 0 and len(faces1) ==0:
                    try:
                        print('人脸检测失败，二次纠偏中。。。')
                        img = front_correct_skew_after_failed(orig_img)
                        img = cv2.resize(img, (800, 506))
                    except Exception as e:
                        print('人脸检测失败')
                try:
                    box_get_front_message(img, front_save_name, orig_img)
                except Exception as e:
                    try:
                        print('进入第二次纠偏定位')
                        # 采用第二种旋转方案
                        img_r = rotation_img(orig_img)
                        img_skew = front_correct_skew_after_failed(img_r)
                        img = cv2.resize(img_skew, (800, 506))
                        box_get_front_message(img, front_save_name, orig_img)
                    except Exception as e:
                        print("正面定位失败！")

            elif len(img.shape) == 3:
                imgWidth1 = 800
                imgHeight2 = 506
                img = cv2.resize(img, (imgWidth1, imgHeight2))

                faces = detector(img, 2)
                faces1 = classfier.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                if is_debug == 1:
                    plt.imshow(img,cmap=plt.gray())
                    plt.show()
                if len(faces) == 0 and len(faces1) ==0:
                    try:
                        print('人脸检测失败，二次纠偏中。。。')
                        img = front_correct_skew_after_failed(orig_img)
                        img = cv2.resize(img, (800, 506))
                    except Exception as e:
                        print('人脸检测失败')
                try:
                    box_get_front_message(img, front_save_name, orig_img)
                except Exception as e:
                    try:
                        print('进入第二次纠偏定位')
                        # 采用第二种旋转方案
                        img_r = rotation_img(orig_img)
                        img_skew = front_correct_skew_after_failed(img_r)
                        img = cv2.resize(img_skew, (800, 506))
                        box_get_front_message(img, front_save_name, orig_img)
                    except Exception as e:
                        print("正面定位失败！")





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
    faces_dlib = detector(np.array(img_rotation), 2)

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
        faces_dlib = detector(np.array(img_rotation), 2)

        if len(img.shape) == 3:
            grey = cv2.cvtColor(np.array(img_rotation), cv2.COLOR_BGR2GRAY)
        else:
            grey = np.array(img_rotation)

        # 使用opencv人脸检测模型
        faces_cv = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faces_dlib) == 0 and len(faces_cv) == 0:
            is_front = 0
    if is_front:
        return np.array(img_rotation)
    else:
        return img


if __name__ == "__main__":

    # 批量处理
    if is_batch:
        batch_process(input_dir, output_dir)

    # 单张处理
    else:
        img_name = img_name
        path = path_without_img_name + img_name
        if single_output_dir[-1] != "/":
            single_output_dir += "/"
        if not os.path.exists(single_output_dir):
            os.makedirs(single_output_dir)
        save_name = single_output_dir + img_name.split(".")[0]+".jpg"
        single_process(path, save_name, save_name)
