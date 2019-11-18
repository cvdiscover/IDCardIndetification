import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from PIL import Image
from foo.tools.front_correct_skew import correct_skew, resize
from foo.tools.config import *

# 加载人脸检测模型
classfier = cv2.CascadeClassifier("D:/python/cascades/haarcascade_frontalface_alt2.xml")
detector = dlib.get_frontal_face_detector()

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

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def box_get_front_correction(img,imgHeight, imgWidth, face_rect):
    """
        获取正面文字位置。先定位地址位置，再根据地址位置校正其他信息位置
        :param img: 图片
        :param imgHeight: 图片高度
        :param imgWidth: 图片宽度
        :param face_rect: 人脸位置
        :return: 文字位置
        """

    regions = []

    #精准确定照片位置
    photo_x1 = face_rect[0][0] - 40
    photo_x2 = face_rect[0][0] + face_rect[0][2] + 40
    photo_y1 = face_rect[0][1] - 40
    photo_y2 = face_rect[0][1] + face_rect[0][3] + 60
    photo_cut = img[photo_y1:photo_y2, photo_x1:photo_x2]
    img=cv2.rectangle(img,(photo_x1,photo_y1),(photo_x2,photo_y2),(0,0,255),2)
    if is_debug == 1:
        plt.imshow(img,cmap=plt.gray())
        plt.show()
    return photo_x1,photo_x2,photo_y1,photo_y2

img=cv2.imread('D://datasets//sfz_problem//20.jpeg')
img = resize(img.copy(), width=500)
img = face_detect(img)
faces = detector(img, 1)
if len(img.shape) == 3:
    grey = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
else:
    grey = np.array(img)

faces_cv = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faces) > 0 and len(faces_cv) > 0:
    face_rect = faces[0]
    faceRects = np.array([[face_rect.left(), face_rect.top(), face_rect.right() - face_rect.left(),
                           face_rect.bottom() - face_rect.top()]])
else:
    faceRects = np.array([])

imgWidth = 500
imgHeight = 316
# img = cv2.resize(img, (imgWidth, imgHeight))

if len(faceRects) > 0:
    # regions = box_get_front(
    #     copy.deepcopy(img), save_name, imgHeight, imgWidth)
    max_face = faceRects[np.where(faceRects[:, 3] == faceRects[:, 3].max())]
    (x1,x2,y1,y2) = box_get_front_correction(
        img.copy(), imgHeight, imgWidth, max_face)
    print(x1, x2, y1, y2)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
sqlKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))

tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
cv_show('tophat',tophat)
gradX=cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
gradX=np.absolute(gradX)
(minVal,maxVal)=(np.min(gradX),np.max(gradX))
gradX=(255*((gradX-minVal)/(maxVal-minVal)))
gradX=gradX.astype('uint8')
cv_show('gradX',gradX)

gradX=cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
cv_show('gradX',gradX)
thresh=cv2.threshold(gradX,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

thresh1=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqlKernel)
cv_show('thresh',thresh)

threshCnts=cv2.findContours(thresh1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
cnts=threshCnts

cur_img=img.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show('img',cur_img)

locs=[]
#遍历轮廓
for (i,c) in enumerate(cnts):
    (x,y,w,h)=cv2.boundingRect(c)
    print(w,h)
    ar=w/float(h)

    #选择合适的区域，根据实际任务来，这里的基本上都是四个数字一组
    if ar>8 and ar<30:
        if (w>100 and w<400) :#and (h>10 and h<20):
            #把符合的留下来
            locs.append((x,y,w,h))

locs=sorted(locs,key=lambda x:x[0])
output=[]
print(len(locs))

for (i,(gX,gY,gW,gH)) in enumerate(locs):
    #initialize the list of group digits
    cv2.rectangle(img,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),2)
    print((gX-5,gY-5),(gX+gW+5,gY+gH+5))

cv2.imshow('image',img)
cv2.waitKey(0)
w=gW+10
h=gH+10
print(w,h)
long=2*w
#canny=cv2.Canny(gray,50, 150, apertureSize=3)
#cv_show('canny',canny)
#lines = cv2.HoughLinesP(canny, 1,np.pi/180, 80, long, 5)
#print(lines)
#for x1, y1, x2, y2 in lines[0]:
#    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('result', img)
cv2.waitKey(0)
