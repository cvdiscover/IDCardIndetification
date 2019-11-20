import cv2
import matplotlib.pyplot as plt
import dlib
import matplotlib.patches as mpatches
from skimage import io,draw,transform,color
import numpy as np


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):#插值方法选用基于局部像素的重采样
    dim = None
    (h, w) = image.shape[:2]  #image.shape[0]指图片长（垂直尺寸），image.shape[1]指图片宽（水平尺寸）
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)  #h为真实高度，height为设定高度0
        dim = (int(w * r), height)  #dim括号内的是按比例调整后的长和宽
    else:
        r = width / float(w)  #为真实宽度，width为设定宽度
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)  #interpolation为插值方法，缩写inter
    return resized


def cv_show(name, img):
    cv2.imshow(name,img)
    cv2.waitKey(0)


# 开操作 腐蚀+膨胀 cv.MORPH_OPEN
def open_demo(image):
    print(image.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("open_result", binary)
    return binary


# 闭操作 膨胀+腐蚀 cv.MORPH_CLOSE
def close_demo(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("close_result", binary)
    return binary


def getpart_info(part):

    result = cv2.bilateralFilter(part.copy(), 0, 50, 5)
    cv2.imshow("bilateralFilter", result)
    cv2.waitKey(0)

    thresh = cv2.adaptiveThreshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 5)
    # cv2.imshow("thresh o", thresh)
    # cv2.waitKey(0)

    # 开操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    morphologyEx = cv2.dilate(thresh, kernelX)
    morphologyEx = open_demo(morphologyEx)

    # cv2.imshow('morphologyEx1', morphologyEx)
    # cv2.waitKey(0)

    # 膨胀腐蚀
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    Element = cv2.dilate(morphologyEx, kernelX)
    Element = cv2.dilate(Element, kernelY,iterations=2)
    Element = cv2.erode(Element, kernelX)
    Element = cv2.erode(Element, kernelY)

    cv2.imshow('getStructuringElement', Element)
    cv2.waitKey(0)

    _, contours, hair = cv2.findContours(Element, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def box_get_front_correction(img,addX,addY ,imgHeight,imgWidth,faceLeft,name_y,sex_y,birth_year_y):
    '''
    根据地址的坐标 定位其他信息的位置
    :param img: 输入图片
    :param addX: 地址的x坐标
    :param addY: 地址的y坐标
    :param imgHeight: 图片高度
    :param imgWidth: 图片宽度
    :param faceLeft: 人脸左边坐标
    :return: 文字的位置
    '''
    regions = []

    # 地址
    address_addHeight = int(addY + imgHeight / 2.8)
    address_addWidth = faceLeft
    address = img[addY:address_addHeight, addX:address_addWidth]
    img = cv2.rectangle(img, (addX, addY), (address_addWidth, address_addHeight), (0, 0, 255), 2)

    # 名字
    name_x = addX
    name_y = name_y # int(addY - imgHeight/1.65)
    name_addHeight = int(name_y + 46) + 10
    name_addWidth = int(name_x + imgWidth / 4)
    name = img[name_y:name_addHeight, name_x:name_addWidth]
    img = cv2.rectangle(img, (name_x, name_y), (name_addWidth, name_addHeight), (0, 0, 255), 2)

    # 性别
    sex_x= addX
    sex_y= sex_y # int(addY - imgHeight/2.65)
    sex_addHeight = int(sex_y + imgHeight / 7.71)
    sex_addWidth = int(sex_x + 45)
    sex = img[sex_y:sex_addHeight, sex_x:sex_addWidth]
    img = cv2.rectangle(img, (sex_x, sex_y), (sex_addWidth, sex_addHeight), (0, 0, 255), 2)

    # 民族
    nationality_x = addX + 160
    nationality_y = sex_y
    nationality_addHeight = int(nationality_y + imgHeight / 7.65)
    nationality_addWidth = int(nationality_x + 50)
    nationality = img[nationality_y:nationality_addHeight, nationality_x:nationality_addWidth]
    img = cv2.rectangle(img, (nationality_x, nationality_y), (nationality_addWidth, nationality_addHeight), (0, 0, 255),
                        2)

    # 出生 年
    birth_year_x = addX
    birth_year_y = birth_year_y # int(addY - imgHeight/4.6)
    birth_year_addHeight = int(birth_year_y + imgHeight / 7.71)
    birth_year_addWidth = int(birth_year_x + 75)
    birth_year = img[birth_year_y:birth_year_addHeight, birth_year_x:birth_year_addWidth]
    img = cv2.rectangle(img, (birth_year_x, birth_year_y), (birth_year_addWidth, birth_year_addHeight), (0, 0, 255), 2)

    #出生 月
    birth_month_x = birth_year_x + 123
    birth_month_y = birth_year_y
    birth_month_addHeight = int(birth_month_y + imgHeight / 7.6)
    birth_month_addWidth = int(birth_month_x + 40)
    birth_month = img[birth_month_y:birth_month_addHeight, birth_month_x:birth_month_addWidth]
    img = cv2.rectangle(img, (birth_month_x, birth_month_y), (birth_month_addWidth, birth_month_addHeight), (0, 0, 255),
                        2)

    #出生 日
    birth_day_x = birth_year_x + 190
    birth_day_y = birth_year_y
    birth_day_addHeight = int(birth_day_y + imgHeight / 7.6)
    birth_day_addWidth = int(birth_day_x + 43)
    birth_day = img[birth_day_y:birth_day_addHeight, birth_day_x:birth_day_addWidth]
    img = cv2.rectangle(img, (birth_day_x, birth_day_y), (birth_day_addWidth, birth_day_addHeight), (0, 0, 255), 2)

#对数组进行排序
def findSmallest(arr):
    smallest = arr[0]  # 将第一个元素的值作为最小值赋给smallest
    smallest_index = 0  # 将第一个值的索引作为最小值的索引赋给smallest_index
    for i in range(1, len(arr)):
        if arr[i] < smallest:  # 对列表arr中的元素进行一一对比
            smallest = arr[i]
            smallest_index = i
    return smallest_index


def selectionSort(arr):
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)  # 一共要调用5次findSmallest
        newArr.append(arr.pop(smallest))  # 每一次都把findSmallest里面的最小值删除并存放在新的数组newArr中
    return newArr

if __name__ == '__main__':


    detector = dlib.get_frontal_face_detector()
    image = io.imread("D://front_cut_image//a3eb0fe5-c517-409f-9b3b-139a30e6bae7.jpeg")
    image = resize(image, width=800)
    image_copy = image.copy()
    dets = detector(image, 2)  # 使用detector进行人脸检测 dets为返回的结果
    # 将识别的图像可视化
    plt.figure()
    ax = plt.subplot(111)
    ax.imshow(image)
    plt.axis("off")

    predictor = dlib.shape_predictor("D://shape_predictor_5_face_landmarks.dat")

    left = dets[0].left()
    top = dets[0].top()
    right = dets[0].right()
    bottom = dets[0].bottom()
    Wwidth=right - left
    Hheigt=top - bottom
    # rect=cv2.rectangle(image,(left,bottom), (left+Wwidth, bottom+Hheigt),(0,255,0),2)
    # cv_show('rect',rect)

    # 照片的位置（不怎么精确）

    width = right - left
    high = top - bottom
    left2 = np.uint(left - 0.3*width)
    bottom2 = np.uint(bottom + 0.6*width)

    img = cv2.rectangle(image,(left2,bottom2),(left2+2*width,bottom2+2*high),(0,0,255),2)

    top2 = np.uint(bottom2 + 1.8 * high)
    right2 = np.uint(left2 + 1.6 * width)
    face_img = image[top2:bottom2, left2:right2, :]
    # cv_show('face_img',face_img)

    face_gray = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
    ret = cv2.threshold(face_gray,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)[1]
    # cv_show('ret',ret)
    erode = cv2.erode(ret,(9,9),iterations=3)
    dilate = cv2.dilate(erode,(9,9),iterations=3)
    cv_show('dilate',dilate)
    # rets = cv2.findContours(dilate.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    # cv2.drawContours(face_img,rets,-1,(0,255,0),2)
    # cv_show('face_img',face_img)
    # rets = sorted(rets, key=cv2.contourArea, reverse=True)[0]
    # for i in range(len(rets)):
    #     arclen = cv2.arcLength(rets[i], True)
    #     epsilon = max(3, int(arclen * 0.02))  # 拟合出的多边形与原轮廓最大距离，可以自己设置，这里根据轮廓周长动态设置
    #     approx = cv2.approxPolyDP(rets[i], epsilon, False)  # 轮廓的多边形拟合
    #     area = cv2.contourArea(rets[i])  # 计算面积
    #     rect = cv2.minAreaRect(rets[i])
    # box = np.int0(cv2.boxPoints(rect))  # 计算最小外接矩形顶点
    # print(box)
    # cv2.drawContours(img, [box], 0, (255, 0, 0), 2)


    cv2.imshow('img',img)
    cv2.waitKey(0)
    # rect = mpatches.Rectangle((left2,bottom2), 1.6*width, 1.8*high,
    #                           fill=False, edgecolor='blue', linewidth=1)
    #
    # ax.add_patch(rect)
    # plt.show()
    '''
    找身份证号码的位置
    '''
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
    sqlKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))

    tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,sqlKernel)
    cv_show('tophat',tophat)
    gradX = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
    gradX = np.absolute(gradX)
    (minVal,maxVal) = (np.min(gradX),np.max(gradX))
    gradX = (255*((gradX-minVal)/(maxVal-minVal)))
    gradX = gradX.astype('uint8')
    # cv_show('gradX1',gradX)

    gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,sqlKernel)
    # cv_show('gradX2',gradX)

    thresh=cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # cv_show('thresh',thresh)

    thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,rectKernel,iterations=1)
    cv_show('thresh',thresh)

    threshCnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts=threshCnts

    cur_img=img.copy()
    cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
    # cv_show('img',cur_img)

    locs=[]
    #遍历轮廓
    for (i,c) in enumerate(cnts):
        (x,y,w,h)=cv2.boundingRect(c)
        ar =w/float(h)
        print(ar,w)
        # 选择合适的区域，根据实际任务来，这里的基本上都是四个数字一组
        if ar>5 and ar<40:
            if (w>350 and w<600) :
                # 把符合的留下来
                locs.append((x,y,w,h))

    locs=sorted(locs,key=lambda x:x[0])
    # output=[]

    for (i,(gX,gY,gW,gH)) in enumerate(locs):
        # initialize the list of group digits
        cv2.rectangle(img,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),2)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    # ax.add_patch(rect)
    # plt.show()

    result_img=img.copy()
    '''
     框出信息区域
    '''
    Xwidth = gW+10
    Yheight = gH+10
    long = 2*Xwidth
    X1 = 2*width
    Y1 = 2*high
    left3 = left2-(long-X1)

    if left3 > 0:
        left3 = left3
        top2 = bottom2 + 2 * high - 60
        point1 = (left3, bottom2)
        point2 = (left2, top2)
    else:
        left3 = 90
        top2 = bottom2 + 2 * high - 60
        point1 = (left3, bottom2)
        point2 = (left2, top2)

    img_copy = img.copy()
    cv2.rectangle(img, point1, point2, (0, 0, 255), 2)
    cv2.imshow('result',img)
    cv2.waitKey(0)
    part = image_copy[top2:bottom2,left3:left2+15]
    print(part.shape[:2])
    cv2.imshow('part', part)
    cv2.waitKey(0)

    retCnts = getpart_info(part)

    # color = ('b', 'g', 'r')
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([part], [i], None, [256], [0, 256])
    #     plt.plot(histr, color=col)
    # plt.xlim([0, 256])
    # plt.show()
    #
    # img2_color_equl = part.copy()
    # #需要按通道分别均衡化
    # img2_color_equl[:,:,0] = cv2.equalizeHist(part[:,:,0])
    # img2_color_equl[:,:,1] = cv2.equalizeHist(part[:,:,1])
    # img2_color_equl[:,:,2] = cv2.equalizeHist(part[:,:,2])
    #
    # '''
    #   颜色直方图均衡化
    # '''
    # equl=cv2.cvtColor(img2_color_equl,cv2.COLOR_BGR2RGB)
    # cv_show('equl',equl)
    # hsv = cv2.cvtColor(equl, cv2.COLOR_BGR2HSV)
    # low_hsv = np.array([0,0,0])
    # high_hsv = np.array([180,255,30])
    # mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    # cv2.imshow("test",mask)
    # cv2.waitKey(0)
    #
    # ret=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,sqlKernel,iterations=2)
    # cv_show('ret',ret)
    #
    # retCnts=cv2.findContours(ret.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    # Rcnts = sorted(retCnts,key=cv2.contourArea, reverse=True)[0]
    # retCnts = sorted(retCnts, key=lambda a: cv2.boundingRect(a)[3], reverse=True)
    retCnts = sorted(retCnts, key=cv2.contourArea, reverse=True)
    h = retCnts[0]
    i = 0
    (x, y, w, h) = cv2.boundingRect(h)
    x = x + left3
    y = y + top2
    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 255), 2)
    # cv_show('image',img)
    cut = x - 5
    part_in=image_copy[top2:y+100,cut:left2+15]
    # cv_show('part_in',part_in)

    X = x-5
    Y = y-5
    conCnts = getpart_info(part_in)
    countY = []
    # countW = []
    for c in conCnts:

    #      i += 1
          (x, y, w, h) = cv2. boundingRect(c)
          x = x + cut
          y = y + top2
          if abs(x-cut) < 30 and w > 20:
              countY.append(y)
              cv2.rectangle(img_copy, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
              cv_show('image', img_copy)

    print('countY为',countY)

    countY = selectionSort(countY)
    print('排序后的countY为', countY)

    temp = 1
    for i in range(1,len(countY)):
         temp += 1
    print(temp)
    if temp == 4 or temp == 5:
        box_get_front_correction(result_img, X , Y ,8 * gH, long, left2,countY[0],countY[1]
                                  ,countY[2])

    else:
        box_get_front_correction(result_img, X , Y ,8 * gH, long, left2,countY[0],countY[1]
                                 ,countY[1]+60)

    cv_show('image_copy',result_img)

    #      if (w + 10 > h) :
    #         cv2.rectangle(img_copy, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 255), 2)
    #         cv_show('image',img_copy)
        # if w + 10 > h and i == 3 and i == 7:
        #     cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 255), 2)
        #     cv2.imshow('image', img)
        #     cv2.waitKey(0)

        # if w + 15 > h:
        #     #initialize the list of group digits
        #     cv2.rectangle(img,(x-5,y-5),(x+w+5,y+h+5),(0,255,255),2)
        #     cv2.imshow('image', img)
        #     cv2.waitKey(0)



    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


