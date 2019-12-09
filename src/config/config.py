import os
pwd = os.getcwd()

###################################################################
# proporation 为裁剪的国徽左上角点到四条边的距离与裁剪边长度的比例      #
# 到左边缘：28/129
# 到下边缘：369/129
# 到右边缘：607/129
# 到上边缘：24/129
# 其中向原点偏移为正向（上左）其余为反向（下右）                       #
###################################################################
guohui_direct = "../data/national_emblem.png"
proportion = [32/129, -375/129, -627/129, 30/129]

is_debug = 0
is_show_mark = 0
is_show_id_rect = 0
is_show_id_binary = 0
is_show_lines = 0
is_show_select_lines = 0
is_show_predict_lines = 0
is_show_point_lines = 0


###################################################################
# 输入输出参数
# input_dir：输入地址
# output_dir：输出地址
# path_without_img_name：单张地址根目录
# img_name: 单张图片名称
###################################################################
input_dir = "C:/Users/Administrator/PycharmProjects/cread_ocr/sfz_correct/"
output_dir = "C:/Users/Administrator/PycharmProjects/cread_ocr/last_test_4/"
path_without_img_name = "C:/Users/Administrator/PycharmProjects/cread_ocr/sfz_correct/"
img_name = "ff5ee0f8-97f6-4915-99ce-2eda27d1cfb5.jpeg"

