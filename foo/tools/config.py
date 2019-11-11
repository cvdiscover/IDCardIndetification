import os
pwd = os.getcwd()

# proporation 为裁剪的国徽左上角点到四条边的距离与裁剪边长度的比例
# 到左边缘：28/129
# 到下边缘：369/129
# 到右边缘：607/129
# 到上边缘：24/129
# 其中向原点偏移为正向（上左）其余为反向（下右）
guohui_direct = "../tools/guohui_direct.png"  # C:/Users/Alexi/Desktop/IDCard_Identification/
proportion = [32/129, -375/129, -627/129, 30/129]
is_debug = 1
is_show_mark = 0
is_show_id_rect = 0
is_show_id_binary = 0
is_show_lines = 0
is_show_select_lines = 0
is_show_predict_lines = 0
is_show_point_lines = 0
