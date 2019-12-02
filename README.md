
# 身份证信息检测

## 运行环境

python：3.7.3 \
opencv-python：4.1.1.26 \
opencv-contrib-python：3.4.2.17 \
Pillow：6.2.1 \
cmake：3.15.3 \
dlib：19.18.0 \
matplotlib：3.1.1 \
numpy：1.17.3 \
scipy：1.3.1 \

## 项目结构

``` python
|--[data]                           静态数据
|--[docs]                           项目文档
|--[src]                            源代码
|  |-- [com]                        通用工具模块
|  |   |--tools.py                  通用工具类
|  |   
|  |-- [config]                     参数模块
|  |   |--config.py                 参数配置文件
|  | 
|  |-- id_detect.py                 程序入口
|  |-- front_correct_skew.py        身份证正面纠偏程序文件
|  |-- back_correct_skew.py         身份证背面纠偏程序文件
|  |-- idcard_front_detection.py    身份证正面识别程序文件
|  |-- idcard_back_detection.py     身份证背面识别程序文件
|  
|--README.md                        介绍文件
```

## 使用方法

1. 修改 src/config/config.py 中的输入、输出文件或路径信息。
2. 进入src目录，修改id_detect.py，如果执行批处理检测，将is_batch设置为1，否则设置为0。
3. 执行id_detect.py
``` python
python id_detect.py
```

## 注意事项
1. data 目录存放的是静态数据，程序运行时会调用，不可更改。
2. config.py 存放的proporation 一般不需要修改。