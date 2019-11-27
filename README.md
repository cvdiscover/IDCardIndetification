
#身份证信息检测

### 1. 项目的基本结构
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

### 2. 程序的使用方式

1. 使用前先更改src - config - config.py 的文件读取输出的绝对路径。
2. 程序的主入口在src - id_detect.py 
3. 当选择批量识别时，只需要输入输出路径的根目录地址即可；当选择单张识别时，需要给定根目录地址和单张图片全名（包括后缀）

### 3. 注意事项
1. data内存放的时静态数据，需要调用，不可更改。
2. config.py 存放的proporation原则上无需更改。