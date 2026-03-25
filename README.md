# 基于YOLO的电动车驾驶员佩戴安全帽检测识别系统
![readme.p                                                                                            ng](readme.png)

## 项目介绍
参考文档          
[【YOLO 系列】基于YOLO V8的电动车驾驶员安全帽佩戴检测识别系统【python源码+Pyqt5界面+数据集+训练代码】](https://mp.weixin.qq.com/s/n7qPWy-OkCwQipZSiWzahg)

## 项目目录
```
| --
    --img                           # 测试图片和视频
    --output                        # 推理结果保存位置
    --runs                          # 模型训练相关文件
    --static                        # 网页相关文件
        --css                       # 布局设置
        --icon                      # 图标
        --js                        # script.js
    --templates                     # 网页界面文件
    --VOCData                       # 数据处理文件夹
        -- mydata.yaml              # 数据配置文件
        -- splitDataset.py          # 划分数据集
        -- ViewCategory.py          # 查看类别
        -- xml2txt.py               # 标注转换
    --weights                       # 预训练权重
    --detect.py                     # 主界面代码
    --GUI.py                        # 界面代码
    --README.md                     # 文档
    --requirements.txt              # 依赖库（不含torch）
    --train.py                      # 训练代码
    --ui.py                         # 界面代码
    --val.py                        # 验证代码
    --web.py                        # 网页代码
```


## 可视化界面
```
python GUI.py
```

## 网页端
```
python web.py
```

1. 将解压的数据集放在 VOCData 文件夹里面； 
2. 运行splitDataset.py，用于划分数据集；
3. 运行xml2txt.py，用于得到训练标注文件；
4. 运行ViewCategory.py，用于查看一共有那些类别；

### 训练
```
python train.py
```

## 验证
```
python val.py
```

## 推理
```
python detect.py
```
