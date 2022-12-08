# 基于神经网络的超声波三维成像（SonicNet）GUI程序
## 配置环境
1. 创建conda虚拟环境`conda create -n SonicNet python=3.10`
2. 激活conda虚拟环境`conda activate SonicNet`
2. 使用`cd`命令进入`SonicNet`目录
3. 运行环境自动配置指令`pip install requirements.txt`
4. 运行程序`python train_gui.py`
## 程序说明
1. 可通过修改`SonicNet/config`目录下的`config.yaml`文件调整超参
2. 程序执行`Render`指令后将在`SonicNet/output`文件夹下生成结果视频文件
3. 由于程序调用的`PyTorch`及`Matplotlib`等模块线程不安全，程序运行期间可能偶然闪退，重新运行即可
