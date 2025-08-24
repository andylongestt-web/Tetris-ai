HELP!
my ai works like S***, any idea on how to improve my reward function?

```bash
# 创建 conda 环境，将其命名为 TetrisAI，Python 版本 3.8.16
conda create -n TetrisAI python=3.8.16
conda activate TetrisAI
```

Windows:

```bash
# 使用 GPU 训练需要手动安装完整版 PyTorch
conda install pytorch=2.0.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 运行程序脚本测试 PyTorch 是否能成功调用 GPU
python .\utils\check_gpu_status.py

# 降级安装外部代码库
pip install setuptools==65.5.0 pip==21
默认路径 /data/coding
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

