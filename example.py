# ① 针对有网络模型，但还没有训练保存 .pth 文件的情况
import netron
import torch.onnx
from torchvision.models import resnet18  # 以 resnet18 为例

myNet = resnet18()  # 创建模型网络
x = torch.randn(1, 3, 224, 224)  # 随机生成一个输入
modelData = "./demo.pth"  # 定义模型数据保存的路径
torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
netron.start(modelData)  # 输出网络结构


# ② 针对已经存在网络模型 .pth 文件的情况
import netron

modelData = "./demo.pth"  # 定义模型数据保存的路径
netron.start(modelData)  # 输出网络结构
