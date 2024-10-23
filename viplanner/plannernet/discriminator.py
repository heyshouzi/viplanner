import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNet18DiscriminatorShared(nn.Module):
    def __init__(self, depth_channels, semantic_channels, goal_channels):
        super(ResNet18DiscriminatorShared, self).__init__()
        
        # 加载预训练的ResNet-18模型
        self.resnet18 = resnet18(pretrained=True)
        # 修改ResNet的输入通道数为3，处理路径点 (n, 3)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改最后一层为输出特征的大小，去掉最后的分类层
        self.resnet18.fc = nn.Identity()  # 我们不需要分类层
        
        # 计算embedding的维度
        embedding_dim = depth_channels + semantic_channels + goal_channels
        # 全连接层结合ResNet的输出与embedding
        self.fc1 = nn.Linear(embedding_dim + 512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # 使用sigmoid函数输出0到1之间的值
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedding, keypoint_path, gt_path):
        """
        embedding: (batch_size, depth_channels + semantic_channels + goal_channels, H', W')
        keypoint_path: (batch_size, n, 3) -> 生成的路径 (key-point path)
        gt_path: (batch_size, n, 3) -> Ground truth路径
        """
        batch_size = keypoint_path.shape[0]
        
        # ResNet输入是图片格式的 (batch_size, 3, H, W)，我们需要调整路径点的格式
        keypoint_path = keypoint_path.unsqueeze(1)  # 添加一个虚拟的维度作为通道
        gt_path = gt_path.unsqueeze(1)  # 同样调整GT的格式
        
        # 使用同一个ResNet-18提取路径特征
        keypoint_features = self.resnet18(keypoint_path)  # (batch_size, 512)
        gt_features = self.resnet18(gt_path)  # (batch_size, 512)
        
        # 将生成路径与真实路径特征进行concat
        path_features = torch.cat([keypoint_features, gt_features], dim=1)  # (batch_size, 1024)

        # 在这里我们需要处理embedding的形状，进行展平操作
        embedding = embedding.view(batch_size, -1)  # (batch_size, depth_channels + semantic_channels + goal_channels * H' * W')

        # 将路径特征与embedding进行结合
        combined_features = torch.cat([embedding, path_features], dim=1)  # (batch_size, embedding_dim + 1024)

        # 通过全连接层处理融合特征
        x = F.relu(self.fc1(combined_features))  # (batch_size, 256)
        x = F.relu(self.fc2(x))  # (batch_size, 128)
        x = self.fc3(x)  # (batch_size, 1)
        
        # 最终输出判断真假
        output = self.sigmoid(x)  # (batch_size, 1)
        
        return output

