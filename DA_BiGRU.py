import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, \
    classification_report
from torch.autograd import Function
from sklearn.model_selection import train_test_split
# 全局参数
Lam_reversal = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_length=1125
num_classes = 2
batch_size = 128
validation_ratio = 0.1  # 验证集比例
test_ratio = 0.1  # 测试集比例


class GRUClassifier(nn.Module):
    def __init__(self, input_size, seq_length, hidden_size=96, num_layers=3):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size

        # 动态序列编码器
        self.seq_encoder = nn.Sequential(
            nn.Linear(64, input_size * seq_length),
            nn.Mish(),
            nn.Dropout(0.3)
        )

        # 双向GRU核心层
        self.bi_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.25 if num_layers > 1 else 0
        )

        # 层次注意力机制
        self.layer_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_size, 48),
                nn.GELU(),
                nn.Linear(48, 1),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])

        # 时间步注意力
        self.step_attention = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 分类输出层
        self.fc = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, NUM_CLASSES)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 特征序列化
        encoded = self.seq_encoder(x)
        seq = encoded.view(batch_size, self.seq_length, -1)

        # 初始化隐藏状态
        h = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(x.device)

        layer_outputs = []
        for t in range(seq.size(1)):
            # 逐层处理
            out, h = self.bi_gru(seq[:, t:t + 1, :], h)

            # 层次注意力加权
            weighted_out = 0
            for layer in range(self.num_layers):
                layer_h = h[2 * layer:2 * (layer + 1)].transpose(0, 1).contiguous()
                layer_h = layer_h.view(batch_size, -1)
                attn = self.layer_attention[layer](layer_h)
                weighted_out += attn * out

            layer_outputs.append(weighted_out)

        # 时间步注意力聚合
        all_outputs = torch.cat(layer_outputs, dim=1)
        step_weights = self.step_attention(all_outputs)
        context = torch.sum(all_outputs * step_weights, dim=1)

        return self.fc(context)


# 全局配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_LENGTH = 1125
NUM_CLASSES = 2
BATCH_SIZE = 128
LAM_REVERSAL = 0.5  # 领域适应损失权重


# 自定义梯度反转层
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_grl=1.0):
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_grl)


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


# 领域自适应模型
class DomainAdaptationModel(nn.Module):
    def __init__(self, gru_hidden_size=128, num_gru_layers=2, bidirectional=False):
        super().__init__()
        # 保持原有特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            nn.MaxPool1d(2),
            ResidualBlock(64, 64),
            nn.AdaptiveAvgPool1d(1)  # 输出: (batch, 64, 1)
        )

        # GRU分类器 (关键修改点)
        self.gru_classifier = nn.GRU(
            input_size=4,  # 调整后的特征维度
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        # 动态计算分类头维度
        gru_output_dim = gru_hidden_size * (2 if bidirectional else 1)
        self.class_head = nn.Sequential(
            nn.Linear(gru_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )

        # 保持域判别器结构
        self.domain_classifier = AdversarialNetwork(input_dim=64)

    def forward(self, x):
        # 特征提取 [batch, 64, 1]
        features = self.feature_extractor(x)

        # 重塑特征为GRU输入格式
        batch_size = features.size(0)
        seq_features = features.view(batch_size, 16, 4)  # 保持总维度64=16×4

        # GRU处理 (与LSTM的主要区别)
        gru_out, hn = self.gru_classifier(seq_features)  # 无细胞状态
        last_time_step = gru_out[:, -1, :]  # 取最后时间步

        # 分类预测
        class_out = self.class_head(last_time_step)

        # 域预测保持原特征
        domain_features = features.view(batch_size, -1)
        domain_out = self.domain_classifier(domain_features)

        return class_out, domain_out

    def get_config(self):
        """获取当前GRU配置"""
        return {
            'gru_hidden': self.gru_classifier.hidden_size,
            'gru_layers': self.gru_classifier.num_layers,
            'bidirectional': self.gru_classifier.bidirectional
        }

class AdversarialNetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()
        self.grl = GradientReversal(lambda_grl=1.0)  # 初始化梯度反转层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.grl(x)  # 应用梯度反转
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 数据加载与预处理
def load_datasets():
    try:
        # 加载源域数据
        src_fhr = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/centre/fhrbs.csv', delimiter=',')
        src_uc = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/centre/uc.csv', delimiter=',')
        src_fm = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/centre/fm.csv', delimiter=',')
        src_labels = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/centre/label.csv', delimiter=',').astype(int)

        # 加载目标域数据
        tgt_fhr = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/mobile/fhrbs.csv', delimiter=',')
        tgt_uc = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/mobile/uc.csv', delimiter=',')
        tgt_fm = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/mobile/fm.csv', delimiter=',')
        tgt_labels = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/mobile/label.csv', delimiter=',').astype(int)

        # 数据标准化
        def normalize(x):
            return (x - x.min(axis=1, keepdims=True)) / (
                        x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 1e-8)

        # 构建3通道输入
        X_source = np.stack([normalize(src_fhr), normalize(src_uc), normalize(src_fm)], axis=1)
        X_target = np.stack([normalize(tgt_fhr), normalize(tgt_uc), normalize(tgt_fm)], axis=1)

        # 划分目标域数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_target, tgt_labels,
            test_size=0.2,
            stratify=tgt_labels,
            random_state=42
        )

        # 修正维度处理
        source_dataset = TensorDataset(
            torch.FloatTensor(X_source),  # 直接使用[N, 3, 1125]格式
            torch.LongTensor(src_labels)
        )

        target_train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )

        target_test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )

        return source_dataset, target_train_dataset, target_test_dataset

    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return None, None, None


# 训练流程
def train_model(model, source_loader, target_loader, test_loader):
    model.to(device)

    # 优化器配置
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    # 损失函数
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    early_stop_counter = 0

    for epoch in range(100):
        model.train()
        total_loss = 0.0

        # 混合数据迭代
        for (src_data, src_labels), (tgt_data, _) in zip(source_loader, target_loader):
            # 数据准备
            src_data = src_data.to(device)
            src_labels = src_labels.to(device)
            tgt_data = tgt_data.to(device)

            # 合并批次
            combined_data = torch.cat([src_data, tgt_data], dim=0)
            batch_size = src_data.size(0)

            # 前向传播
            class_out, domain_out = model(combined_data)

            # 分类损失（仅源域）
            cls_loss = class_criterion(class_out[:batch_size], src_labels)

            # 领域分类损失
            domain_labels = torch.cat([
                torch.zeros(batch_size, dtype=torch.long),  # 源域
                torch.ones(tgt_data.size(0), dtype=torch.long)  # 目标域
            ]).to(device)
            dom_loss = domain_criterion(domain_out, domain_labels)

            # 总损失
            loss = cls_loss + LAM_REVERSAL * dom_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # 验证评估
        val_auc = evaluate(model, test_loader)
        scheduler.step(val_auc)

        # 早停机制
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 7:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f} | Val AUC: {val_auc:.4f}")

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model


# 评估函数
def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            outputs, _ = model(data)
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    metrics = {
        'AUC': roc_auc_score(y_true, y_probs),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred)
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'Sensitivity': tp / (tp + fn),
        'Specificity': tn / (tn + fp)
    })

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics['AUC']


# 主程序
if __name__ == "__main__":
    # 数据加载
    source_data, target_train_data, target_test_data = load_datasets()
    if None in (source_data, target_train_data, target_test_data):
        exit("Data loading failed")

    # 创建数据加载器
    source_loader = DataLoader(source_data, batch_size=BATCH_SIZE, shuffle=True)
    target_loader = DataLoader(target_train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(target_test_data, batch_size=BATCH_SIZE)

    # 初始化模型
    model = DomainAdaptationModel()

    # 训练模型
    trained_model = train_model(model, source_loader, target_loader, test_loader)

    # 最终评估
    print("\nFinal Test Performance:")
    final_metrics = evaluate(trained_model, test_loader)

    # 保存完整模型
    torch.save({
        'model_state': trained_model.state_dict(),
        'input_spec': (3, INPUT_LENGTH),
        'class_map': {0: 'Normal', 1: 'Abnormal'}
    }, 'domain_adaptation_model.pth')
