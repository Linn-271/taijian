import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, \
    classification_report
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Function

# 全局参数
Lam_reversal = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义梯度反转层
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)  # Use padding=1
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)  # Use padding=1
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)

# main mordol
class DomainAdaptationModel(nn.Module):
    def __init__(self, input_shape=1125, num_classes=2):
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv1d(3, 32, 3, padding=1)  # Use padding=1
        self.bn1 = nn.BatchNorm1d(32)
        self.res_block1 = ResidualBlock(32, 32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)  # Use padding=1
        self.bn2 = nn.BatchNorm1d(64)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes))

        # Adversarial network
        self.adversarial = AdversarialNetwork(64, 64)

    def forward(self, fhr, uc, fm):
        x = torch.cat([fhr.unsqueeze(1), uc.unsqueeze(1), fm.unsqueeze(1)], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.res_block1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gap(x).squeeze(-1)

        # Classification output
        class_out = self.classifier(x)

        # Domain classification output
        domain_out = self.adversarial(x)

        return class_out, domain_out


# 对抗网络
class AdversarialNetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()
        self.grl = GradientReversal(alpha=Lam_reversal)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.grl(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 自定义损失函数
def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    ce_loss = nn.CrossEntropyLoss(reduction='none')(y_pred, y_true)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


# 训练函数
def train(model, Xs, Xt, ys, yt, max_epochs=50, batch_size=128):
    # 准备数据
    (Xs_fhr, Xs_uc, Xs_fm) = Xs
    (Xt_fhr, Xt_uc, Xt_fm) = Xt

    # 合并源域和目标域数据
    X_adv_fhr = torch.FloatTensor(np.concatenate([Xs_fhr, Xt_fhr]))
    X_adv_uc = torch.FloatTensor(np.concatenate([Xs_uc, Xt_uc]))
    X_adv_fm = torch.FloatTensor(np.concatenate([Xs_fm, Xt_fm]))

    # 创建领域标签
    domain_labels = torch.LongTensor(np.concatenate([np.zeros(len(Xs_fhr)), np.ones(len(Xt_fhr))]))
    pseudo_labels = torch.LongTensor(np.concatenate([ys, np.zeros(len(yt))]))  # 目标域伪标签

    # 创建数据集
    dataset = TensorDataset(X_adv_fhr, X_adv_uc, X_adv_fm, pseudo_labels, domain_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_auc = 0.0
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            fhr, uc, fm, labels, domains = batch
            fhr, uc, fm = fhr.to(device), uc.to(device), fm.to(device)
            labels, domains = labels.to(device), domains.to(device)

            optimizer.zero_grad()

            # 前向传播
            class_out, domain_out = model(fhr, uc, fm)

            # 计算损失
            class_loss = focal_loss(class_out, labels)
            domain_loss = nn.CrossEntropyLoss()(domain_out, domains)
            total_loss = class_loss + 0.5 * domain_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            total_loss += total_loss.item()

        scheduler.step()

        # 评估目标域
        model.eval()
        with torch.no_grad():
            Xt_tensor = [torch.FloatTensor(Xt_fhr).to(device),
                         torch.FloatTensor(Xt_uc).to(device),
                         torch.FloatTensor(Xt_fm).to(device)]
            y_pred, _ = model(*Xt_tensor)
            y_probs = torch.softmax(y_pred, dim=1).cpu().numpy()
            y_pred_labels = np.argmax(y_probs, axis=1)

            # 计算指标
            acc = accuracy_score(yt, y_pred_labels)
            auc = roc_auc_score(yt, y_probs[:, 1])
            f1 = f1_score(yt, y_pred_labels, average='weighted')
            precision = precision_score(yt, y_pred_labels)
            recall = recall_score(yt, y_pred_labels)

            print(f"Epoch {epoch + 1}/{max_epochs}")
            print(f"Loss: {total_loss / len(loader):.4f}")
            print(f"Target Domain - Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
            print(classification_report(yt, y_pred_labels))

            # 保存最佳模型
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "/raid/MBDAI/ctg/qklwwm/best_model.pth")
                print("Saved best model with AUC:", auc)

    return model


# 数据加载函数
def load_data():
    try:

        Xs_fhr = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/centre/fhrbs.csv', delimiter=',')
        Xs_uc = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/centre/uc.csv', delimiter=',')
        Xs_fm = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/centre/fm.csv', delimiter=',')
        ys = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/centre/label.csv', delimiter=',').astype(int)

        Xt_fhr = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/mobile/fhrbs.csv', delimiter=',')
        Xt_uc = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/mobile/uc.csv', delimiter=',')
        Xt_fm = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/mobile/fm.csv', delimiter=',')
        yt = np.loadtxt('/raid/MBDAI/ctg/qklwwm/Dataset/mobile/label.csv', delimiter=',').astype(int)

        return (Xs_fhr, Xs_uc, Xs_fm), (Xt_fhr, Xt_uc, Xt_fm), ys, yt
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None  # 确保返回四个None


if __name__ == "__main__":
    # 加载数据（修改后的正确方式）
    Xs, Xt, ys, yt = load_data()

    # 正确的None检查方式
    if Xs is None or Xt is None or ys is None or yt is None:
        print("Failed to load data")
    else:
        # 初始化模型
        model = DomainAdaptationModel().to(device)

        # 转换并验证数据形状
        print(f"Source FHR shape: {Xs[0].shape}")
        print(f"Source UC shape: {Xs[1].shape}")
        print(f"Source FM shape: {Xs[2].shape}")
        print(f"Target FHR shape: {Xt[0].shape}")

        # 训练模型
        trained_model = train(model, Xs, Xt, ys, yt)

        # 模型保存
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'input_shape': 1125,
            'num_classes': 2
        }, "/raid/MBDAI/ctg/qklwwm/complete_model.pth")
        print("Model saved with metadata")


