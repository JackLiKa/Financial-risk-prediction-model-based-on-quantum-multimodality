import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pennylane as qml
import matplotlib.pyplot as plt
import glob
import warnings
import traceback
from sklearn.metrics import confusion_matrix
import random

# 忽略警告
warnings.filterwarnings("ignore")

# 设置图形后端避免中文显示问题
plt.switch_backend('Agg')  # 使用非交互式后端

# 设置默认数据类型为float32
torch.set_default_dtype(torch.float32)


# ==================== 高风险数据集创建 ====================
def create_high_risk_dataset(num_samples=50, output_dir="corpus_Corpus/high_risk"):
    """
    创建模拟高风险数据集
    :param num_samples: 样本数量
    :param output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 高风险文本示例
    high_risk_texts = [
        "该股票存在重大财务风险，建议卖出",
        "公司负债率高达90%，违约风险极高",
        "市场波动性剧增，投资风险显著上升",
        "行业监管政策突变，业务连续性风险",
        "现金流持续为负，破产风险增加",
        "信用评级下调，融资成本上升",
        "主要客户流失，收入稳定性风险",
        "技术迭代加速，产品过时风险高",
        "法律诉讼缠身，声誉风险突出",
        "宏观经济下行，系统性风险加剧"
    ]

    # 高风险图像特征 - 创建红色预警图表
    def create_high_risk_image():
        img = Image.new('RGB', (100, 100), color=(255, 200, 200))  # 浅红色背景
        draw = ImageDraw.Draw(img)

        # 绘制下降箭头
        draw.line([(50, 20), (50, 80)], fill='red', width=3)
        draw.line([(35, 65), (50, 80)], fill='red', width=3)
        draw.line([(65, 65), (50, 80)], fill='red', width=3)

        # 添加感叹号
        draw.ellipse([(45, 25), (55, 45)], fill='red')
        draw.rectangle([(48, 50), (52, 70)], fill='red')

        # 添加风险文字
        draw.text((30, 5), "风险!", fill='darkred')

        return img

    # 创建高风险数据
    data = []
    for i in range(num_samples):
        # 随机选择高风险文本
        text = random.choice(high_risk_texts) + f" 情景#{i + 1}"

        # 创建高风险图像
        img = create_high_risk_image()
        img_path = os.path.join(output_dir, f"high_risk_{i}.png")
        img.save(img_path)

        data.append({
            "问题": text,
            "参考答案": "高风险"
        })

    # 保存到Excel
    df = pd.DataFrame(data)
    df.to_excel(os.path.join(output_dir, "高风险数据集.xlsx"), index=False)
    print(f"已创建 {num_samples} 个高风险样本到 {output_dir}")


# ==================== 数据预处理 ====================
class FinancialDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.text_features = []
        self.image_features = []
        self.labels = []

        try:
            # 确保高风险数据集存在
            high_risk_dir = os.path.join(root_dir, "high_risk")
            if not os.path.exists(high_risk_dir):
                print("创建高风险数据集...")
                create_high_risk_dataset(output_dir=high_risk_dir)

            # 加载所有Excel文件
            ipo_data = pd.read_excel(os.path.join(root_dir, "金融专业认知能力-IPO数据.xlsx"))
            kg_data = pd.read_excel(os.path.join(root_dir, "金融专业认知能力-知识图谱数据.xlsx"))
            risk_data = pd.read_excel(os.path.join(root_dir, "金融风险控制能力.xlsx"))
            high_risk_data = pd.read_excel(os.path.join(root_dir, "high_risk/高风险数据集.xlsx"))

            # 合并文本数据
            text_data = pd.concat([
                ipo_data[['问题1', '问题1答案']].rename(columns={'问题1': 'text', '问题1答案': 'label'}),
                kg_data[['问题2', '问题2答案']].rename(columns={'问题2': 'text', '问题2答案': 'label'}),
                risk_data[['问题', '参考答案']].rename(columns={'问题': 'text', '参考答案': 'label'}),
                high_risk_data[['问题', '参考答案']].rename(columns={'问题': 'text', '参考答案': 'label'})
            ])

            # 打印原始答案分布
            print("\n原始答案分布:")
            print(text_data['label'].value_counts())

            # 改进风险分类逻辑
            def classify_risk(label):
                label_str = str(label).lower()
                if '风险' in label_str or '高' in label_str or '危险' in label_str or '卖出' in label_str:
                    return '高风险'
                elif '低' in label_str or '安全' in label_str or '买入' in label_str:
                    return '低风险'
                else:
                    return '未知'

            risk_labels = text_data['label'].apply(classify_risk)

            # 检查未知标签
            if '未知' in risk_labels.values:
                unknown_samples = text_data[risk_labels == '未知']
                print(f"警告: 发现 {len(unknown_samples)} 个未知风险标签的样本:")
                print(unknown_samples['label'].unique())

                # 手动处理或设为低风险
                risk_labels = risk_labels.replace('未知', '低风险')

            print("\n转换后的风险标签分布:")
            print(risk_labels.value_counts())

            # 文本向量化 - 保持16维
            self.vectorizer = TfidfVectorizer(max_features=16)
            text_features = self.vectorizer.fit_transform(text_data['text']).toarray().astype(np.float32)

            # 标签编码
            self.le = LabelEncoder()
            labels = self.le.fit_transform(risk_labels)

            # 加载图像路径
            ipo_images = glob.glob(os.path.join(root_dir, "IPO图表/*.png"))
            kg_images = glob.glob(os.path.join(root_dir, "知识图谱/*.png"))
            high_risk_images = glob.glob(os.path.join(root_dir, "high_risk/*.png"))
            all_image_paths = ipo_images + kg_images + high_risk_images

            # 确保数据对齐
            min_length = min(len(text_features), len(all_image_paths), len(labels))

            # 处理图像并收集有效样本
            valid_indices = []
            for idx in range(min_length):
                try:
                    img = Image.open(all_image_paths[idx]).resize((4, 4))
                    img_array = np.array(img).astype(np.float32)

                    # 处理不同通道数的图像
                    if len(img_array.shape) == 2:  # 灰度图
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.shape[2] == 4:  # RGBA图
                        img_array = img_array[..., :3]

                    # 添加到数据集
                    self.text_features.append(text_features[idx])
                    self.image_features.append(img_array[..., :3].flatten() / 255.0)
                    self.labels.append(labels[idx])
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"跳过图像 {all_image_paths[idx]} - 错误: {e}")

            print(f"成功加载 {len(valid_indices)} 个有效样本")
        except Exception as e:
            print(f"数据加载错误: {e}")
            traceback.print_exc()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.text_features[idx], dtype=torch.float32),
            torch.tensor(self.image_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# ==================== 量子神经网络 ====================
class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # 文本特征处理器
        self.text_processor = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, n_qubits)
        )

        # 图像特征处理器
        self.image_processor = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits)
        )

        # 量子电路参数 - 使用float32
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3, dtype=torch.float32))

        # 经典分类器
        self.classifier = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        # 量子设备
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # 创建量子节点 - 固定接口
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")

    def quantum_circuit(self, inputs, weights):
        # 角度嵌入
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        # 变分层
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.Rot(
                    weights[layer, i, 0],
                    weights[layer, i, 1],
                    weights[layer, i, 2],
                    wires=i
                )

            # 纠缠层
            for i in range(self.n_qubits - 1):
                qml.CZ(wires=[i, i + 1])

        # 测量
        return qml.expval(qml.PauliZ(0))

    def forward(self, text, image):
        # 处理文本特征
        text_features = self.text_processor(text)

        # 处理图像特征
        image_features = self.image_processor(image)

        # 合并特征
        combined_features = (text_features + image_features) / 2.0

        # 量子计算
        quantum_outputs = []
        for features in combined_features:
            # 确保使用float32
            output = self.qnode(features.float(), self.weights.float())
            quantum_outputs.append(output)

        quantum_outputs = torch.stack(quantum_outputs).float()

        # 分类
        return self.classifier(quantum_outputs.unsqueeze(1))


# ==================== 训练与评估 ====================
def train_model(dataset, num_epochs=10):
    try:
        # 使用分层抽样划分数据集 (60%训练, 20%验证, 20%测试)
        labels = np.array([label.item() for _, _, label in dataset])
        train_idx, temp_idx, _, _ = train_test_split(
            range(len(dataset)), labels, test_size=0.4, stratify=labels, random_state=42
        )
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, labels[temp_idx], test_size=0.5, stratify=labels[temp_idx], random_state=42
        )

        # 创建子集
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        print(f"\n数据集划分: 训练集 {len(train_dataset)} | 验证集 {len(val_dataset)} | 测试集 {len(test_dataset)}")
        print(f"类别分布 - 训练集: 低风险 {np.sum(labels[train_idx] == 0)} / 高风险 {np.sum(labels[train_idx] == 1)}")
        print(f"类别分布 - 验证集: 低风险 {np.sum(labels[val_idx] == 0)} / 高风险 {np.sum(labels[val_idx] == 1)}")
        print(f"类别分布 - 测试集: 低风险 {np.sum(labels[test_idx] == 0)} / 高风险 {np.sum(labels[test_idx] == 1)}")

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # 初始化模型 - 确保使用float32
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=2).float()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # 添加学习率调度器 (移除了 verbose 参数)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # 记录当前学习率
        prev_lr = optimizer.param_groups[0]['lr']

        # 训练历史记录
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': [], 'test_acc': []
        }

        best_val_acc = 0.0
        best_model_state = None

        print("\n开始训练量子模型...")
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for text, image, labels in train_loader:
                optimizer.zero_grad()

                outputs = model(text, image)
                loss = criterion(outputs, labels)

                # 添加L2正则化
                l2_lambda = 0.001
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for text, image, labels in val_loader:
                    outputs = model(text, image)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # 更新学习率
            scheduler.step(val_loss)

            # 检查学习率是否变化
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != prev_lr:
                print(f"学习率更新: {prev_lr:.6f} -> {new_lr:.6f}")
                prev_lr = new_lr

            # 测试阶段 (仅用于最终报告)
            test_loss = 0.0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for text, image, labels in test_loader:
                    outputs = model(text, image)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_loss /= len(test_loader)
            test_acc = 100 * test_correct / test_total
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                print(f"发现新的最佳模型! 验证准确率: {val_acc:.2f}%")

            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}%")

        # 保存最佳模型
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), 'quantum_model.pth')
        print(f"\n模型已保存为 quantum_model.pth (最佳验证准确率: {best_val_acc:.2f}%)")

        # 返回测试集用于最终评估
        return model, history, test_dataset

    except Exception as e:
        print(f"训练过程中出错: {e}")
        traceback.print_exc()
        return None, None, None


# ==================== 量子态可视化 ====================
def visualize_quantum_state(model, sample_text, sample_image):
    try:
        # 处理样本
        with torch.no_grad():
            text_features = model.text_processor(sample_text.unsqueeze(0))
            image_features = model.image_processor(sample_image.unsqueeze(0))
            combined_features = (text_features + image_features) / 2.0

            # 创建量子设备
            dev = qml.device("default.qubit", wires=model.n_qubits)

            # 定义可视化电路
            @qml.qnode(dev)
            def circuit(inputs):
                # 角度嵌入
                for i in range(model.n_qubits):
                    qml.RY(inputs[i], wires=i)

                # 变分层
                for layer in range(model.n_layers):
                    for i in range(model.n_qubits):
                        qml.Rot(
                            model.weights[layer, i, 0].item(),
                            model.weights[layer, i, 1].item(),
                            model.weights[layer, i, 2].item(),
                            wires=i
                        )

                    # 纠缠层
                    for i in range(model.n_qubits - 1):
                        qml.CZ(wires=[i, i + 1])

                return qml.state()

            # 获取量子态
            state = circuit(combined_features.squeeze(0).numpy().astype(np.float32))

            # 绘制电路图
            fig, ax = qml.draw_mpl(circuit)(combined_features.squeeze(0).numpy().astype(np.float32))
            ax.set_title("Quantum Circuit")
            plt.savefig("quantum_circuit.png", bbox_inches='tight')
            plt.close()
            print("量子电路图已保存为 quantum_circuit.png")

            # 绘制量子态
            plt.figure(figsize=(12, 6))
            num_states = 2 ** model.n_qubits
            state_labels = [format(i, f'0{model.n_qubits}b') for i in range(num_states)]
            plt.bar(range(num_states), np.abs(state) ** 2)
            plt.xlabel("Quantum State Basis")
            plt.ylabel("Probability")
            plt.title("Quantum State Distribution")
            plt.xticks(range(num_states), state_labels, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("quantum_state.png")
            plt.close()
            print("量子态分布图已保存为 quantum_state.png")

    except Exception as e:
        print(f"量子可视化出错: {e}")
        traceback.print_exc()


# ==================== 高风险样本预测 ====================
def predict_high_risk_samples(model, dataset, num_samples=5):
    """
    专门预测高风险样本
    :param model: 训练好的模型
    :param dataset: 完整数据集
    :param num_samples: 预测样本数量
    """
    print("\n[高风险样本预测测试]")

    # 查找高风险样本
    high_risk_indices = [i for i, (_, _, label) in enumerate(dataset) if label == 1]

    if not high_risk_indices:
        print("警告: 未找到高风险样本")
        return

    # 随机选择高风险样本
    selected_indices = np.random.choice(high_risk_indices, size=min(num_samples, len(high_risk_indices)), replace=False)

    print(f"随机选择 {len(selected_indices)} 个高风险样本进行预测:")

    for i, idx in enumerate(selected_indices):
        text, image, label = dataset[idx]

        with torch.no_grad():
            model.eval()
            prediction = model(text.unsqueeze(0), image.unsqueeze(0))
            predicted_class = torch.argmax(prediction).item()
            probabilities = torch.softmax(prediction, dim=1)[0].tolist()

            print(f"\n样本 {i + 1}/{len(selected_indices)} (ID: {idx}):")
            # 获取原始文本（部分显示）
            try:
                original_text = dataset.text_features[idx]
                text_str = " ".join([f"{v:.2f}" for v in original_text[:5]]) + " ..."
                print(f"文本特征: {text_str}")
            except:
                print("无法获取原始文本")

            print(f"真实标签: {'高风险' if label == 1 else '低风险'}")
            print(f"预测结果: {'高风险' if predicted_class == 1 else '低风险'}")
            print(f"低风险概率: {probabilities[0]:.4f}, 高风险概率: {probabilities[1]:.4f}")

            # 风险分析
            if predicted_class == 1:
                if probabilities[1] > 0.8:
                    print("风险评估: ★★★★★ 极高风险 (置信度 > 80%)")
                elif probabilities[1] > 0.6:
                    print("风险评估: ★★★★ 高风险 (置信度 60-80%)")
                else:
                    print("风险评估: ★★★ 中等风险 (置信度 < 60%)")
            elif probabilities[1] > 0.4:
                print("风险评估: ★★ 潜在风险 (模型检测到风险迹象)")
            else:
                print("风险评估: ★ 低风险")

            # 显示图像特征
            try:
                img_array = image.numpy().reshape(4, 4, 3) * 255
                plt.figure(figsize=(3, 3))
                plt.imshow(img_array.astype(np.uint8))
                plt.title(f"样本 {i + 1} 图像")
                plt.axis('off')
                plt.savefig(f"sample_{i + 1}_image.png", bbox_inches='tight')
                plt.close()
                print(f"图像已保存为 sample_{i + 1}_image.png")
            except:
                print("无法保存图像")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("量子多模态金融风险预测系统")
    print("=" * 50)

    try:
        # 设置全局数据类型为float32
        torch.set_default_dtype(torch.float32)

        # 初始化数据集
        print("\n[步骤1/4] 加载数据集...")
        dataset = FinancialDataset("corpus_Corpus")

        if len(dataset) == 0:
            print("错误: 未加载到数据，请检查路径和文件")
            exit(1)

        # 查看样本
        sample_text, sample_image, sample_label = dataset[0]
        print(f"\n数据集大小: {len(dataset)} 个样本")
        print(f"文本特征维度: {sample_text.shape}")
        print(f"图像特征维度: {sample_image.shape}")
        print(f"标签: {sample_label.item()} (0=低风险, 1=高风险)")

        # 检查类别平衡
        labels = np.array([label.item() for _, _, label in dataset])
        unique, counts = np.unique(labels, return_counts=True)
        print("\n类别分布统计:")
        for cls, count in zip(unique, counts):
            print(f"{'低风险' if cls == 0 else '高风险'}: {count} 样本 ({count / len(dataset):.2%})")

        # 如果只有一个类别，警告用户
        if len(unique) == 1:
            print("\n警告: 数据集中只有一个风险类别，模型只能学习预测单一类别")

        # 训练模型
        print("\n[步骤2/4] 训练量子模型...")
        trained_model, training_history, test_dataset = train_model(dataset, num_epochs=10)

        if trained_model is None:
            print("训练失败，无法继续")
            exit(1)

        # 可视化量子态
        print("\n[步骤3/4] 可视化量子模型...")
        visualize_quantum_state(trained_model, sample_text, sample_image)

        # 测试预测
        print("\n[步骤4/4] 测试模型预测...")

        # 在整个测试集上评估
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        all_labels = []
        all_preds = []
        all_probs = []

        trained_model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0

            for text, image, labels in test_loader:
                outputs = trained_model(text, image)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())

        test_acc = 100 * test_correct / test_total
        print(f"\n测试集最终性能: 准确率 = {test_acc:.2f}%")

        # 计算混淆矩阵（添加保护性检查）
        cm = confusion_matrix(all_labels, all_preds)

        print("\n混淆矩阵:")
        if cm.shape == (1, 1):
            print(f"所有样本均为 {'低风险' if all_labels[0] == 0 else '高风险'}")
            print(f"正确预测数: {cm[0, 0]}")
        else:
            print(f"真正例(TP): {cm[1, 1]} | 假正例(FP): {cm[0, 1]}")
            print(f"假负例(FN): {cm[1, 0]} | 真负例(TN): {cm[0, 0]}")

            # 只在有高风险样本时计算指标
            if 1 in all_labels:
                precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
                recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                print("\n金融风险预测指标:")
                print(f"高风险召回率(覆盖度): {recall:.4f}")
                print(f"高风险精确率: {precision:.4f}")
                print(f"F1分数: {f1:.4f}")
            else:
                print("\n无高风险样本，无法计算高风险指标")

        # 高风险样本专项预测
        predict_high_risk_samples(trained_model, dataset, num_samples=5)

        # 随机选择一个测试样本展示
        test_idx = np.random.randint(0, len(test_dataset))
        test_text, test_image, test_label = test_dataset[test_idx]

        with torch.no_grad():
            trained_model.eval()
            prediction = trained_model(test_text.unsqueeze(0), test_image.unsqueeze(0))
            predicted_class = torch.argmax(prediction).item()
            probabilities = torch.softmax(prediction, dim=1)[0].tolist()

            print("\n随机测试样本结果:")
            print(f"真实标签: {'高风险' if test_label == 1 else '低风险'}")
            print(f"预测结果: {'高风险' if predicted_class == 1 else '低风险'}")
            print(f"低风险概率: {probabilities[0]:.4f}, 高风险概率: {probabilities[1]:.4f}")

        print("\n量子金融风险预测完成!")

    except Exception as e:
        print(f"\n程序运行时发生严重错误: {e}")
        traceback.print_exc()