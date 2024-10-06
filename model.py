import torch
import os
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_scheduler,
)
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm

class BCEFocalLoss(torch.nn.Module):
    """
    在Focal Loss中，gamma 和 alpha 是两个重要的超参数，它们分别用来调整损失函数的行为，以更好地应对不同场景下的问题。下面是这两个参数的具体作用：

    Gamma (γ) 参数
    作用：gamma 控制容易分类的样本对总损失的影响程度。Focal Loss的核心思想是减少已经很好地分类样本的损失贡献，从而使模型更加专注于那些难以分类的样本。
    效果：
    当 gamma = 0 时，Focal Loss退化为标准的二分类交叉熵损失。
    当 gamma > 0 时，随着 gamma 值的增加，容易分类的样本（即预测概率很高或很低的样本）的损失会被进一步降低，而难分类的样本（即预测概率接近0.5的样本）的损失会被放大。
    典型的 gamma 值通常在 [0, 5] 范围内选择，常见的默认值是 2。
    Alpha (α) 参数
    作用：alpha 是一个平衡正负样本之间损失的权重因子。在二分类问题中，它用来解决类别不平衡的问题。
    效果：
    当 alpha = 0.5 时，正负样本的损失权重相等，相当于没有进行任何调整。
    当 alpha < 0.5 时，负样本的损失权重更高；当 alpha > 0.5 时，正样本的损失权重更高。
    典型的 alpha 值通常在 [0, 1] 范围内选择，常见的默认值是 0.25 或 0.75，具体取决于数据集的正负样本比例。
    """
    def __init__(self, gamma=1, alpha=0.75, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.epsilon = 1e-7  # 防止log(0)

    def forward(self, input, target):
        pt = torch.sigmoid(input) # 使用sigmoid获取概率
        target = target.float()  # 确保target是浮点型
        # 在原始交叉熵基础上增加动态权重因子
        loss = - self.alpha * (1 - pt + self.epsilon) ** self.gamma * target * torch.log(pt + self.epsilon) \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt + self.epsilon)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def train(train_dataset, test_dataset, model, gamma=1, alpha=0.75, batch_size=64, num_epochs=20, lr=1e-4, eval_rounds=5, assign_gate=0.3, savingpath="./results/AI_Usage/", device='cuda:5'):
    batch_size = batch_size
    num_epochs = num_epochs
    if not os.path.exists(savingpath):
        os.makedirs(savingpath)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = BCEFocalLoss(gamma=gamma, alpha=alpha)   #torch.nn.BCEWithLogitsLoss()
    scheduler = get_scheduler(
        "cosine_with_restarts",
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=len(train_loader) * 20
    )

    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    print(f"we start to train model on device {device}")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        # Progress bar for training loop
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training", leave=False)

        for batch in train_loader_tqdm:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)

            loss = criterion(outputs.logits, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            train_loader_tqdm.set_postfix({"Train Loss": loss.item()})

        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
        with open(f"{savingpath}+training_log.txt", "a") as f:
            f.write(f"Epoch {epoch}: Train Loss: {train_loss:.4f}\n")



        if epoch % eval_rounds == 0:
            # Eval Process
            model.eval()
            total_val_loss = 0
            all_preds, all_labels = [], []

            # Progress bar for validation loop
            test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch}/{num_epochs} - Validating", leave=False)

            with torch.no_grad():
                for batch in test_loader_tqdm:
                    inputs = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(inputs, attention_mask=attention_mask)
                    # outputs = torch.sigmoid(outputs.logits)
                    loss = criterion(outputs.logits, labels.float())
                    total_val_loss += loss.item()

                    preds = torch.sigmoid(outputs.logits).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

                    test_loader_tqdm.set_postfix({"Val Loss": loss.item()})

            val_loss = total_val_loss / len(test_loader)


            # calculate acc of different data
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            # accuracy1: all label and mean
            binary_preds = torch.tensor((all_preds > assign_gate).astype(float))
            binary_labels = torch.tensor(all_labels)
            correct = (binary_preds == binary_labels).float()
            accuracy = (correct == 1).sum().item()/len(correct)
            print(f"The num of correct bit is: {(correct == 1).sum().item()} and total amount is: {len(correct)}")
            print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}\n")

            
            # Acc 2, Partial Accuracy
            positive_indices = torch.where(binary_labels == 1)[0]
            negative_indices = torch.where(binary_labels == 0)[0]

            num_positives = positive_indices.size(0)
            num_negatives = negative_indices.size(0)

            selected_negative_indices = np.random.choice(negative_indices.numpy(), num_positives, replace=False)
            combined_indices = torch.cat([positive_indices, torch.tensor(selected_negative_indices)])

            selected_binary_preds = binary_preds[combined_indices]
            selected_binary_labels = binary_labels[combined_indices]
            accuracy = (selected_binary_preds == selected_binary_labels).sum().item() / len(combined_indices)
            print(f"Partial Accuracy: {accuracy:.4f}")
            
            # Post Processing
            with open(f"{savingpath}+training_log.txt", "a") as f:
                f.write(
                    f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}\n\n")
            torch.save(model.state_dict(), f"{savingpath}model_{str(epoch)}.pt")


def load_ckpt(model, savingpath, device='cuda:5'):
    # 加载模型权重
    checkpoint_path = savingpath                                            # 替换为你实际保存的模型文件路径
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 设置模型为评估模式
    model.eval()
    return model

# # 推理示例
# with torch.no_grad():  # 关闭梯度计算，提高推理速度并减少内存消耗
#     input_ids = torch.tensor([[1, 2, 3]])  # 示例输入，替换为实际输入
#     attention_mask = torch.tensor([[1, 1, 1]])  # 示例注意力掩码，替换为实际输入
#     input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
#     outputs = model(input_ids, attention_mask=attention_mask)
#     predictions = torch.sigmoid(outputs.logits).cpu().numpy()  # 将输出转换为概率分布
#     print(predictions)