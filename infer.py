from dataset.utils import prepare_data, prepare_data_without_model
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizerFast
from train import path_dict
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import csv

# settings
device = "cuda:5"
ckpt_dict =    {"2":  "20",
                "3":  "20",
                "4":  "20",
                "5":  "20",
                "6":  "20",
                "7":  "20",
                "8":  "20",
                "9":  "20",
                "10": "20",
                "11": "20"}
name_dict = {
    "0": "AI-Usage",
    "1": "Humanlikeness_Mental",
    "2": "Humanlikeness_Visual",
    "3": "PSI_Object_AI",
    "4": "PSI_Object_Original",
    "5": "PSI_Identification",
    "6": "PSI_Opinion",
    "7": "PSI_Group",
    "8": "PSI_Interest",
    "9": "AI_Principle_Development"
}

columns_dict = {
    "0": 'AI-Usage',
    "1": "Humanlikeness-Mental",
    "2": "Humanlikeness-Visual",
    "3": "PSI Object-AI",
    "4": "PSI object-charactor itself",
    "5": "PSI-Agreement",
    "6": "PSI-Eexpress opinion",
    "7": "PSI-group",
    "8": "PSI-intersted",
    "9": "AI-Merged"
}

batch_size = 100
assign_gate = 0.5
infer_on_unlabeled = True
data_name =  "L11005uncleaned.xlsx" #"L11005uncleaned.xlsx" #"real_L1L2_uncleaned.xlsx" #"L1_2_infer.xlsx" #"L1_training.xlsx" #"infer_data_1005.xlsx"

class Eval_10bit_info:
    def __init__(self, all_label_pres, all_label_reals, all_label_match, name_dict, saving_path="./"):
        self.all_label_pres, self.all_label_reals, self.all_label_match = all_label_pres, all_label_reals, all_label_match
        self.saving_path = saving_path
        self.name_dict = name_dict

    def tensor_pearsonr(self, tensor1, tensor2):
        """计算两个张量之间的皮尔逊相关系数"""
        x = tensor1.squeeze().numpy()
        y = tensor2.squeeze().numpy()
        corr, _ = pearsonr(x, y)
        return corr   

    def cal_relationship(self, caled_list, fig_name):
        # 计算相关性矩阵
        correlation_matrix = []
        for i in range(len(caled_list)):
            row = []
            for j in range(len(caled_list)):
                corr = self.tensor_pearsonr(caled_list[i], caled_list[j])
                row.append(corr)
            correlation_matrix.append(row)

        # 保存相关性矩阵为CSV文件
        csv_path = self.saving_path + 'correlation_matrix_' + fig_name + '.csv'
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 格式化矩阵并写入CSV
            formatted_matrix = [[f"{val:.2f}" for val in row] for row in correlation_matrix]
            writer.writerows(formatted_matrix)


        correlation_matrix = np.array(correlation_matrix)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
        ax.set_title('Correlation Matrix')
        
        labels = [self.name_dict[str(i)] for i in range(len(caled_list))]
        ax.set_xticks(np.arange(len(caled_list)))
        ax.set_yticks(np.arange(len(caled_list)))
        ax.set_xticklabels(labels, rotation=90)  # x轴标签竖直显示
        ax.set_yticklabels(labels)  # y轴标签水平显示

        for i in range(len(caled_list)):
            for j in range(len(caled_list)):
                text = ax.text(j, i, round(correlation_matrix[i, j], 2),
                               ha="center", va="center", color="w")

        # 指定保存路径
        save_path = self.saving_path + 'correlation_matrix_' + fig_name + '.png'
        fig.tight_layout()
        plt.savefig(save_path)
        plt.show()



    def cal_corlation(self):
        self.cal_relationship(self.all_label_reals, "all_label_reals")
        self.cal_relationship(self.all_label_pres, "all_label_pres")    

    def count_perfect_matches(self):
        stacked_tensor = torch.stack(self.all_label_match)
        column_perfect_matches = torch.all(stacked_tensor == 1, dim=0)
        perfect_matches_count = torch.sum(column_perfect_matches).item()
        perfect_match_rate = perfect_matches_count / stacked_tensor.shape[1] * 100
        return perfect_match_rate


# inference
all_label_pres, all_label_reals, all_label_match = [], [], []
_, _, model = prepare_data(label_columns=[2,12], device=device)    
for label_index in range(2,12):
    # load data
    cur_test_dataset= prepare_data_without_model(label_columns=label_index, device=device, data_name=data_name, infer_on_unlabeled=infer_on_unlabeled)
    test_loader = DataLoader(cur_test_dataset, batch_size=batch_size, sampler=SequentialSampler(cur_test_dataset))
    
    # load model
    savingpath = "./results/"+path_dict[str(label_index)]+"/"
    model.load_state_dict(torch.load(f"{savingpath}model_{ckpt_dict[str(label_index)]}.pt"))
    model.eval()
    all_preds, all_labels = [], []

    # Progress bar for validation loop
    test_loader_tqdm = tqdm(test_loader, desc=f"Validating {path_dict[str(label_index)]}", leave=False)

    with torch.no_grad():
        for batch in test_loader_tqdm:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)


            preds = torch.sigmoid(outputs.logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            test_loader_tqdm.set_postfix({"Val": 1.000})

    # calculate acc of different data
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # accuracy1: all label and mean
    binary_preds = torch.tensor((all_preds > assign_gate).astype(float))
    binary_labels = torch.tensor(all_labels)
    correct = (binary_preds == torch.tensor(all_labels)).float()

    all_label_pres.append(binary_preds)
    all_label_reals.append(torch.tensor(all_labels))
    all_label_match.append(correct)


    print(f"We are currently processing the part {path_dict[str(label_index)]} and the results are show below")
    # Acc 1, over all acc
    accuracy = (correct == 1).sum().item()/len(correct)
    print(f"The num of correct bit is: {(correct == 1).sum().item()} and total amount is: {len(correct)}")
    print(f"Overall Accuracy: {accuracy:.4f}\n")


    # Acc 2, Partial Accuracy
    positive_indices = torch.where(binary_labels == 1)[0]
    negative_indices = torch.where(binary_labels == 0)[0]

    num_positives = positive_indices.size(0)
    num_negatives = negative_indices.size(0)
    print("Current number ratios are: ", num_positives, num_negatives)


    selected_negative_indices = np.random.choice(negative_indices.numpy(), num_positives, replace=False)
    combined_indices = torch.cat([positive_indices, torch.tensor(selected_negative_indices)])
    print("Final num is: ", len(combined_indices))

    selected_binary_preds = binary_preds[combined_indices]
    selected_binary_labels = binary_labels[combined_indices]
    accuracy = (selected_binary_preds == selected_binary_labels).sum().item() / len(combined_indices)
    print(f"Partial Accuracy: {accuracy:.4f}")

# Acc 3 and Corrlation
eval_class = Eval_10bit_info(all_label_pres, all_label_reals, all_label_match, name_dict)
eval_class.cal_corlation()
perfect_match_rate = eval_class.count_perfect_matches()
print(f"the all bit acc is {perfect_match_rate}")


df = pd.read_excel("./dataset/"+data_name)
print(df.columns)
for index in range(len(all_label_pres)):
    # 确保 tensor 的长度与 DataFrame 的行数匹配
    len1=len(all_label_pres[index].numpy())
    len2=len(df[columns_dict[str(index)]])
    if len1 == len2:
        df[columns_dict[str(index)]] = all_label_pres[index].numpy()  # 将 tensor 转换为 numpy 数组后写入 DataFrame
    else:
        print(f"please make sure ({len1}) and ({len2}) match!")

file_path = './new_'+data_name
df.to_excel(file_path, index=False)