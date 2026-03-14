import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import csv
from pathlib import Path

def get_data_new_path(source_datapath, flag="train"):
    """根据source_datapath的文件名字, 获取encoder_path和shuffle_path,保持文件名字一致"""
    # 获取name : 如hit
    file_name = Path(source_datapath).stem

    os.makedirs("./data/encoder", exist_ok=True)
    os.makedirs("./data/shuffle", exist_ok=True)

    if(flag == "train"):
        encoder_path = f"./data/encoder/{file_name}_80pct.pt"
        shuffle_path = f"./data/shuffle/{file_name}_80pct.csv"
    elif(flag == "test"):
        encoder_path = f"./data/encoder/{file_name}_20pct.pt"
        shuffle_path = f"./data/shuffle/{file_name}_20pct.csv"
    elif(flag == "pred"):
        encoder_path = f"./data/encoder/{file_name}_pred.pt"
        shuffle_path = f"./data/shuffle/{file_name}_pred.csv"

    return encoder_path, shuffle_path


class TrainLogger:
    def __init__(self, name, timestamp):
        self.name = name
        self.time = timestamp
        self.losses = []
        self.accuracy = []
        self.auc = []
        self.f1 = []
        self.precision = []
        self.recall = []
        self.best_f1 = 0
        self.best_precision = 0
        self.best_recall = 0
        self.best_accuracy = 0
        self.best_auc = 0
        self.y_true = []
        self.y_score = []

    def log_loss(self, loss):
        self.losses.append(loss)

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.losses) + 1), self.losses, label='Loss', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve During Training')
        plt.legend()
        plt.grid(True)

        plt.ylim(bottom=0)

        save_path = os.path.join("logs", self.time)
        save_path = os.path.join(save_path, self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "loss.png"))

    def log_metrix(self, recall, precision, f1, accuracy, auc):
        self.accuracy.append(accuracy)
        self.auc.append(auc)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_precision = precision
            self.best_recall = recall
            self.best_accuracy = accuracy
            self.best_auc = auc

    def plot_metrics(self):
        plt.figure(figsize=(10, 6))

        plt.plot(range(1, len(self.accuracy) + 1), self.accuracy, label='Accuracy', color='g')
        plt.plot(range(1, len(self.f1) + 1), self.f1, label='F1 Score', color='b')
        plt.plot(range(1, len(self.precision) + 1), self.precision, label='Precision', color='r')
        plt.plot(range(1, len(self.recall) + 1), self.recall, label='Recall', color='orange')
        plt.plot(range(1, len(self.auc) + 1), self.auc, label='AUC', color='purple')

        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Metrics During Training')
        plt.legend()
        plt.grid(True)

        plt.ylim(0, 1)

        save_path = os.path.join("logs", self.time)
        save_path = os.path.join(save_path, self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "metrix.png"))

    def write(self, content):
        save_path = os.path.join("logs", self.time)
        save_path = os.path.join(save_path, self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        log_file_path = os.path.join(save_path, "log.txt")
        with open(log_file_path, "a") as f:
            f.write(content + '\n')
            f.flush()

    def update_true_score(self, y_true, y_score):
        self.y_true = y_true
        self.y_score = y_score

    def update_protein_drug(self, protein_list, drug_list):
        self.protein_list = protein_list
        self.drug_list = drug_list
    
    def plot_auc(self):
        if len(self.y_true) == 0 or len(self.y_score) == 0:
            print("⚠️ [Warning] No data found for AUC plotting. Skipping...")
            return
    
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_score)

        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')

        save_path = os.path.join("logs", self.time)
        save_path = os.path.join(save_path, self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "roc_curve.png"), dpi=300)

    def __del__(self):
        self.plot_loss()
        self.plot_metrics()
        self.plot_auc()
        self.write(f"[BEST] recall = {round(self.best_recall, 4)}, precision = {round(self.best_precision, 4)}, f1 = {round(self.best_f1, 4)}, accuracy ={round(self.best_accuracy, 4)}, auc = {round(self.best_auc, 4)}")
        save_path = os.path.join("logs", self.time)
        save_path = os.path.join(save_path, self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'output.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['protein', 'drug', 'True Label', 'Predicted Probability'])
            # for protein, drug, true_label, predicted_prob in zip(self.protein_list, self.drug_list, self.y_true, self.y_score):
                # writer.writerow([protein, drug, true_label, predicted_prob])
            for true_label, predicted_prob in zip(self.y_true, self.y_score):
                writer.writerow([true_label, predicted_prob])


class PredictLogger:
    def __init__(self, name, timestamp):
        self.name = name
        self.time = timestamp
        self.losses = []
        self.accuracy = []
        self.auc = []
        self.f1 = []
        self.precision = []
        self.recall = []
        self.y_true = []
        self.y_score = []

    def log_metrix(self, recall, precision, f1, accuracy, auc):
        self.accuracy.append(accuracy)
        self.auc.append(auc)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)

    def write(self, content):
        save_path = os.path.join("logs", self.time)
        save_path = os.path.join(save_path, self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        log_file_path = os.path.join(save_path, "log.txt")
        with open(log_file_path, "a") as f:
            f.write(content + '\n')
            f.flush()

    def update_true_score(self, y_true, y_score):
        self.y_true = y_true
        self.y_score = y_score

    def update_protein_drug(self, protein_list, drug_list):
        self.protein_list = protein_list
        self.drug_list = drug_list
    
    def plot_auc(self):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_score)

        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')

        save_path = os.path.join("logs", self.time)
        save_path = os.path.join(save_path, self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "roc_curve.png"), dpi=300)

    def __del__(self):
        self.plot_auc()
        save_path = os.path.join("logs", self.time)
        save_path = os.path.join(save_path, self.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'output.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['protein', 'drug', 'True Label', 'Predicted Probability'])
            for true_label, predicted_prob in zip(self.y_true, self.y_score):
                writer.writerow([true_label, predicted_prob])
            
            # for protein, drug, true_label, predicted_prob in zip(self.protein_list, self.drug_list, self.y_true, self.y_score):
                # writer.writerow([protein, drug, true_label, predicted_prob])

