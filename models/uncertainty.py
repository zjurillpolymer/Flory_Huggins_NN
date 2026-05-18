import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.attention_net import ChiPredictor
from features.data_loader import load_data, df_to_loader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



'''mc dropout'''
def mc_dropout(train_loader, num_samples=50):
    model = ChiPredictor().to(device)
    model.load_state_dict(torch.load('/Users/arcadio/flory_huggins/models/best_model.pt'))
    model.train()

    all_predictions=[]

    with torch.no_grad():
        for i in range(num_samples):
            batch_preds=[]
            for inputs,_ in train_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                batch_preds.append(outputs.cpu().numpy())
            all_predictions.append(np.concatenate(batch_preds, axis=0))

    all_predictions = np.array(all_predictions)

    # 计算均值：作为最终的预测结果
    mean_prediction = np.mean(all_predictions, axis=0)

    # 计算方差：作为模型的不确定性 (Epistemic Uncertainty)
    uncertainty = np.var(all_predictions, axis=0)

    return mean_prediction, uncertainty



def ensemble(train_loader,num_models=4,num_epochs=100):

    model = ChiPredictor().to(device)
    # 创建文件夹专门存放这组集成模型
    os.makedirs("ensemble_weights", exist_ok=True)


    '''每次独立训练模型'''
    for i in range(num_models):
        model=ChiPredictor().to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step(epoch_loss / len(train_loader))
            if (epoch + 1) % 10 == 0:
                print(f"Model {i + 1}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

        save_path = f"ensemble_weights/model_{i + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"第 {i + 1} 个模型已保存至 {save_path}")



def ensemble_inference(test_loader,num_models=4):
    models=[]
    print("正在加载集成模型...")
    for i in range(1, num_models + 1):
        model = ChiPredictor().to(device)
        load_path = f"ensemble_weights/model_{i}.pth"
        model.load_state_dict(torch.load(load_path))
        model.eval()  # 集成推理时，每个子模型设为评估模式
        models.append(model)

    all_predictions = []
    print("开始集成推理...")
    with torch.no_grad():
        # B. 遍历每一个模型，让它们分别对整个测试集做一次预测
        for i, model in enumerate(models):
            print(f"模型 {i + 1} 正在预测...")
            batch_preds = []
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                batch_preds.append(outputs.cpu().numpy())

            # 将当前模型对全量测试集的预测拼接起来
            all_predictions.append(np.concatenate(batch_preds, axis=0))

    # C. 整理结果并计算不确定性
    # all_predictions 形状: [num_models, 数据总量, 输出维度]
    all_predictions = np.array(all_predictions)

    # 均值：作为最终的集成预测结果 (通常比单模型更准、更鲁棒)
    mean_prediction = np.mean(all_predictions, axis=0)

    # 方差：作为认知不确定性 (Epistemic Uncertainty)，反映模型间的分歧度
    uncertainty = np.var(all_predictions, axis=0)

    return mean_prediction, uncertainty


if __name__ == '__main__':
    data = load_data(batch_size=8)
    pool_df = data["pool_df"]
    pool_df = pool_df.sample(frac=1, random_state=42)
    n_val = int(len(pool_df) * 0.2)
    train_loader = df_to_loader(pool_df[:-n_val], shuffle=True)
    val_loader = df_to_loader(pool_df[-n_val:])

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ensemble":
        print("=== Training Ensemble ===")
        ensemble(train_loader, num_models=4, num_epochs=50)

        print("\n=== Ensemble Inference ===")
        mean_pred, uncert = ensemble_inference(val_loader)
        print("Mean:", mean_pred)
        print("Uncertainty:", uncert)
    else:
        print("=== MC Dropout ===")
        mean_prediction, uncertainty = mc_dropout(val_loader)
        print("Mean:", mean_prediction)
        print("Uncertainty:", uncertainty)

