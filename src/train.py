import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import time

# --- BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU ---
print("--- Đang chuẩn bị dữ liệu... ---")
df = pd.read_csv('cleaned_data.csv')

df = df.dropna()
df = df[(df['age'] > 0) & (df['age'] < 120)]

le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'].astype(str))
le_drug = LabelEncoder()
df['drugname'] = le_drug.fit_transform(df['drugname'].astype(str))

scaler = StandardScaler()
df[['age', 'wt']] = scaler.fit_transform(df[['age', 'wt']])

X = df[['age', 'sex', 'wt', 'drugname']].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

# --- BƯỚC 2: KIẾN TRÚC MÔ HÌNH ---
class ResBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class DrugResNet(nn.Module):
    def __init__(self, num_drugs):
        super().__init__()
        self.embed = nn.Embedding(num_drugs, 16)
        self.input_layer = nn.Linear(3 + 16, 64)
        self.res_layers = nn.Sequential(ResBlock(64), ResBlock(64))
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        num_feats = x[:, :3]
        drug_id = x[:, 3].long()
        embedded_drug = self.embed(drug_id)
        x = torch.cat([num_feats, embedded_drug], dim=1)
        x = torch.relu(self.input_layer(x))
        x = self.res_layers(x)
        return self.output(x)

model = DrugResNet(len(le_drug.classes_)).to(device)

# --- BƯỚC 3: HUẤN LUYỆN ---
# Điều chỉnh trọng số để xử lý imbalance (tăng trọng số cho lớp dương tính)
weights = torch.tensor([1.0, 20.0]).to(device) 

criterion = nn.CrossEntropyLoss(weight=weights)


optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

EPOCHS = 40

print(f"--- Bắt đầu huấn luyện trên: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ---")
print(f"--- Sẽ chạy tổng cộng {EPOCHS} Epochs ---")

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # Quá trình Train
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    avg_train_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train
    
    scheduler.step()
    
    # Quá trình Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_t)
        val_loss = criterion(val_outputs, y_test_t)
        _, val_preds = torch.max(val_outputs, 1)
        val_acc = (val_preds == y_test_t).sum().item() / len(y_test_t)
        
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss.item())
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    duration = time.time() - start_time
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_train_loss:.4f}/{val_loss.item():.4f} | Acc: {train_acc:.4f}/{val_acc:.4f}")

# --- BƯỚC 4: VẼ BIỂU ĐỒ CHÍNH ---
print("\n--- Đang tạo biểu đồ... ---")
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Đặt scale trục Y cho Accuracy để biểu đồ 
axs[1].set_ylim([0.90, 1.0])

# 1. Biểu đồ Learning Curve (Loss)
axs[0].plot(train_losses, label='Train Loss', color='blue', marker='o', markersize=3)
axs[0].plot(val_losses, label='Validation Loss', color='orange', marker='x', markersize=3)
axs[0].set_title('Learning Curve (Loss)')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.7)

# 2. Biểu đồ Accuracy
axs[1].plot(train_accs, label='Train Accuracy', color='green', marker='o', markersize=3)
axs[1].plot(val_accs, label='Validation Accuracy', color='red', marker='x', markersize=3)
axs[1].set_title('Model Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('accuracy_loss_charts.png')
print("--- Đã lưu biểu đồ vào file: accuracy_loss_charts.png ---")
plt.show()