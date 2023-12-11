import json
import numpy as np
import os
import glob
import cv2
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import ipywidgets as widgets
# 파일 경로 설정
image_root_path = 'C:/Users/KimDoHyum/Desktop/folders/자세교정 ai/013.피트니스자세_sample/data/realData'
label_root_path = 'C:/Users/KimDoHyum/Desktop/folders/자세교정 ai/013.피트니스자세_sample/data/labalData'
reductionX = 480
reductionY = 270
# 데이터 로딩 함수
def get_image_path_from_key(img_key):
    parts = img_key.split('/')
    relative_path = os.path.join(*parts[1:])
    return os.path.join(image_root_path, relative_path)

def load_label_data(label_path):
    with open(label_path, 'r') as file:
        label_data = json.load(file)
    all_image_paths, all_joint_data = [], []
    for frame in label_data['frames']:
        for view in ['view1', 'view2', 'view3', 'view4', 'view5']:
            if view in frame and frame[view]['active'] == "Yes":
                img_key = frame[view]['img_key']
                image_path = get_image_path_from_key(img_key)
                joints = frame[view]['pts']
                joint_positions = [np.array([joint['x'], joint['y']]) for joint in joints.values()]
                all_image_paths.append(image_path)
                all_joint_data.append(np.concatenate(joint_positions, axis=0))
    return all_image_paths, all_joint_data

# 이미지와 관절 데이터 로드
label_paths = glob.glob(os.path.join(label_root_path, '*.json'))
all_image_paths, all_joint_data = [], []
for label_path in label_paths:
    image_paths, joint_data = load_label_data(label_path)
    all_image_paths.extend(image_paths)
    all_joint_data.extend(joint_data)

# PyTorch Dataset 클래스
class JointDataset(Dataset):
    def __init__(self, image_paths, joint_data, transform=None):
        self.image_paths = image_paths
        self.joint_data = joint_data
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        print(idx)
        # 이미지 로딩 및 변환 (cv2 사용)
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 이미지 리사이징
        resized_image = cv2.resize(image, (reductionX, reductionY))
        
        # cv2 이미지를 PIL 이미지로 변환
        resized_image = Image.fromarray(resized_image)
        # 좌표 스케일링
        scale_x = reductionX / image.shape[1]
        scale_y = reductionY / image.shape[0]
        joints = self.joint_data[idx]
        adjusted_coordinates = np.empty_like(joints)
        for i in range(0, len(joints), 2):
            adjusted_coordinates[i] = joints[i] * scale_x
            adjusted_coordinates[i + 1] = joints[i + 1] * scale_y

        # 텐서 변환
        joints = torch.from_numpy(adjusted_coordinates).float()

        # 이미지 변환
        if self.transform:
            resized_image = self.transform(resized_image)

        return resized_image, joints.view(-1)
        
class PoseCNN(nn.Module):
    def __init__(self, num_joints):
        super(PoseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # 예시: 이미지 크기가 480x270이고, MaxPool2d를 세 번 사용하면 최종 특성 맵의 크기는 60x33
        self.fc1 = nn.Linear(128 * 60 * 33, 512)
        self.fc2 = nn.Linear(512, num_joints * 2)  # 각 관절의 x, y 좌표

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 60 * 33)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
# GPU 사용 가능 여부 확인
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 이미지 변환 설정
transform = transforms.Compose([
    transforms.Resize((reductionY, reductionX)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])
# 모델 학습 함수
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=10):
    print("학습 시작")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, joints in dataloader:
            images, joints = images.to(device), joints.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, joints)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}]')

        scheduler.step()
num_joints = 24
model = PoseCNN(num_joints).to(device)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, dataloader):
    model.eval()  # 모델을 평가 모드로 설정
    predictions, actuals = [], []

    with torch.no_grad():
        for images, joints in dataloader:
            images, joints = images.to(device), joints.to(device)
            outputs = model(images)
            predictions.append(outputs.cpu())
            actuals.append(joints.cpu())

    predictions = torch.cat(predictions, dim=0).numpy()
    actuals = torch.cat(actuals, dim=0).numpy()

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    return mse, mae, rmse, r2
# 메인 코드 실행


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def evaluate_model(model, dataloader, criterion):
    model.eval()  # 모델을 평가 모드로 설정
    predictions, actuals = [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, joints in dataloader:
            images, joints = images.to(device), joints.to(device)
            outputs = model(images)

            # 평가를 위한 손실 계산
            loss = criterion(outputs, joints)
            total_loss += loss.item()

            predictions.append(outputs.cpu())
            actuals.append(joints.cpu())

    predictions = torch.cat(predictions, dim=0).numpy()
    actuals = torch.cat(actuals, dim=0).numpy()

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    average_loss = total_loss / len(dataloader)  # 평균 손실 계산

    return mse, mae, rmse, r2, average_loss

# 평가 함수 실행


# 모델 평가 실행

# 모델 평가 실행
# 메인 코드 실행
if __name__ == '__main__':
    num_joints = 24
    model.load_state_dict(torch.load("C:/Users/KimDoHyum/model654321.pth"))
    model.eval()
    print("데이터 불러오기")
    try :
        train_dataset = JointDataset(all_image_paths, all_joint_data, transform=transform)        
        train_loader = DataLoader(dataset=train_dataset, batch_size=10, persistent_workers=True, num_workers=12)
    finally:
        print("데이터 불러오기 완료")
        
        # 모델 평가 실행
        criterion = nn.MSELoss()  # 또는 모델 학습에 사용한 다른 손실 함수
        mse, mae, rmse, r2, avg_loss = evaluate_model(model, train_loader, criterion)
        print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Avg Loss: {avg_loss:.4f}')
        # 모델 불러오기 및 관절 검출 크기 줄여서 띄우기.
       
        # criterion = nn.MSELoss()
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        # train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=20)# 모델 저장
        # torch.save(model.state_dict(), 'model10987654321.pth')
        





# # # 모델 불러오기 및 관절 검출 크기 줄여서 띄우기.

# def calculate_accuracy(model, dataloader, threshold=10):
#     total_joints = 0
#     correct_joints = 0

#     for images, true_joints in dataloader:
#         images = images.to(device)
#         true_joints = true_joints.to(device)

#         with torch.no_grad():
#             predicted_joints = model(images).view(-1, num_joints, 2)
#             true_joints = true_joints.view(-1, num_joints, 2)

#             distances = torch.sqrt(torch.sum((predicted_joints - true_joints) ** 2, dim=2))
#             correct_joints += torch.sum(distances < threshold)
#             total_joints += distances.numel()

#     accuracy = correct_joints.float() / total_joints
#     return accuracy

# # Calculate the accuracy
# accuracy = calculate_accuracy(model, train_loader)
# print(f"Model Accuracy: {accuracy:.4f}")


# def detect_joints(image_path):
#     image = Image.open(image_path)
#     # 이미지 리사이징
#     resized_image = image.resize((reductionX, reductionY))

#     image = transform(resized_image).unsqueeze(0).to(device)  # 이미지를 GPU로 이동
#     with torch.no_grad():
#         outputs = model(image)
#     joint_positions = outputs.view(-1, num_joints, 2)
#     joint_positions = joint_positions.cpu().numpy()  # 결과를 CPU로 이동하여 넘파이 배열로 변환
#     return joint_positions

# # 이미지에서 관절 검출 및 시각화
# image_path = "C:/Users/KimDoHyum/Desktop/folders/자세교정 ai/013.피트니스자세_sample/data/realData/1/C/033-1-1-21-Z17_C/033-1-1-21-Z17_C-0000020.jpg"
# joint_positions = detect_joints(image_path)
# print(joint_positions)

# # 관절 검출 및 시각화 함수
# def visualize_joints_interactive(image_path, joint_positions):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     resized_image = cv2.resize(image, (reductionX, reductionY))

#     def plot_image(scale=1.0):
#         fig, ax = plt.subplots(figsize=(16*scale, 9*scale))
#         ax.imshow(resized_image)
#         radius = 2/scale  # 점의 크기를 스케일에 반비례하게 조정
#         for joint in joint_positions[0]:
#             circle = patches.Circle((joint[0], joint[1]), radius=radius, color='red')
#             ax.add_patch(circle)
#         plt.show()

#     scale_slider = widgets.FloatSlider(value=1.0, min=0.5, max=2.0, step=0.1, description='Scale:')
#     widgets.interactive(plot_image, scale=scale_slider)

# # 모델 검증 및 정확도 계산 함수
# def validate_model(model, dataloader, threshold=10):
#     model.eval()
#     correct_predictions = 0
#     total_predictions = 0

#     for images, joints in dataloader:
#         images, joints = images.to(device), joints.to(device)

#         with torch.no_grad():
#             outputs = model(images)
#             outputs = outputs.view(-1, num_joints, 2)
#             joints = joints.view(-1, num_joints, 2)

#             distances = torch.sqrt(torch.sum((outputs - joints) ** 2, dim=2))
#             correct_predictions += torch.sum(distances < threshold)
#             total_predictions += distances.numel()

#     accuracy = correct_predictions.float() / total_predictions
#     print(f'Validation Accuracy: {accuracy:.4f}')
# def calculate_accuracy(model, dataloader, threshold=10):
#     total_joints = 0
#     correct_joints = 0

#     for images, true_joints in dataloader:
#         images = images.to(device)
#         true_joints = true_joints.to(device)

#         with torch.no_grad():
#             predicted_joints = model(images).view(-1, num_joints, 2)
#             true_joints = true_joints.view(-1, num_joints, 2)

#             distances = torch.sqrt(torch.sum((predicted_joints - true_joints) ** 2, dim=2))
#             correct_joints += torch.sum(distances < threshold)
#             total_joints += distances.numel()

#     accuracy = correct_joints.float() / total_joints
#     return accuracy

# # Calculate the accuracy
# accuracy = calculate_accuracy(model, train_loader)
# print(f"Model Accuracy: {accuracy:.4f}")


# # 사용 예시
# visualize_joints_interactive(image_path, joint_positions)
