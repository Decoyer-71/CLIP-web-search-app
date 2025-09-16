import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tqdm import tqdm

### 클래스 정의 ###
# --- pytorch에서 Dataset클래스 상속받아 커스텀데이터셋 만들기---
class ImageCaptionDataset(Dataset):
    def __init__(self, df, processor, image_base_path=''):
        self.df = df
        self.processor = processor
        self.image_base_path = image_base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_base_path, row['image_path']) if self.image_base_path else row['image_path']
        image = Image.open(image_path).convert(
            'RGB')  # 이미지 파일을 통일하기위해 RGB로 CONVERT(정규화 개념 - CLIP모델 입력 기대값은 RGB(3개 채널)이다.)
        caption = row['caption_en']

        # 프로세서로 이미지와 텍스트 임베딩 처리
        inputs = self.processor(text=caption,
                                images=image,
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True)

        # DataLoader에서 배치로 묶기 위채 차원 정리
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)

        return inputs

#####################################################

### 데이터 불러오기
base_dir = os.getcwd()
csv_path = os.path.join(base_dir, 'final_caption.csv')
save_path = os.path.join(base_dir, 'clip_finetuned')

final_df = pd.read_csv(csv_path)
final_df.set_index('idx', inplace=True, drop=False) # Keep 'idx' column
# print(f'총 데이터 개수 (원본 + 증강) : {len(final_df)}')
# print(f'\n--- 최종 데이터프레임 (상위 5개) ---')
# print(final_df.head())
# print(f'\n--- 증강 데이터 샘플 (하위 5개) ---')
# print(final_df.tail())

##### 모델 학습 ####

# -- 모델 준비 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 학습데이터 : 검증데이터 = 6:4
train_df, test_df = train_test_split(final_df, test_size=0.4, random_state=42)
print('훈련데이터 : 테스트 데이터  = ', len(train_df),':', len(test_df))

# 사전학습 CLIP 모델 및 프로세서 불러오기
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

## 데이터셋 로드
# train
train_dataset = ImageCaptionDataset(train_df, processor)
train_dataloader = DataLoader(train_dataset, batch_size=25, shuffle=True)

# test
test_dataset = ImageCaptionDataset(test_df, processor)
test_dataloader = DataLoader(test_dataset, batch_size=25, shuffle=True)

# 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr = 5e-5)
num_epochs = 30

# 조기종료 설정
patience = 5
patience_counter = 0
best_test_loss = np.inf

# -- 학습 및 검증 로직 --
train_losses = []
test_losses = []

for epoch in tqdm(range(num_epochs), desc = 'Epochs') :
    # -- 학습 부분 ---
    model.train() # 모델 학습 모드 설정
    total_train_loss = 0

    for batch in tqdm(train_dataloader, desc = f'Epoch {epoch+1}/{num_epochs} Training', leave = False) :
        # 배치 데이터를 device로 이동
        batch = {k:v.to(device) for k, v in batch.items()} # key : 'pixel_values', 'input_idx', 'attention_mask' / value : pt tensor

        # 모델 입력(출력, 손실 계산)
        outputs = model(**batch, return_loss = True)
        loss = outputs.loss

        # 역전파
        loss.backward()
        optimizer.step() # 파라미터 업데이트
        optimizer.zero_grad() # 기울기 초기화

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss) # 학습 epoch에 따른 loss 경과 저장

    # -- 검증 부분 --
    model.eval() # 모델 평가 모드 설정
    total_test_loss = 0

    with torch.no_grad() : # 그래디언트 계산 비활성화
        for batch in tqdm(test_dataloader, desc = f'Epoch {epoch+1}/{num_epochs} Testing', leave = False) :
            batch = {k:v.to(device) for k, v in batch.items()}

            # 입력
            outputs = model(**batch, return_loss = True)
            loss = outputs.loss

            total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_dataloader)
        test_losses.append(avg_test_loss)

        print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss : {avg_train_loss:.4f} | Test Loss : {avg_test_loss:.4f}')

        # 조기종료 로직
        if avg_test_loss < best_test_loss :
            best_test_loss = avg_test_loss
            patience_counter = 0

        else :
            patience_counter += 1

        if patience_counter >= patience :
            print('Early stopping triggered')
            break

# # -- 검증 시각화 --
# plt.figure(figsize = (10, 5))
# plt.plot(train_losses, label = 'Training Loss')
# plt.plot(test_losses, label = 'Testing Loss')
# plt.title('Training - Testing Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# -- 파인튜닝 모델 저장 --
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print(f'모델이 {save_path}에 저장되었습니다.')
