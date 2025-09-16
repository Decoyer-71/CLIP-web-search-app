import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import numpy as np
import os, glob
from PIL import Image
import cv2, csv
import albumentations as A
from tqdm import tqdm
import gc


### 함수정의 ###
def create_image_path_df(image_dir):
    """지정된 디렉토리에서 이미지 파일 경로를 찾아 데이터프레임을 생성합니다."""
    image_path_list = glob.glob(os.path.join(image_dir, '*.*'))
    if not image_path_list:
        print(f"경고: '{image_dir}' 디렉토리에서 이미지를 찾을 수 없습니다.")
        return pd.DataFrame({'idx': [], 'image_path': []})

    # 파일 이름에서 확장자를 제외하고 인덱스로 사용
    idx_list = [os.path.splitext(os.path.basename(p))[0] for p in image_path_list]
    
    path_df = pd.DataFrame(
        {
            'idx' : idx_list,
            'image_path' : image_path_list
        }
    )
    path_df['idx'] = pd.to_numeric(path_df['idx'], errors='coerce')
    return path_df.dropna(subset=['idx']).astype({'idx': 'int'})

def process_image(row, num_augmentations, transform_pipeline, output_dir):
    """
    단일 이미지에 대한 모든 처리(읽기, 리사이즈, 증강, 저장)을 수행하고,
    csv에 쓸 데이터 행들을 리스트로 반환
    """
    try:
        original_image_path = row['image_path']
        caption = row['caption_en']
        original_idx = row['idx']

        # opencv로 이미지 bgr형식으로 호출
        image = cv2.imread(original_image_path)

        # 이미지 불러오기 실패시 건너뛰기
        if image is None:
            print(f'warning : 이미지를 불러올 수 없습니다. 건너뜁니다. : {original_image_path}')
            return []  # 빈 리스트 반환

        ### 이미지 리사이즈 ###
        # 이미지 리사이즈로 메모리 사용량 줄이기
        image = cv2.resize(image, (224, 224))
        ######################

        # Albumentations는 rbg형식으로 사용하므로 색상 채널 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_rows = []

        for i in range(num_augmentations):
            # 정의된 파이프라인에 따라 이미지 변한 적용
            augmented = transform_pipeline(image=image)
            augmented_image = augmented['image']

            # 새로운 파일 이름을 생성하고 증강된 이미지를 저장
            new_idx = f'{original_idx}_aug_{i + 1}'
            new_filename = f"{new_idx}.jpg"
            new_image_path = os.path.join(output_dir, new_filename)

            # opencv는 bgr형식 저장이므로, 다시 rgb에서 bgr로 변환
            cv2.imwrite(new_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

            # csv에 쓸 행(row) 데이터를 리스트에 추가
            augmented_rows.append([new_idx, caption, new_image_path])

        return augmented_rows

    except Exception as e:
        print(f'Error : {original_image_path} 처리 중 오류 발생 : {e}')
        return []  # 오류 발생 시 빈 리스트 반환

#####################################################

##### 데이터 증강 작업 #####

# caption csv 불러오기
base_dir = os.getcwd()
original_image_dir = os.path.join(base_dir, 'images_100')
caption_csv_path = os.path.join(base_dir, 'image_to_text_en.csv')

if not os.path.exists(caption_csv_path):
    raise FileNotFoundError(f"캡션 파일이 존재하지 않습니다: {caption_csv_path}")
caption_df = pd.read_csv(caption_csv_path)

# image_path 불러온 후 데이터프레임 형성
path_df = create_image_path_df(original_image_dir)

# cation과 image_path_list 합체
if path_df.empty:
    raise ValueError(f"'{original_image_dir}'에 유효한 이미지 파일이 없습니다.")
df = pd.merge(caption_df, path_df, how='inner', on='idx')
df.reset_index(drop = True, inplace = True)
print(f'원본 데이터 개수: {len(df)}')

### 1. 이미지 데이터 증강 설정
# 증강 이미지 저장 폴더 생성
if not df.empty :
    augmented_image_dir = os.path.join(os.path.dirname(df['image_path'].iloc[0]),
                                       'augmented_images')
    os.makedirs(augmented_image_dir, exist_ok = True)
else :
    print('오류 : 원본 데이터프레임(df)이 비어있습니다. 이미지 경로를 확인할 수 없습니다.')
    augmented_image_dir = './augmented_images' # 임시 기본경로
    os.makedirs(augmented_image_dir, exist_ok = True)

# Albumentations를 사용해 변환 파이프라인 정의
transform = A.Compose([
    A.HorizontalFlip(p=0.5), # 50%확률로 좌우 반전
    A.Rotate(limit = 30, p=0.7), # 30도 내에서 70%확률로 회전
    A.RandomBrightnessContrast(brightness_limit = 0.2,
                              contrast_limit = 0.2,
                              p=0.8), # 80%확률로 밝기/대비 조절
    A.GaussNoise(p=0.5) # 50%확률로 가우시안 노이즈 추가
])

### 2. 최종 csv파일 생성 및 데이터 기록
final_csv_path = os.path.join(base_dir, 'final_caption.csv')
num_augmentations_per_image = 10

with open(final_csv_path, 'w', newline = '', encoding = 'utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['idx', 'caption_en', 'image_path'])

    # 원본 데이터 기록
    print('원본 데이터를 csv파일에 쓰는 중...')
    for index, row in tqdm(df.iterrows(), total = len(df)):
        writer.writerow([row['idx'], row['caption_en'], row['image_path']])

    # 증강 데이터를 함수를 통해 처리하고 바로 기록
    print('\n 데이터 증강 및 csv 기록 시작...')
    for index, row in tqdm(df.iterrows(), total = len(df)):
        # 함수를 호출하여 이미지 처리
        new_rows = process_image(row, num_augmentations_per_image, transform, augmented_image_dir)

        # 함수가 반환한 새로운 행들을 csv에 기록
        if new_rows :
            writer.writerows(new_rows)

    # 루프 종료 후 가비지 컬렉터 호출(안정성을 위한 옵션)
    gc.collect()
