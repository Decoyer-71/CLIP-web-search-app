import pandas as pd
import numpy as np
import os, glob, pickle
from PIL import Image

# cLIP 모델 불러오기
import torch
from transformers import CLIPProcessor, CLIPModel

###############################################################
# 폴더에서 이미지 데이터 확인
path_dir = os.path.join(os.getcwd(), 'images')

jpg_files = glob.glob(path_dir + '/*.jpg')
print(f'총 {len(jpg_files)}개 이미지 파일을 찾았습니다.')

def Embedding_process(image_paths):
    # hugging face에서 clip 사전학습 모델 불러오기
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # CLIP 프로세서로 이미지 임베딩
    image_embeddings = []
    for image_path in image_paths:
        # PIL라이브러리로 이미지 열기
        image = Image.open(image_path)

        # 각 이미지를 모델 입력형태로 가공
        inputs = processor(images=image, return_tensors='pt', padding=True)

        # 모델에 입력값을 넣어 이미지 임베딩 추출
        with torch.no_grad():  # torch.no_grad() : 불필요한 연산을 막아 속도 향상 및 메모리 절약
            features = model.get_image_features(**inputs)

        # 추출된 임베딩을 리스트에 추가
        image_embeddings.append(features)

    # 임베딩 벡터 .npy로 저장
    tensor_cat = torch.cat(image_embeddings, dim = 0) # ex) (1, 512) -> (N, 512)로 합쳐짐

    # 텐서를 CPU로 이동 후 numpy배열 변환 : numpy변환이 cpu에서만 가능
    numpy_array = tensor_cat.cpu().numpy()

    return numpy_array

# 이미지 임베딩
np_array = Embedding_process(jpg_files)

# .npy파일 저장
np.save('image_embeddings.npy', np_array)
print(f'임베딩 배열 저장완료! 최종배열 형태 : {np_array.shape}')

# 이미지 파일경로 .pkl형태로 저장
with open('image_paths.pkl', 'wb') as f:
    pickle.dump(jpg_files, f)
print('이미지 파일 경로를 "image_paths.pkl"파일로 저장했습니다.')

