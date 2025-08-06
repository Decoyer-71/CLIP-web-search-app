import streamlit as st
import torch
import numpy as np
import pickle
from transformers import CLIPProcessor, CLIPModel
import streamlit as st


###################함수 정의######################

@st.cache_resource
def load_model_and_data():
    """
    CLIP 모델, 프로세서, 이미지 임베딩과 경로를 불러옵니다.
    :return: model, processor, image_embedding, image_paths
    """

    # CLIP 모델과 프로세서 불러오기
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # 이미지 특징 벡터(.npy), 경로(.pkl) 불러오기
    with open('image_paths.pkl', 'rb') as f:
        image_paths = pickle.load(f)

    image_embedding = np.load('image_embeddings.npy')

    return model, processor, image_embedding, image_paths

# 텍스트 임베딩 - 이미지 임베딩 유사도 계산 함수
def similarty_caluation(text_embedding, image_embedding):
    sim_score_lst = []
    for num in range(len(image_embedding)):
        torch.nn.functional.cosin

#################UI 구성###############################

st.title('이미지 검색 엔진')

st.write('모델 및 데이터를 메모리에 로드중...')

model, processor, image_embedding, image_path = load_model_and_data()

st.success('✅ 모델 및 데이터 로딩완료')
st.info(f'총 {len(image_path)}개 이미지 경로, {image_embedding.shape} 형태의 임베딩 벡터 준비완료')


##############사용자 검색 로직 구현#############################

# 1) 사용자 텍스트 입력받기
text_query = st.text_input('검색어를 입력하세요 :', placeholder='ex) an woman of cartoon style' )

# 2) 검색로직 실행
# 텍스트 임베딩 처리
if text_query :
    st.write(f'{text_query}(으)로 이미지를 검색합니다...')

    # 단일 검색어를 모델에 넣어 처리
    with torch.no_grad():
        inputs = processor(text = text_query, return_tensor = 'pt', padding = True)

        # get_text_features 메서드로 텍스트 임베딩만 추출
        text_features = model.get_text_features(**inputs)

    st.success('텍스트 임베딩 생성완료')

# 불러오기 한 이미지 임베딩 numpy 배열 -> tensor 변환
embedding_tensor = torch.from_numpy(image_embedding)


