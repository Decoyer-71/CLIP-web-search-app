import pandas as pd
import torch
import numpy as np
import pickle
from transformers import CLIPProcessor, CLIPModel
import streamlit as st
from PIL import Image

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
def similarity_caluation(text_embedding, image_embedding, image_path):
    sim_score_lst = []
    for idx in range(len(image_embedding)):
        similarity = torch.nn.functional.cosine_similarity(text_embedding, image_embedding[idx])
        sim_score_lst.append(round(similarity.item(), 3))

    similarity_df = pd.DataFrame(
        {
            'Score' : sim_score_lst,
            'Image_path' : image_path
        }
    )
    similarity_df = similarity_df.sort_values('Score', ascending=False).head(5)
    similarity_df['Rank'] = [num + 1 for num in range(len(similarity_df))]
    similarity_df.set_index('Rank', inplace=True)

    return similarity_df

#################UI 구성###############################

# st.title('이미지 검색 엔진')
st.markdown("<h1 style='text-align: center; color: black;'>이미지 검색 엔진</h1>", unsafe_allow_html=True)

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
        inputs = processor(text = text_query, return_tensors = 'pt', padding = True)

        # get_text_features 메서드로 텍스트 임베딩만 추출
        text_features = model.get_text_features(**inputs)

    st.success('텍스트 임베딩 생성완료')

    # 불러오기 한 이미지 임베딩 numpy 배열 -> tensor 변환
    embedding_tensor = torch.from_numpy(image_embedding)

    # 텍스트-이미지 유사도 계산
    sim_top5 = similarity_caluation(text_features, embedding_tensor, image_path)

    # 유사도 결과 전시
    # st.header('상위 5개 이미지 유사도 결과')
    st.markdown("<h2 style='text-align: center; color: black;'>상위 5개 이미지 유사도 결과</h1>", unsafe_allow_html=True)
    st.dataframe(sim_top5['Score'])

    # 상위 5개 이미지 불러오기
    top5_path = [path for path in sim_top5['Image_path']]

    for num in range(len(top5_path)) :
        img_object = Image.open(top5_path[num])
        st.image(img_object, caption = f'후보 {num + 1}순위')
