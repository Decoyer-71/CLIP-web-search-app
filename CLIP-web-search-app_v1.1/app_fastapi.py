import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import pickle
from PIL import Image
import base64
from io import BytesIO

from fastapi import FastAPI, Request #request는 사용하고 있음
from fastapi.responses import HTMLResponse

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import CLIPProcessor, CLIPModel
from contextlib import asynccontextmanager

###################함수 정의######################

def get_image_base64(image_path):
    """이미지 파일을 Base64 문자열로 인코딩합니다."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def similarity_search(text_embedding, image_embeddings, image_paths, top_k=10):
    """
    텍스트 임베딩과 이미지 임베딩 간의 코사인 유사도를 계산하여 상위 K개 결과를 반환합니다.
    """
    # 이미지 임베딩을 텐서로 변환
    image_embeddings_tensor = torch.from_numpy(image_embeddings)
    
    # 코사인 유사도 계산 (모든 이미지에 대해 한 번에 계산)
    # text_embedding: [1, 512], image_embeddings_tensor: [N, 512]
    # 유사도 점수는 [N] 형태의 텐서로 계산됩니다.
    similarities = torch.nn.functional.cosine_similarity(text_embedding, image_embeddings_tensor)
    
    # 상위 K개의 점수와 인덱스 찾기
    top_k_scores, top_k_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for i in range(top_k):
        score = top_k_scores[i].item()
        path = image_paths[top_k_indices[i]]
        results.append({
            "rank": i + 1,
            "score": round(score, 4),
            "path": path,
            "image_base64": get_image_base64(path)
        })
        
    return results

################# FastAPI 설정 ###############################

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 모델 로드
    base_dir = os.getcwd()
    model_path = os.path.join(base_dir, 'clip_finetuned')
    paths_file = os.path.join(base_dir, 'image_paths_finetuned.pkl')
    embedding_file = os.path.join(base_dir, 'image_embeddings_finetuned.npy')
    
    print("모델 및 데이터를 메모리에 로드중...")
    ml_models['model'] = CLIPModel.from_pretrained(model_path)
    ml_models['processor'] = CLIPProcessor.from_pretrained(model_path)
    with open(paths_file, 'rb') as f:
        ml_models['image_paths'] = pickle.load(f)
    ml_models['image_embeddings'] = np.load(embedding_file)
    print("✅ 모델 및 데이터 로딩완료")
    print(f"총 {len(ml_models['image_paths'])}개 이미지 경로, {ml_models['image_embeddings'].shape} 형태의 임베딩 벡터 준비완료")
    yield
    # 앱 종료 시 모델 정리 (필요 시)
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

static_dir = os.path.join(os.getcwd(), "static")
templates_dir = os.path.join(os.getcwd(), "templates")
# 정적 파일(CSS, JS) 및 템플릿 설정
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 페이지를 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search")
async def search_images(request: Request, query: str):
    """텍스트 쿼리를 받아 이미지 검색 결과를 HTML 조각으로 렌더링하여 반환합니다."""
    if not query:
        return HTMLResponse("") # 검색어가 없으면 빈 HTML을 반환

    processor = ml_models['processor']
    model = ml_models['model']
    
    # 텍스트 임베딩 생성
    with torch.no_grad():
        inputs = processor(text=query, return_tensors='pt', padding=True, truncation=True)
        text_features = model.get_text_features(**inputs)

    # 유사도 검색 수행
    search_results = similarity_search(
        text_features,
        ml_models['image_embeddings'],
        ml_models['image_paths'],
        top_k=10
    )

    # 검색 결과를 HTML 조각으로 렌더링하여 반환
    return templates.TemplateResponse("partials/search_results.html", {
        "request": request, 
        "results": search_results
    })
