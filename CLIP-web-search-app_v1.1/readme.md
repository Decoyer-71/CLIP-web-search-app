# CLIP-web-search_v1.1
## 1. 개요
> v1.0에서 데이터 증강을 통한 전이학습으로 모델의 성능을 높이고, 단순 단일개체 검색이 아닌 행동이나 패턴이 포함된 질문에 대한 이미지 식별 집중
> Streamlit 기반 단순 웹 구현에서 FastAPI를 적용한 배포로 변경

<br></br>
## 2. 기술 스택
- Python 3.10+
- openai/clip-vit-base-patch32
- Pytorch
- Albumentations
- FastAPI
<br></br>

## 3. 프로젝트 구조
<img width="338" height="512" alt="image" src="https://github.com/user-attachments/assets/6f766ad4-e1b8-4ba3-a280-aa63b050b24b" />

- clip_finetuned/: model_transfer_train.py를 통해 파인튜닝된 CLIP 모델과 토크나이저 파일(merges.txt, tokenizer.json 등)이 저장되는 폴더
- images_100/: 원본 이미지가 저장된 폴더
- augmented_images/: data_augmentation.py 스크립트로 생성된 증강 이미지들이 저장되는 하위 폴더
- static/: 웹 애플리케이션의 CSS, JavaScript와 같은 정적 파일들을 보관하는 폴더
- templates/: app.py에서 사용하는 index.html과 같은 웹페이지 템플릿 파일이 위치하는 폴더
- app.py: FastAPI를 사용하여 웹 애플리케이션을 실행하는 메인 스크립트
- data_augmentation.py: 원본 이미지들을 증강하여 학습 데이터셋을 늘리는 스크립트
- image_embedding.py: 파인튜닝된 모델을 사용하여 폴더 내의 모든 이미지에 대한 임베딩 벡터를 생성하고 저장하는 스크립트
- model_transfer_train.py: final_caption.csv 데이터를 이용해 사전 학습된 CLIP 모델을 파인튜닝하는 스크립트
- 기타 파일:
> - final_caption.csv: 원본 및 증강된 이미지의 캡션과 경로 정보가 포함된 최종 데이터 파일
> - image_embeddings_finetuned.npy: image_embedding.py로 생성된 이미지 임베딩 벡터 파일
> - image_paths_finetuned.pkl: 임베딩된 이미지들의 경로 목록을 저장한 파일
> - image_to_text_en.csv: 원본 이미지에 대한 영어 캡션 정보를 담고 있는 파일

<br></br>
## 4. 설치 및 실행방법
### 1) 프로젝트 다운로드
  - bash
<pre>
  <code>
    git clone https://github.com/Decoyer-71/CLIP-web-search-app.git

    cd CLIP-web-search-app_v1.1
  </code>
</pre>
<br></br>
### 2) 가상환경 설치 및 패키지 설치
#### (1) 가상환경 생성
  - bash
<pre>
  <code>
  python -m venv venv
  </code>
</pre>
#### (2) 가상환경 활성화
##### ※ Windows
  - bash
<pre>
  <code>
  .\venv\Scripts\activate
  </code>
</pre>
##### ※ macOS / Linux
  - bash
<pre>
  <code>
  source venv/bin/activate
  </code>
</pre>
#### (3) 필요 라이브러리 설치
  - bash
<pre>
  <code>
  pip install -r requirements.txt
  </code>
</pre>
<br></br>
### 3) 데이터 및 모델 준비
#### (1) 데이터 증강 및 CSV 생성(data_augmentation.py)
  - bash
<pre>
  <code>
  python data_augmentation.py
  </code>
</pre>
#### (2) CLIP 모델 파인튜닝(model_transfer_train.py)
  - bash
<pre>
  <code>
  python model_transfer_train.py
  </code>
</pre>
#### (3) 이미지 임베딩 생성(image_embedding.py)
  - bash
<pre>
  <code>
  python image_embedding.py
  </code>
</pre>
<br></br>
### 4) 웹 앱 실행
  - bash
<pre>
  <code>
  uvicorn app:app --reload
  </code>
</pre>
<br></br>
  - 실제 웹 페이지 예시화면
    <br></br>
    + ① 웹 실행 - 사용자 텍스트 입력준비
      
      <img width="954" height="592" alt="image" src="https://github.com/user-attachments/assets/bf2db2a0-675a-4293-b18a-d48f10d28565" />
      <br></br>
    + ② 웹 검색결과 - 상위 10개 유사도 결과 제공 및 이미지 전시
      - '달리는 고양이' 검색
      <img width="942" height="1017" alt="image" src="https://github.com/user-attachments/assets/33dfc3a4-ee0c-4f67-bf53-b18c1eec0211" />
      
      - '달리는 강아지' 검색
      <img width="945" height="1019" alt="image" src="https://github.com/user-attachments/assets/56d19c8c-3eb0-4cb5-ac92-92a53d139ff1" />
      


<br></br>

## 5. 평가 및 계획
- FastAPI는 배포를 위한 개념정도 학습후 gemini 도움을 받아 코드를 작성하였으며 이후 docker를 사용해 모델을 배포하는 연습을 하려함
- 전이학습으로 약간의 성능 향상을 보였으나, '달리는 개'검색에 '밥먹는 개'가 가장 유사도가 높게 나오는 상황이 발생 
- 차후에는 DINO를 활용해서 별도의 정답데이터 수집없이 이미지 학습을 통해 식별능력을 개선한 모델을 테스트해볼 예정




