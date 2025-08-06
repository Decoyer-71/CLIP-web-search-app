# CLIP-web-search_v1.0
## 1. 개요
> CLIP 모델을 기반으로 하여 사용자의 텍스트 입력에 적합한 이미지를 검색하는 앱
>  - 'openai/clip', 'pytorch', 'Streamlit'을 사용한 간단한 웹 인터페이스 제공

## 2. 기술 스택
- Python 3.10+
- openai/clip-vit-base-patch32
- Pytorch
- Streamlit(웹 UI)

## 3. 프로젝트 구조
  - app.py : Streamlit 웹앱 실행 및 사용자 이미지 검색 로직 구현
  - image_embedding.py : 수집된 이미지 데이터에 대하여 CLIP processor를 통한 이미지 임베딩 처리
  - image_embeddings.npy : image_embedding.py에서 임베딩된 이미지 벡터의 numpy 배열형태 저장파일
  - image_paths.pkl : 이미지가 존재하는 디렉토리 경로가 리스트로 저장된 파일
  - images.zip : google에서 수집한 dog, cat, man, human, car, train 등 50개 이미지의 zip파일
  - requirement.txt : 필요한 라이브러리 목록

## 4. 설치 및 실행방법
### 1) 저장소 클론 
  - bash
<pre>
  <code>
    git clone https://github.com/Decoyer-71/sentiment_classification.git 

    cd sentiment_classification
  </code>
</pre>

### 2) 가상환경 설치 및 패키지 설치
  - bash
<pre>
  <code>
  pip install -r requirements.txt
  </code>
</pre>

### 3) images.zip 압축풀기
  - bash
<pre>
  <code>
  unzip images.zip
  </code>
</pre>
- 또는 폴더에서 직접 압축풀기

### 4) 웹 앱 실행
  - bash
<pre>
  <code>
  streamlit run app.py
  </code>
</pre>
  - 실행결과 
<img width="650" height="450" alt="image" src="https://github.com/user-attachments/assets/c041b46c-d02f-40c1-a7ba-4ecbac713ced" />

     개뼈다귀라는 말은 사실 긍정적인 의미일지도 모른다.


## 5. 향후계획

