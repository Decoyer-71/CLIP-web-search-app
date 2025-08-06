# CLIP-web-search_v1.0
## 1. 개요
> CLIP 모델을 기반으로 하여 사용자의 텍스트 입력에 적합한 이미지를 검색하는 앱
>  - 'openai/clip', 'pytorch', 'Streamlit'을 사용한 간단한 웹 인터페이스 제공
<br></br>
## 2. 기술 스택
- Python 3.10+
- openai/clip-vit-base-patch32
- Pytorch
- Streamlit(웹 UI)
<br></br>
## 3. 프로젝트 구조
  - app.py : Streamlit 웹앱 실행 및 사용자 이미지 검색 로직 구현
  - image_embedding.py : 수집된 이미지 데이터에 대하여 CLIP processor를 통한 이미지 임베딩 처리
  - image_embeddings.npy : image_embedding.py에서 임베딩된 이미지 벡터의 numpy 배열형태 저장파일
  - image_paths.pkl : 이미지가 존재하는 디렉토리 경로가 리스트로 저장된 파일
  - images.zip : google에서 수집한 dog, cat, man, human, car, train 등 50개 이미지의 zip파일
  - requirement.txt : 필요한 라이브러리 목록
<br></br>
## 4. 설치 및 실행방법
### 1) 저장소 클론 
  - bash
<pre>
  <code>
    git clone https://github.com/Decoyer-71/CLIP-web-search-app.git

    cd CLIP-web-search-app_v1.0
  </code>
</pre>
<br></br>
### 2) 가상환경 설치 및 패키지 설치
  - bash
<pre>
  <code>
  pip install -r requirements.txt
  </code>
</pre>
<br></br>
### 3) images.zip 압축풀기
  - bash
<pre>
  <code>
  unzip images.zip
  </code>
</pre>
- 또는 폴더에서 직접 압축풀기
<br></br>
### 4) 웹 앱 실행
  - bash
<pre>
  <code>
  streamlit run app.py
  </code>
</pre>
<br></br>
  - ① 웹 실행 - 사용자 텍스트 입력준비
<img width="650" height="450" alt="image" src="https://github.com/user-attachments/assets/1e4625ab-937a-417a-b734-114623ee0842" />
<br></br>
- ② 웹 검색결과 - 상위 5개 유사도 결과 제공 및 이미지 전시
<img width="650" height="900" alt="image" src="https://github.com/user-attachments/assets/40c70908-6692-464f-8935-55cfcbc97ab0" />
<br></br>

## 5. 향후계획
- (단기) 전이학습을 통한 모델개선
   + 텍스트와 이미지간 디테일한 연결성 부족(ex : 실제여성을 검색했지만, 만화스타일의 여성유사도가 가장 높음)
- (단기) 한국어 기반 모델 사용
   + openai/clip의 경우 한국어 인식이 부족하여 텍스트-이미지 매칭 정확도 낮음
   + 영어보다는 한국어가 편하니까
- (중기) 웹 크롤링 기반 이미지를 수집하는 모델 개발
  + 이미지 데이터 수작업 수집의 한계 



