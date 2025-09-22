import numpy as np
import os, glob, pickle
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

###############################################################

def get_all_image_paths(base_dir):
    """원본 및 증강된 이미지 파일 경로를 모두 수집합니다."""
    img_path_dir = os.path.join(base_dir, 'images_100')
    augmented_img_path_dir = os.path.join(img_path_dir, 'augmented_images')
    
    all_image_files = []
    search_dirs = [img_path_dir, augmented_img_path_dir]
    extensions = ['*.jpg', '*.jpeg', '*.png']

    for directory in search_dirs:
        if not os.path.exists(directory):
            print(f"경고: 디렉토리가 존재하지 않습니다: {directory}")
            continue
        for ext in extensions:
            # 절대 경로 대신 상대 경로로 변환하여 저장합니다.
            for file_path in glob.glob(os.path.join(directory, ext), recursive=True):
                relative_path = os.path.relpath(file_path, start=base_dir).replace(os.sep, '/')
                all_image_files.append(relative_path)
            
    return sorted(list(set(all_image_files)))

def embedding_process(image_paths, model, processor, base_dir, batch_size=32):
    """이미지 경로 리스트를 받아 배치 단위로 임베딩을 생성합니다."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    image_embeddings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="이미지 임베딩 중"):
        batch_paths = image_paths[i:i+batch_size]
        try:
            images = [Image.open(os.path.join(base_dir, path)).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors='pt', padding=True).to(device)
            
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            image_embeddings.append(features.cpu())
        except Exception as e:
            print(f"오류 발생: {batch_paths} 처리 중. 건너뜁니다. 오류: {e}")
            
    if not image_embeddings:
        return np.array([])
        
    tensor_cat = torch.cat(image_embeddings, dim=0)
    return tensor_cat.numpy()

if __name__ == "__main__":
    base_dir = os.getcwd()
    model_path = os.path.join(base_dir, 'clip_finetuned')
    embedding_file = os.path.join(base_dir, 'image_embeddings_finetuned.npy')
    paths_file = os.path.join(base_dir, 'image_paths_finetuned.pkl')
    
    all_image_files = get_all_image_paths(base_dir)
    print(f'총 {len(all_image_files)}개 이미지 파일을 찾았습니다.')
    
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    
    np_array = embedding_process(all_image_files, model, processor, base_dir=base_dir)
    
    if np_array.size > 0:
        np.save(embedding_file, np_array)
        print(f'임베딩 배열 저장완료! 최종배열 형태 : {np_array.shape}')
        
        with open(paths_file, 'wb') as f:
            pickle.dump(all_image_files, f)
        print(f'이미지 파일 경로를 "{os.path.basename(paths_file)}"파일로 저장했습니다.')
    else:
        print("생성된 임베딩이 없어 파일을 저장하지 않습니다.")
