from fastapi import FastAPI

# FastAPI 인스턴스 생성(app : 모든 상호작용의 주요 지점으로 app이 아니라 my_app으로 정의하면 terminal에서 uvicorn main:my_app --reload로 실행해야 함)
app = FastAPI()

# 
@app.get("/")
async def root():
    return {"message": "Hello World"}