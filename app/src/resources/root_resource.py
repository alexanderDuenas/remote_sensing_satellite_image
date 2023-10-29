from fastapi import APIRouter

root_api = APIRouter()


@root_api.get("/")
async def root():
    return {"message": "Template Computer Vision"}
