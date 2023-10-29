import logging

import uvicorn
from fastapi import FastAPI

from app.src.modules.__index__ import init_modules
from config import api_config

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger('template_vc')

app = FastAPI(
    title="Template VC",
    description="Template Project for Computer Vision Area",
    version="0.0.1",

)
init_modules(app)

if __name__ == '__main__':
    logger.info("### Application Starts")
    uvicorn.run(
        "main:app",
        host=api_config.get('SERVER_HOST'),
        port=api_config.get('SERVER_PORT')
    )
    logger.info("### Application Ends")
