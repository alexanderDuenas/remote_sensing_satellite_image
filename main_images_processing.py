import argparse

from app.src.modules.minio.noise_processing import salt_pepper_process
from app.src.modules.minio.segmentation_cilo_hsv import segment_hsv_img
from app.src.modules.minio.segmentation_vegetation_hsv import segmentation_vegetation_hsv



parser = argparse.ArgumentParser(
        description="Script for processing satellite images"
    )
    
parser.add_argument(
        "--rgb_img_noise",
        type=str,
        help="path to noise image rgb",
        default='/path/to/rgb_noise_img'
    )

parser.add_argument(
        "--path_seg_silo",
        type=int,
        help="path to rgb image for silo segmentation",
        default= '/path/to/rgb_satellite_img'
    )
    
parser.add_argument(
        "--path_seg_veg",
        type=int,
        help="path to raster image for vegetation segmentation",
        default= '/path/to/raster_satellite_img'
    )

parser.add_argument(
        "--path_output",
        type=str,
        help="Output folder for image processing",
        default= "path/images/destination"
    )

def run(config):

    '''
    Método para realizar o processamento das imagens RGB e Raster satelitais. 
    
    args: 

    rgb_img_noise = caminho que contem a imagem rgb com ruido
    ath_seg_silo = caminho que contem a imagem rgb para realizar a segmentação da classe silo
    path_seg_veg = caminho que contem a imagem raster para realizar a segmentação da classe vegetação
    
    '''
    salt_pepper_process(config.rgb_img_noise, config.path_output)
    segment_hsv_img(config.path_seg_silo, config.path_output)
    segmentation_vegetation_hsv(config.path_seg_veg, config.path_output)

if __name__ == '__main__':
    
    config = parser.parse_args()
    run(config)
    