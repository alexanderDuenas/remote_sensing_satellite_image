import cv2
import matplotlib.pyplot as plt
import numpy as np

def salt_pepper_process(path_image_noise, output_path):
    
    '''
    Função que tem como objetivo fazer um processamento duma imagem de entrada contendo ruido (fenômeno conhecido como salt and pepper) 
    
    Args: 
    
    path_image_raw  = Caminho relativo que contem a imagem original em escala de cinza, para realizar comparações
    path_image_noise = Caminho relativo que contem a imagem contendo ruido. 
    
    Return:
    
    3 mascaras que representam o processamento da imagem com ruido utilizando três kernels com tamanhos diferentes 
    (3,3), (5,5), (7,7).
    
    '''
    
    img_noise = cv2.imread(path_image_noise) #path/to/the/image/noise
    
    img_median_3 = cv2.medianBlur(img_noise, 3)
    img_median_5 = cv2.medianBlur(img_noise, 5)
    img_median_7 = cv2.medianBlur(img_noise, 7)

    cv2.imwrite(output_path+"/img_noise_reduction_3.png",img_median_3)
    cv2.imwrite(output_path+"/img_noise_reduction_5.png",img_median_5)
    cv2.imwrite(output_path+"/img_noise_reduction_7.png",img_median_7)
    
    return img_median_3, img_median_5, img_median_7
    
