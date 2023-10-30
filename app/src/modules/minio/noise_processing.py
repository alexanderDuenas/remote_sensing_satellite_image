import cv2
import matplotlib.pyplot as plt
import numpy as np

def salt_pepper_process(path_image_raw, path_image_noise):
    
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
    img_original = cv2.imread(path_image_raw) #path/to/the/image_original
    
    img_median_3 = cv2.medianBlur(img_noise, 3)
    img_median_5 = cv2.medianBlur(img_noise, 5)
    img_median_7 = cv2.medianBlur(img_noise, 7)
    
    return img_median_3, img_median_5, img_median_7
    
