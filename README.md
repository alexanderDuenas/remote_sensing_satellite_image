# remote_sensing_satellite_images

Este repositório tem como principal objetivo mostrar uma série de técnicas de visão computacional aplicadas ao sensoreamento remoto em imagens satelitais para realizar tarefas como remoção de ruido em imagens e segmentação de pixels em diferentes cenários. Para isto foram utilizadas uma série bibliotecas tais como rasterio, gdal e earthpy a fim de realizar o tratamento e visualização dos dados satelitais.  

Para o processamento das imagens foram utilizadas técnicas clássicas de visão computacional como transformação de espaço de cores, pre e pós processamento de imagens e transformações morfológicas, além da utilização da arquitetura U-net baseada em deep learning para treinar um modelo que vise realizar a segmentação de residências em um conjunto de imagens satelitais.

## Installation

### Ambiente virtual
Para a instalação do repositório e as dependências recomenda-se criar um ambiente virtual. 

```
python3 -m venv sat_imgs_vc
source sat_imgs_vc/bin/activate
```
Como alternativa pode ser instalado um ambiente virtual conda 

```
conda create --name sat_imgs_vc
conda activate sat_imgs_vc
```
### Requirements
Uma vez ativado o ambiente virtual pode ser instaladas as diferentes dependências:  

  - Python >= 3.6
  - keras >= 2.2.0 or tensorflow >= 1.13
  - CUDA > 9.0
  - segmenation-models==1.0
  - albumentations==0.3.0 

A arquitetura de deep Learning  utilizada para realizar a tarefa de segmentação de pixels foi a [U-net](https://arxiv.org/abs/1505.04597). A implementação e treinamento da arquitetura foi realizada utilizando
o framework Tensorflow junto com Keras. 

```
pip install -U -q segmentation-models
pip install -q tensorflow==2.2.1
pip install -q keras==2.5
```
Sobre instalar as bibliotecas de CUDA and CudNN para realizar treinamentos com GPU pode observar o seguinte [link](https://santhoshpkumar.github.io/Cuda-Install-and-Setup/)
 

Instalar outros requerimentos (OpenCV, Rasterio, gdal, etc), para realizar tarefas de tratamento das imagens satelitais, assim como processamento das imagens utilizadas: 

```
python3 -m pip install -r requirements.txt
```
## Tratamento de imagens
Nesta seção amostra-se o procedimento para responder cada uma das perguntas elaboradas no questionario





