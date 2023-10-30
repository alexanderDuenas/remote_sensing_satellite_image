import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import albumentations as A

import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

from segmentation_models import Unet
from keras.layers import Input, Conv2D
from keras.models import Model

from keras import backend as K
import argparse
import glob
import cv2

parser = argparse.ArgumentParser(
        description="Script that execute building segmentation training using U-net"
    )
    
parser.add_argument(
        "--data_folder",
        type=str,
        help="Input folder containing all mask and raw RGB images",
        default='/path/to/data_folder'
    )

parser.add_argument(
        "--batch_size",
        type=int,
        help="Input folder containing all masks images",
        default= 64
    )
    
parser.add_argument(
        "--epochs",
        type=int,
        help="Input folder containing all masks images",
        default= 20
    )

parser.add_argument(
        "--path_model",
        type=str,
        help="Output folder for saving the training model",
        default= "path/model/destination"
    )

    
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.set_visible_devices(physical_devices[1],'GPU')


class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['building']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        #self.directory_mask = ""    
        # convert str names to class values on masks
        print(f"classes {classes}")
        self.class_values = [self.CLASSES.index(cls.lower()) + 1 for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def pad_image(self,img, aspect_width=320, aspect_height=320):
      img_width = img.shape[1]
      img_height = img.shape[0]

      original_w2h_ratio = float(img_width) / img_height
      target_w2h_ratio = float(aspect_width) / aspect_height
      is_skinny_image = original_w2h_ratio < target_w2h_ratio

      if is_skinny_image:
        dx = int((target_w2h_ratio * img_height) - img_width)
        pad = ((0, 0), (0, dx))
        if(img.ndim == 3):
          img = np.stack([np.pad(img[:,:,c], pad, mode='constant', constant_values=0) for c in range(3)], axis=2)
        else:
          img = np.pad(img[:,:], pad, mode='constant', constant_values=0)
          print(img.shape)
      else:
        dy = int((img_width / target_w2h_ratio) - img_height)
        pad = ((0, dy), (0, 0))
        if(img.ndim == 3):
          img = np.stack([np.pad(img[:,:,c], pad, mode='constant', constant_values=0) for c in range(3)], axis=2)
          print(img.shape)
        else:
          img = np.pad(img[:,:], pad, mode='constant', constant_values=0)

      return img

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320,320), interpolation = cv2.INTER_AREA)
        #print(self.images_fps[i])
        
        name_mask = self.masks_fps[i].split(".")
        name_mask = name_mask[0] + ".png"
        
#        print(name_mask)
        #print(self.masks_fps[i])
         
    
        mask = cv2.imread(name_mask, 0)
        mask = cv2.resize(mask, (320,320), interpolation = cv2.INTER_AREA)
        mask_aux = np.zeros_like(mask)
        mask_aux[mask != 0] = 1
        mask_aux = np.expand_dims(mask_aux, axis = 2).astype('float')
        mask = mask_aux


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
            



def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

def run(config):

    '''
    Método para realizar o treinamento do modélo utilizando a arquitetura U-net
    
    args: 

    path_folder = caminho que contem o folder com os dados de treinamento e validação
    batch_size = Tamanho do batch que será utilizado no treinamento e validação
    epochs = numero de épocas que seram utilizadas para realizar o treinamento 

    '''

    #list_of_masks_imgs = os.listdir(args.masks_loc)
    DATA_DIR = config.data_folder
    BATCH_SIZE = config.batch_size
    EPOCHS = config.epochs
    path_model = config.path_model  
    
    # path dir
    x_train_dir = os.path.join(DATA_DIR, 'train/')
    y_train_dir = os.path.join(DATA_DIR, 'train_mask/')
    x_valid_dir = os.path.join(DATA_DIR, 'validation/')
    y_valid_dir = os.path.join(DATA_DIR, 'validation_mask/')
    
    
    n_train_images = len(glob.glob(x_train_dir  + '*.jpg'))
    n_train_masks = len(glob.glob(y_train_dir  + '*.png'))
    n_val_images = len(glob.glob(x_valid_dir + '*.jpg'))
    n_val_masks = len(glob.glob(y_valid_dir + '*.png'))

    print('Number of train images =', n_train_images)
    print('Number of train masks =', n_train_masks)
    print('Number of val images =',  n_val_images)
    print('Number of val masks =',  n_val_masks)


    
    #BATCH_SIZE = 64 #64
    CLASSES = ['building']
    LR = 0.001
    #EPOCHS = 20

    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    
    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.Unet(BACKBONE, classes=n_classes, encoder_weights='imagenet',activation=activation)
    
    
    # define optomizer
    optim = keras.optimizers.Adam(LR)
    jaccar_loss= sm.losses.JaccardLoss()
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    BiorCat_loss = sm.losses.BinaryCELoss() if n_classes == 1 else sm.losses.CategoricalCELoss()
    total_loss = jaccar_loss # + (1 * BiorCat_loss)
    
    iou_score = sm.metrics.IOUScore(threshold=0.5)
    f1_score = sm.metrics.FScore(threshold=0.5)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    model.compile(optim, total_loss, metrics=metrics)
    
    print(model.summary())
    
    train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=CLASSES,
    #augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
    )
    
    # Dataset for validation images
    valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=CLASSES,
    #augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
    )
    
    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)
    
    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)
    
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(path_model + "best_model" ,
                                                monitor='val_iou_score',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='max',
                                                save_freq = 'epoch')
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor = tf.math.exp(-0.1),
    patience = 10,
    verbose=1,
    mode="auto",
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=5,mode='auto',
                                              #min_delta=0.001
                                              )
    
    # Log tensor board
    log_dir = path_model + 'logs/'
    callback_tensorbord = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)
    
    callbacks = [
    early_stop,
    checkpoint,
    reduce_lr,
    callback_tensorbord,
    ]
    
    # train model

    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
        )
    
    print("end training")

if __name__ == '__main__':
    
    config = parser.parse_args()
    run(config)
    

