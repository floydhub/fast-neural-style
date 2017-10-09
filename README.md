# fast-neural-style :city_sunrise: :rocket:
This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content of an image with the style of another image. For example, here is a photograph of a door arch rendered in the style of a stained glass painting. This is a porting of [pytorch/examples/fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style) making it usables on [FloydHub](https://www.floydhub.com/).

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). The saved-models for examples shown in the README can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0).

<p align="center">
    <img src="images/style-images/mosaic.jpg" height="200px">
    <img src="images/content-images/amber.jpg" height="200px">
    <img src="images/output-images/amber-mosaic.jpg" height="440px">
</p>

## Usage

`neural_style.py` options:

```bash
usage: neural_style.py [-h] {train,eval} ...

parser for fast-neural-style

optional arguments:
  -h, --help    show this help message and exit

subcommands:
  {train,eval}
    train       parser for training arguments
    eval        parser for evaluation/stylizing arguments
```

#### Train model

`neural_style.py train` options:

```bash
usage: neural_style.py train [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                             --dataset DATASET [--style-image STYLE_IMAGE]
                             --save-model-dir SAVE_MODEL_DIR
                             [--checkpoint-model-dir CHECKPOINT_MODEL_DIR]
                             [--image-size IMAGE_SIZE]
                             [--style-size STYLE_SIZE] --cuda CUDA
                             [--seed SEED] [--content-weight CONTENT_WEIGHT]
                             [--style-weight STYLE_WEIGHT] [--lr LR]
                             [--log-interval LOG_INTERVAL]
                             [--checkpoint-interval CHECKPOINT_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of training epochs, default is 2
  --batch-size BATCH_SIZE
                        batch size for training, default is 4
  --dataset DATASET     path to training dataset, the path should point to a
                        folder containing another folder with all the training
                        images
  --style-image STYLE_IMAGE
                        path to style-image
  --save-model-dir SAVE_MODEL_DIR
                        path to folder where trained model will be saved.
  --checkpoint-model-dir CHECKPOINT_MODEL_DIR
                        path to folder where checkpoints of trained models
                        will be saved
  --image-size IMAGE_SIZE
                        size of training images, default is 256 X 256
  --style-size STYLE_SIZE
                        size of style-image, default is the original size of
                        style image
  --cuda CUDA           set it to 1 for running on GPU, 0 for CPU
  --seed SEED           random seed for training
  --content-weight CONTENT_WEIGHT
                        weight for content-loss, default is 1e5
  --style-weight STYLE_WEIGHT
                        weight for style-loss, default is 1e10
  --lr LR               learning rate, default is 1e-3
  --log-interval LOG_INTERVAL
                        number of images after which the training loss is
                        logged, default is 500
  --checkpoint-interval CHECKPOINT_INTERVAL
                        number of batches after which a checkpoint of the
                        trained model will be created

```

Train template:
```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --cuda 1
```

There are several command line arguments, the important ones are listed below
* `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. I used COCO 2014 Training images dataset [80K/13GB] [(download)](http://mscoco.org/dataset/#download).
* `--style-image`: path to style-image.
* `--save-model-dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Refer to ``neural_style/neural_style.py`` for other command line arguments. For training new models you might have to tune the values of `--content-weight` and `--style-weight`. The mosaic style model shown above was trained with `--content-weight 1e5` and `--style-weight 1e10`. The remaining 3 models were also trained with similar order of weight parameters with slight variation in the `--style-weight` (`5e10` or `1e11`).

#### Stylize image

`neural_style.py eval` options:

```bash
usage: neural_style.py eval [-h] --content-image CONTENT_IMAGE
                            [--content-scale CONTENT_SCALE] --output-image
                            OUTPUT_IMAGE --model MODEL --cuda CUDA

optional arguments:
  -h, --help            show this help message and exit
  --content-image CONTENT_IMAGE
                        path to content image you want to stylize
  --content-scale CONTENT_SCALE
                        factor for scaling down the content image
  --output-image OUTPUT_IMAGE
                        path for saving the output image
  --model MODEL         saved model to be used for stylizing the image
  --cuda CUDA           set it to 1 for running on GPU, 0 for CPU

```

Eval template:
```
python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0
```
* `--content-image`: path to content image you want to stylize.
* `--model`: saved model to be used for stylizing the image (eg: `mosaic.pth`)
* `--output-image`: path for saving the output image.
* `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.


## Models

Models for the examples shown below can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0) or by running the script ``download_saved_models.sh``.

<div align='center'>
  <img src='images/content-images/amber.jpg' height="174px">
</div>

<div align='center'>
  <img src='images/style-images/mosaic.jpg' height="174px">
  <img src='images/output-images/amber-mosaic.jpg' height="174px">
  <img src='images/output-images/amber-candy.jpg' height="174px">
  <img src='images/style-images/candy.jpg' height="174px">
  <br>
  <img src='images/style-images/rain-princess-cropped.jpg' height="174px">
  <img src='images/output-images/amber-rain-princess.jpg' height="174px">
  <img src='images/output-images/amber-udnie.jpg' height="174px">
  <img src='images/style-images/udnie.jpg' height="174px">
</div>


## Architecture

## Run on FloydHub

Here's the commands to training, evaluating and serving your Fast Neural Transfer model on FloydHub.

### Project Setup

Before you start, log in on FloydHub with the [floyd login](http://docs.floydhub.com/commands/login/) command, then fork and init the project:

```bash
$ git clone https://github.com/ReDeiPirati/fast-neural-style.git
$ cd fast-neural-style
$ floyd init fast-neural-style
```


### Training


### Evaluating


### Try pre-trained model


### Serve model through REST API


## More resources


## Contributing

For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!