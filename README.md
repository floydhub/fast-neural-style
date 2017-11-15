# fast-neural-style :city_sunrise: :rocket:
This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content of an image with the style of another image. For example, here is a photograph of a door arch rendered in the style of a stained glass painting. This is a porting of [pytorch/examples/fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style) making it usables on [FloydHub](https://www.floydhub.com/).

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). The saved-models for examples shown in the README can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0).

**Requirement**: Run this project on FloydHub GPU instace. CPU instace does not satisfy the RAM requirement.

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


## Pretrained Models

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

*Soon*

## Run on FloydHub

Here's the commands to training, evaluating and serving your Fast Neural Transfer model on FloydHub.

### Project Setup

Before you start, log in on FloydHub with the [floyd login](http://docs.floydhub.com/commands/login/) command, then fork and init the project:

```bash
$ git clone https://github.com/floydhub/fast-neural-style.git
$ cd fast-neural-style
$ floyd init fast-neural-style
```

You can follow along the progress by using the [logs](http://docs.floydhub.com/commands/logs/) command.

### Training

After creating, uploading a dataset of images as FloydHub dataset for the training and put the style image in the `images/style-images/` folder, run:

```bash
floyd run --gpu --env pytorch-0.2 --data <YOUR_USERNAME>/dataset/<DATASET_NAME>/<VERSION>:input "python neural_style/neural_style.py train --dataset /input/<TRAIN_PATH> --style-image images/style-images/<STYLEIMAGE> --save-model-dir /output/<OUTPUT_NAME> --epochs 2 --cuda 1"
```

Note:

- `--gpu` run your job on a FloydHub GPU instance
- `--env pytorch-0.2` prepares a pytorch environment for python 3.
- `--data <YOUR_USERNAME>/dataset/<DATASET_NAME>/<VERSION>` mounts the pytorch mnist dataset in the `/input` folder inside the container

### Evaluating

Now it's time to evaluate your model on some images:

```bash
floyd run --gpu --env pytorch-0.2 --data <REPLACE_WITH_JOB_OUTPUT_NAME>:model "python neural_style/neural_style.py eval --content-image images/content-images/<YOUR_IMAGE>  --model /model/<CHOOSE_YOUR_MODEL>  --output-image /output/<OUTPUT_FILE_NAME> --cuda 1"
```


### Try pre-trained model

I've already uploaded for you the pretrained model provided by the Pytorch authors of fast-neural-style. Put the image you want to style in the `images/content-images/` path and run:

```bash
floyd run --gpu --env pytorch-0.2 --data redeipirati/datasets/fast-neural-style-models/1:model "python neural_style/neural_style.py eval --content-image images/content-images/<YOUR_IMAGE>  --model /model/<CHOOSE_YOUR_MODEL>  --output-image /output/<OUTPUT_FILE_NAME> --cuda 1"
```


### Serve model through REST API

FloydHub supports seving mode for demo and testing purpose. Before serving your model through REST API,
you need to create a `floyd_requirements.txt` and declare the flask requirement in it. If you run a job
with `--mode serve` flag, FloydHub will run the `app.py` file in your project
and attach it to a dynamic service endpoint:


```bash
floyd run --gpu --mode serve --env pytorch-0.2 --data <REPLACE_WITH_JOB_OUTPUT_NAME>:input
```

The above command will print out a service endpoint for this job in your terminal console.

The service endpoint will take a couple minutes to become ready. Once it's up, you can interact with the model by sending your image file with a POST request and the service will return stylized image:

```bash
# Template
curl -X POST -o <NAME_&_PATH_DOWNLOADED_IMG> -F "file=@<IMAGE_TO_STYLE>" -F "checkpoint=<MODEL_CHECKPOINT>" <SERVICE_ENDPOINT>

# e.g. of a POST req
curl -X POST -o myfile-udnie.jpg -F "file=@./myfile.jpg" -F "checkpoint=udnie.pth" https://www..floydlabs.com/expose/BhZCFAKom6Z8RptVKskHZW
```

Any job running in serving mode will stay up until it reaches maximum runtime. So
once you are done testing, **remember to shutdown the job!**

*Note that this feature is in preview mode and is not production ready yet*


## More resources

- [Convolutional neural networks for artistic style transfer](https://harishnarayanan.org/writing/artistic-style-transfer/)
- [Artistic Style Transfer with Deep Neural Networks](https://shafeentejani.github.io/2016-12-27/style-transfer/)
- [Draw me like one of your French girls: Neural artistic style-transfer explained](https://medium.com/@hhl60492/draw-me-like-one-of-your-french-girls-neural-artistic-style-transfer-explained-5996dfc8e26f)
- [Creating art with deep neural networks](https://blog.paperspace.com/art-with-neural-networks/)
- [Supercharging Style Transfer - GoogleBlog](https://research.googleblog.com/2016/10/supercharging-style-transfer.html)
- [How to Generate Art](https://youtu.be/Oex0eWoU7AQ)

## Contributing

For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!
