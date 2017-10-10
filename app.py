"""
Flask Serving

This file is a sample flask app that can be used to test your model with an API.

This app does the following:
	- Handles uploads and looks for an image file send as "file" parameter
	- Stores the image at ./images dir
	- Invokes ffwd_to_img function from evaluate.py with this image
	- Returns the output file generated at /output

Additional configuration:
	- You can also choose the checkpoint file name to use as a request parameter
	- Parameter name: checkpoint
	- It is loaded from /input
"""

import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from neural_style import utils
from neural_style.transformer_net import TransformerNet
from neural_style.vgg import Vgg16
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
MODEL_PATH = "/input"
CUDA = torch.cuda.is_available()



# Image preprocessing, Loading model, Style and return output
def stylize(input_filepath, output_filepath, checkpoint, content_scale=None):
	content_image = utils.load_image(input_filepath, scale=content_scale)
	content_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.mul(255))
	])
	content_image = content_transform(content_image)
	content_image = content_image.unsqueeze(0)
	if CUDA:
		content_image = content_image.cuda()
	content_image = Variable(content_image, volatile=True)

	style_model = TransformerNet()
	style_model.load_state_dict(torch.load(checkpoint))
	if CUDA:
		style_model.cuda()
	output = style_model(content_image)
	if CUDA:
		output = output.cpu()
	output_data = output.data[0]
	utils.save_image(output_filepath, output_data)


app = Flask(__name__)

@app.route('/<path:path>', methods=["POST"])
def style_transfer(path):
	"""
	Take the input image and style transfer it
	"""
	# check if the post request has the file part
	if 'file' not in request.files:
		return BadRequest("File not present in request")
	file = request.files['file']
	if file.filename == '':
		return BadRequest("File name is not present in request")
	if not allowed_file(file.filename):
		return BadRequest("Invalid file type")
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		input_filepath = os.path.join('./images/content-images/', filename)
		output_filepath = os.path.join('/output/', filename)
		file.save(input_filepath)

		# Get checkpoint filename
		checkpoint = os.path.join(MODEL_PATH, request.form.get("checkpoint")) # or one of the pretrained models
		try:
			os.path.isfile(checkpoint)
		except IOError as e:
			print(e)
			sys.exit(1)

		stylize(input_filepath, output_filepath, checkpoint)
		return send_file(output_filepath, mimetype='image/jpg')


def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
	app.run(host='0.0.0.0')