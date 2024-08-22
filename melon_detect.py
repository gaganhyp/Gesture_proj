import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib

data_dir = pathlib.Path()
cnt = len(list(data_dir.glob('**/*')))
