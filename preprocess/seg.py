# -*- coding: utf-8 -*-
# @Time   : 2022/3/28 15:55
# @Author : LWDZ
# @File   : seg.py
# @aim    : segment image
#--------------------------------------------------

import os
from io import BytesIO

import numpy as np
from PIL import Image

import tensorflow as tf
import sys
import datetime
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    graph_def = tf.compat.v1.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.compat.v1.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    start = datetime.datetime.now()

    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    end = datetime.datetime.now()

    diff = end - start

    return resized_image, seg_map

def drawSegment(baseImg, matImg):
  width, height = baseImg.size
  dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
  for x in range(width):
            for y in range(height):
                color = matImg[y,x]
                (r,g,b) = baseImg.getpixel((x,y))
                if color == 0:
                    dummyImg[y,x,3] = 0
                else :
                    dummyImg[y,x] = [r,g,b,255]
  img = Image.fromarray(dummyImg)
  img.save(outputFilePath)

def getSegment(baseImg, matImg):
  width, height = baseImg.size
  dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
  for x in range(width):
            for y in range(height):
                color = matImg[y,x]
                (r,g,b) = baseImg.getpixel((x,y))
                if color == 0:
                    dummyImg[y,x,3] = 0
                else :
                    dummyImg[y,x] = [r,g,b,255]
  img = Image.fromarray(dummyImg)
  return img

def run_visualization(filepath):
  """Inferences DeepLab model and visualizes result."""
  try:
  	#print("Trying to open : " + filepath)
  	# f = open(sys_temp[1])
  	jpeg_str = open(filepath, "rb").read()
  	orignal_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check file: ' + filepath)
    return

  #print('running deeplab on image %s...' % filepath)
  resized_im, seg_map = MODEL.run(orignal_im)

  # vis_segmentation(resized_im, seg_map)
  drawSegment(resized_im, seg_map)


def get_seg_img(filepath):
  """Inferences DeepLab model and visualizes result."""
  try:
  	jpeg_str = open(filepath, "rb").read()
  	orignal_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check file: ' + filepath)
    return
  resized_im, seg_map = MODEL.run(orignal_im)
  return getSegment(resized_im, seg_map)    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Segment image")
    # data
    parser.add_argument('--modelType',  default='xception_model',help='model name')
    parser.add_argument('--inputRootPath',default="../dataset/origin_data", help="input data path")
    parser.add_argument('--outputRootPath', default="../dataset/seg_origin_data", help="input data path")
    args = parser.parse_args()


    rootPath = sys.argv[0]
    inputRootPath = args.inputRootPath
    outputRootPath =args.outputRootPath

    files_list = os.listdir(inputRootPath)
    filter_files_list = [fn for fn in files_list if fn.endswith("jpg")]
    input_files_list = [os.path.join(inputRootPath, fn) for fn in files_list]
    temp_files_list = [os.path.join(outputRootPath, fn) for fn in files_list]
    output_files_list = [fn.replace(".jpg", ".png") for fn in temp_files_list]

    modelType = args.modelType # "xception_model"
    MODEL = DeepLabModel(modelType)
    print('model loaded successfully : ' + modelType)

    for i in range(len(input_files_list)):
        inputFilePath = input_files_list[i]
        outputFilePath = output_files_list[i]

        if i % 50 == 0:
            print("{}:now, {} of the job is done".format(i, i / len(input_files_list)))
        if inputFilePath is None or outputFilePath is None:
            print("Bad parameters. Please specify input file path and output file path")
            exit()
        if not os.path.exists(outputFilePath):
            run_visualization(inputFilePath)

