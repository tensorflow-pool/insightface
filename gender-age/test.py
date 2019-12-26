import argparse

import cv2
import face_model
import glob

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
# parser.add_argument('--image', default='Tom_Hanks_54745.png', help='')
parser.add_argument('--image', default='output/*', help='')
parser.add_argument('--model', default='model/model,0', help='path to load model.')
# parser.add_argument('--model', default='train/agesex_2019-12-26-14:52:14_back/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
args = parser.parse_args()

model = face_model.FaceModel(args)
files = glob.glob(args.image)
images = []
for file in files:
    print(file)
    img = cv2.imread(file)
    images.append(img)
gender, age = model.get_ga(images)
print('gender is', gender)
print('age is', age)
