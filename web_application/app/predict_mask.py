# from flask import Flask, request, jsonify
import numpy as np

import cv2
import torch
from PIL import Image as im
import matplotlib.pyplot as plt
from fastai import *
from fastai.vision import *
import posixpath
import base64
from stairpose import process_mask
import io
import time

path_images = 'train'
path_valid = 'val'
fnames = get_image_files(path_images)

class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)

class SegItemListCustom(SegmentationItemList):
    _label_cls = SegLabelListCustom

def get_y_fn(x): return (posixpath.join('train_masks', f'{x.stem}{x.suffix}')) if (
    x.parent.stem == 'train') else (posixpath.join('val_masks', f'{x.stem}{x.suffix}'))

colors = np.loadtxt('stair.txt', delimiter='\n', dtype=str)
src = (SegItemListCustom.from_folder(Path(''))
       .split_by_folder(train='train', valid='val')
       .label_from_func(get_y_fn, classes=colors)
       # tfms_y=True because transforms we r applying on trainset,will be also applied on train_masks
       .transform(get_transforms(), tfm_y=True, size=(240, 320))
       #  .add_test_folder(test_folder='test',tfms=None,tfm_y=False)#since test_masks are empty we dont need tfms on ground truth here so tfm_y=False
       )

data = (src
        .databunch(bs=1)
        .normalize(imagenet_stats))

learn = unet_learner(data, models.resnet34)

def apply_resnet(arr, save=True):
    test_image = Image(pil2tensor(arr, dtype=np.float32).div_(255))
    img_segment = learn.predict(test_image)[0]
    # print(img_segment)
    test_image_data = test_image.data.permute(1, 2, 0)
    img_segment_data = img_segment.data*255
    img_segment_data = img_segment_data.permute(1, 2, 0)
    mask = img_segment.data.permute(1, 2, 0).numpy()
    mask = (np.squeeze(mask, axis=2))*255
    # plt.imshow(mask)
    # plt.show()
    return mask


model = torch.hub.load('ultralytics/yolov5',  'custom', path='models/best3.pt',
                       force_reload=True)  # or yolov5m, yolov5l, yolov5x, custom


def apply_yolov5(bbox_array, img):
    res = model(img)
    try:
        l = res.xyxy[0]
        xmin = int(l[0, 0])
        ymin = int(l[0, 1])
        xmax = int(l[0, 2])
        ymax = int(l[0, 3])
        return cv2.rectangle(bbox_array, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2), xmin, xmax, ymin, ymax
    except:
        return bbox_array, 0, 320, 0, 240


learn.load('w1')


def evaluate(img):
    # img_orig = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    uploaded_image_shape = img.shape
    print(uploaded_image_shape)
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
    bbox_array = np.zeros([240, 320, 4], dtype=np.uint8)
    bbox_array, xmin, xmax, ymin, ymax = apply_yolov5(bbox_array, img)
    # bbox_array, xmin, xmax, ymin, ymax = ([], 0, 239, 0, 319)
    m_orig = img.shape[0]
    n_orig = img.shape[1]
    img = img[ymin:ymax, xmin:xmax, :]
    m = img.shape[0]
    n = img.shape[1]
    img = cv2.copyMakeBorder(img, (m_orig-m)//2, (m_orig-m+1)//2,
                             (n_orig-n)//2, (n_orig-n+1)//2, cv2.BORDER_CONSTANT, (0, 0, 0))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = apply_resnet(gray)
    start = time.time()
    pose, stair_mask = process_mask(
        mask, ymin, ymax, xmin, xmax, m_orig, n_orig, m, n, uploaded_image_shape)
    end = time.time()
    print("time=", end-start)

    return stair_mask, pose


# app = Flask(__name__)


# @app.route("/", methods=["GET", "POST"])
# def home():
#     img_string = request.json["img_string"]
#     mask, pose, plot_string1, plot_string2 = evaluate(img_string)

#     retval, buffer = cv2.imencode('.png', mask)
#     mask_string = base64.b64encode(buffer)
#     return jsonify({
#         "mask_string": mask_string.decode("utf-8"),
#         "pose": pose,
#         "plot_string1": plot_string1.decode("utf-8"),
#         "plot_string2": plot_string2.decode("utf-8")
#     })


# @app.route("/test", methods=["GET", "POST"])
# def test():
#     img_string = request.json["img_string"]
#     print(img_string)
#     return f"api is working"


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
