import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml',encoding='utf-8') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]

def process_video(pairs, predictor, output_dir):
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0]+'_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        tqdm.write(f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)

# def main(img_pattern: str,
#          mask_pattern: Optional[str] = None,
#          weights_path='fpn_inception.h5',
#          out_dir='submit/',
#          side_by_side: bool = False,
#          video: bool = False):
#     def sorted_glob(pattern):
#         return sorted(glob(pattern))

#     imgs = sorted_glob(img_pattern)
#     masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
#     pairs = zip(imgs, masks)
#     names = sorted([os.path.basename(x) for x in glob(img_pattern)])
#     predictor = Predictor(weights_path=weights_path)

#     os.makedirs(out_dir, exist_ok=True)
#     if not video:
#         for name, pair in tqdm(zip(names, pairs), total=len(names)):
#             f_img, f_mask = pair
#             img, mask = map(cv2.imread, (f_img, f_mask))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#             pred = predictor(img, mask)
#             if side_by_side:
#                 pred = np.hstack((img, pred))
#             pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
#             cv2.imwrite(os.path.join(out_dir, name),
#                         pred)
#     else:
#         process_video(pairs, predictor, out_dir)


def pad_image_to_multiple(img, min_size=256, divisor=32):
    """Pads image symmetrically using reflection to reach at least min_size and multiple of divisor."""
    h, w = img.shape[:2]

    # Ensure minimum size
    h_new = max(min_size, ((h + divisor - 1) // divisor) * divisor)
    w_new = max(min_size, ((w + divisor - 1) // divisor) * divisor)

    top = (h_new - h) // 2
    bottom = h_new - h - top
    left = (w_new - w) // 2
    right = w_new - w - left

    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    return padded, (h, w), (top, left)


def check_for_device():
    print("Is CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")



def main(img_root: str,
         mask_pattern: Optional[str] = None,
         weights_path='fpn_inception.h5',
         out_dir='submit/',
         side_by_side: bool = False,
         video: bool = False):

    check_for_device()

    predictor = Predictor(weights_path=weights_path)

    # Recursively find all .jpg files
    img_paths = sorted(glob(os.path.join(img_root, '**', '*.jpg'), recursive=True))

    for f_img in tqdm(img_paths, desc="Processing"):
        rel_path = os.path.relpath(f_img, img_root)  # Get relative path for output folder
        out_path = os.path.join(out_dir, rel_path)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = cv2.imread(f_img)
        if img is None:
            print(f"[WARNING] Failed to read image: {f_img}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_padded, original_size, pad_offsets = pad_image_to_multiple(img)

        pred = predictor(img_padded, None)

        # Crop to original size
        top, left = pad_offsets
        h, w = original_size
        pred = pred[top:top + h, left:left + w]

        if side_by_side:
            img_vis = img_padded[top:top + h, left:left + w]
            pred = np.hstack((img_vis, pred))

        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, pred)

# def getfiles():
#     filenames = os.listdir(r'.\dataset1\blur')
#     print(filenames)
def get_files():
    list=[]
    for filepath,dirnames,filenames in os.walk(r'.\dataset1\blur'):
        for filename in filenames:
            list.append(os.path.join(filepath,filename))
    return list





if __name__ == '__main__':
   Fire(main)
#增加批量处理图片：
    # img_path=get_files()
    # for i in img_path:
    #     main(i)
    # main('test_img/tt.mp4')
