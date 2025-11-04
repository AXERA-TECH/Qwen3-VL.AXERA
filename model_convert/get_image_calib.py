# from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from preprocess import Qwen2VLImageProcessorExport
from transformers.image_utils import PILImageResampling
from glob import glob 
import numpy as np
import random
from PIL import Image
import cv2
import os 

if __name__=="__main__":
    paths = sorted(glob("./video-test-04/*"))
    print(paths)

    os.makedirs("calib_img_640")
    for i,p in enumerate(paths):
        img = Image.open(p).resize((640,640))
        
        img_processor = Qwen2VLImageProcessorExport(max_pixels=640*640, patch_size=16, temporal_patch_size=2, merge_size=2)
        image_mean = [
            0.5,
            0.5,
            0.5
        ]

        image_std =  [
            0.5,
            0.5,
            0.5
        ]
        # pixel_values, grid_thw = img_processor._preprocess(images, do_resize=True, resample=PILImageResampling.BICUBIC, 
        #                                     do_rescale=True, rescale_factor=1/255, do_normalize=True, 
        #                                     image_mean=image_mean, image_std=image_std,do_convert_rgb=True)
        pixel_values, grid_thw = img_processor._preprocess([img], do_resize=True, resample=PILImageResampling.BICUBIC, 
                                            do_rescale=False, do_normalize=False, 
                                            do_convert_rgb=True)
        print("pixel_values_videos", pixel_values.shape)


       
        cv2.imwrite(f"calib_img_640/h{i}.jpg", pixel_values[0].astype(np.uint8))