from torchvision.transforms.functional import to_grayscale, pil_to_tensor, to_pil_image
from subprocess import DEVNULL, STDOUT, check_call
import shlex
import glob
import torch
import math
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    image_temp_size = 50
    image_final_size = 40

    imgs_to_resize_paths = glob.glob("./img_raw/other" + '/**/*.png', recursive=True)
    for i, image_path in enumerate(tqdm(imgs_to_resize_paths)):
        with (Image.open(image_path) as im):
            if im.mode != 'RGB':
                im = im.convert('RGBA')
                white_background = Image.new("RGBA", im.size, "WHITE")
                white_background.paste(im, (0, 0), im)
                white_background = white_background.convert('RGB')
                white_background = white_background.resize((image_temp_size, image_temp_size),
                                                           resample=Image.Resampling.BICUBIC)
                color_background = Image.new("RGBA", im.size, "WHITE")
                color_background.paste(im, (0, 0), im)
                im = color_background
                im = im.convert('RGB')
            else:
                white_background = im.resize((image_temp_size, image_temp_size),
                                             resample=Image.Resampling.BICUBIC)

            white_background = to_grayscale(white_background, num_output_channels=1)
            white_background = pil_to_tensor(white_background)
            white_background = 255 - white_background
            white_background = white_background.squeeze()
            non_white = torch.argwhere(white_background)
            crop_lower = non_white[:, 0].max().item()
            crop_upper = non_white[:, 0].min().item()
            crop_left = non_white[:, 1].min().item()
            crop_right = non_white[:, 1].max().item()
            x_new_size = crop_right - crop_left
            y_new_size = crop_lower - crop_upper

            if x_new_size > y_new_size:
                padding = x_new_size - y_new_size
                down_padding = math.floor(padding / 2.0)
                up_padding = padding - down_padding
                crop_lower += down_padding
                if crop_lower >= image_temp_size:
                    crop_upper -= crop_lower - (image_temp_size - 1)
                    crop_lower = image_temp_size - 1
                crop_upper -= up_padding
                if crop_upper < 0:
                    crop_lower += -crop_upper
                    crop_upper = 0
            elif x_new_size != y_new_size:
                padding = y_new_size - x_new_size
                left_padding = math.floor(padding / 2.0)
                right_padding = padding - left_padding
                crop_right += right_padding
                if crop_right >= image_temp_size:
                    crop_left -= crop_right - (image_temp_size - 1)
                    crop_right = image_temp_size - 1
                crop_left -= left_padding
                if crop_left < 0:
                    crop_right += -crop_left
                    crop_left = 0

            if crop_left < 0 or crop_right >= image_temp_size or crop_upper < 0 or crop_lower >= image_temp_size:
                raise Exception("Incorrect crop bounding box!")

            im = im.resize((image_temp_size, image_temp_size), resample=Image.Resampling.BICUBIC)
            im = to_grayscale(im, num_output_channels=1)
            im = im.crop((crop_left, crop_upper, crop_right, crop_lower))

            if im.width != im.height:
                raise Exception("The resulting image is not square!")

            im = im.resize((image_final_size, image_final_size), resample=Image.Resampling.BICUBIC)

            im.save(f"./img_preprocessed/{i}.png")


if __name__ == '__main__':
    main()
