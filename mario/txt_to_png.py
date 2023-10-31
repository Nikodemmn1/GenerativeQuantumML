import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def char_array_to_image(array, chars2pngs):
    """
    Convert a 16-by-16 array of integers into a PIL.Image object
    param: array: a 16-by-16 array of integers
    """
    image = Image.new('RGB', (array.shape[1] * 16, array.shape[0] * 16))
    for row, seg in enumerate(array):
        for col, char in enumerate(seg):
            image.paste(chars2pngs[char], (col * 16, row * 16))
    return image


def txt_levels_to_img(txt_levels, out_name):
    tiles_dir = 'tiles'
    chars2pngs = {
        "-": Image.open(f'{tiles_dir}/smb-background.png'),
        "X": Image.open(f'{tiles_dir}/smb-unpassable.png'),
        "S": Image.open(f'{tiles_dir}/smb-breakable.png'),
        "?": Image.open(f'{tiles_dir}/smb-question.png'),
        "Q": Image.open(f'{tiles_dir}/smb-question.png'),
        "o": Image.open(f'{tiles_dir}/smb-coin.png'),
        "E": Image.open(f'{tiles_dir}/smb-enemy.png'),
        "<": Image.open(f'{tiles_dir}/smb-tube-top-left.png'),
        ">": Image.open(f'{tiles_dir}/smb-tube-top-right.png'),
        "[": Image.open(f'{tiles_dir}/smb-tube-lower-left.png'),
        "]": Image.open(f'{tiles_dir}/smb-tube-lower-right.png'),
        "x": Image.open(f'{tiles_dir}/smb-path.png'),  # self-created
        "Y": Image.open(f'{tiles_dir}/Y.png'),  # self-created
        "N": Image.open(f'{tiles_dir}/N.png'),  # self-created
        "B": Image.open(f'{tiles_dir}/cannon_top.png'),
        "b": Image.open(f'{tiles_dir}/cannon_bottom.png'),
    }

    if not os.path.exists(f'generated_levels/{out_name}/'):
        os.makedirs(f'generated_levels/{out_name}/')

    for i, txt_level in enumerate(txt_levels):
        txt_level_parsed = np.array([list(line.rstrip()) for line in txt_level.splitlines(True)])
        plt.figure(figsize=(20, 4))
        plt.imshow(char_array_to_image(txt_level_parsed, chars2pngs))
        plt.axis('off')
        plt.savefig(f'generated_levels/{out_name}/{out_name}_{i}.png', dpi=200, bbox_inches='tight')
        plt.close()
