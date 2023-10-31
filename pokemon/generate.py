import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PokemonConvVAESmall import PokemonConvVAE
from PokemonConvQVAESmall import PokemonConvVAE as PokemonConvQVAE
from wand.image import Image
from PIL import Image as PILImage
from PIL import ImageOps


def create_collage(out_path, per_row):
    postfixes = [""]
    for postfix in postfixes:
        raw_images = []
        for img_file_name in os.listdir(f"{out_path}/raw"):
            img = Image(filename=f"{out_path}/raw/{img_file_name}")
            img.adaptive_resize(40, 40)
            img.border("Black", 1, 1)
            raw_images.append(img)
        for row_img_num in range(0, len(raw_images), per_row):
            img_row = raw_images[row_img_num]
            for other_image_num in range(1, per_row):
                img_row.sequence.append(raw_images[row_img_num + other_image_num])
            img_row.smush(False, 0)
            if row_img_num != 0:
                raw_images[0].sequence.append(img_row)
        raw_images[0].smush(True, 0)
        raw_images[0].border("Black", 1, 1)
        raw_images[0].save(filename=f"{out_path}/collage{postfix}.png")


def generate(model_path, out_path, model_class, z):
    imgs_to_generate = 400
    latent_size = 1024
    noise_std = 0.5

    model = model_class(latent_size, noise_std).cuda()
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        generated_imgs = model.generate_from_z(z).numpy()
    generated_imgs = [generated_imgs[i, ...] for i in range(imgs_to_generate)]
    for i, im in enumerate(generated_imgs):
        im = (im*255.9999).astype(np.uint8)
        img = PILImage.fromarray(im)
        img.save(f"./{out_path}/raw/{i}.png")

    create_collage(out_path, 16)


def main():
    imgs_to_generate = 400
    latent_size = 1024
    z = torch.randn(size=(imgs_to_generate, latent_size))

    out_dir_prefix = "generated_imgs/"
    for model_file_name, m_class in zip(["q.pt", "nq.pt"], [PokemonConvQVAE, PokemonConvVAE]):
        m_path = f"models_for_generation/{model_file_name}"
        o_path = f"{out_dir_prefix}{model_file_name.replace('.pt', '')}"
        if not os.path.exists(o_path + "/raw"):
            os.makedirs(o_path + "/raw")
        generate(m_path, o_path, m_class, z)


if __name__ == '__main__':
    main()
