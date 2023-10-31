import math
import torch
import glob
import os
import datetime
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import autocast
from torchvision.transforms.functional import to_pil_image
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PokemonConvVAESmall import PokemonConvVAE
from PokemonDataset import PokemonDataset


def main():
    print("Num GPUs Available: ", torch.cuda.device_count())

    # HYPER-PARAMETERS AND OTHER CONSTANTS

    learning_rate = 3e-4
    batch_size = 1280
    val_batch_size = 1280
    val_every_n_epochs = 5
    max_epochs = 1000
    validation_prop = 0.01
    latent_size = 1024
    noise_std = 0.5
    imgs_path = "./img_preprocessed"
    imgs_per_validation = 3
    deconv = False

    training_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dir_name = (f"latent_size={latent_size}_batch_size={batch_size}_" 
                f"noise_std={noise_std}_timestamp={training_timestamp}")
    if deconv:
        dir_name += "_deconv"
    os.mkdir(f"./training_generated_imgs_no_quantum/{dir_name}")
    os.mkdir(f"./trained_models_no_quantum/{dir_name}")

    # DATA LOADING

    imgs_path += '/**/*.png'
    imgs_folder_path = imgs_path.replace("//", "/")
    imgs_paths = glob.glob(imgs_folder_path, recursive=True)
    imgs_train, imgs_val = train_test_split(imgs_paths, test_size=validation_prop)

    train_dataset = PokemonDataset(imgs_train)
    val_dataset = PokemonDataset(imgs_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    # HYPER-PARAMETERS BETA_* FOR VAE TRAINING

    beta_r = 5
    beta_max = 4
    beta_period = math.floor(len(train_dataloader) * beta_r)
    beta_delta = beta_max / beta_period

    # MODEL, OPTIMIZER, LOSS, ETC.

    model = PokemonConvVAE(latent_size, noise_std).cuda()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()
    loss_function = nn.L1Loss(reduction='sum')

    # TRAINING LOOP

    for epoch in range(0, max_epochs):
        train_running_loss = 0.0
        train_running_kl_loss = 0.0
        train_running_reconstruction_loss = 0.0
        train_batches = 0
        beta = 0

        model.train(True)
        for x in tqdm(train_dataloader):
            optimizer.zero_grad()

            y, mean, logvar = model(x.cuda())

            kl_loss = ((-0.5) * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar))) * beta / batch_size
            reconstruction_loss = loss_function(y, x.cuda()) / batch_size
            loss = kl_loss + reconstruction_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 30.0)
            optimizer.step()

            train_running_loss += loss.item()
            train_running_kl_loss += kl_loss.item()
            train_running_reconstruction_loss += reconstruction_loss.item()
            train_batches += 1

            if beta < beta_max:
                beta += beta_delta
                if beta >= beta_max:
                    beta = beta_max

        train_loss = train_running_loss / train_batches
        train_kl_loss = train_running_kl_loss / train_batches
        train_reconstruction_loss = train_running_reconstruction_loss / train_batches

        print(f"Epoch {epoch} - train loss: {train_loss} - train kl loss: {train_kl_loss} - "
              f"train reconstruction loss: {train_reconstruction_loss}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss KL/train", train_kl_loss, epoch)
        writer.add_scalar("Loss Reconstruction/train", train_reconstruction_loss, epoch)

        if epoch % val_every_n_epochs == 0:
            val_running_loss = 0.0
            val_running_kl_loss = 0.0
            val_running_reconstruction_loss = 0.0
            val_batches = 0

            model.eval()
            with torch.no_grad():
                for x in tqdm(val_dataloader):
                    y, mean, logvar = model(x.cuda())

                    kl_loss = ((-0.5) * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar))) * beta / batch_size
                    reconstruction_loss = loss_function(y, x.cuda()) / batch_size
                    loss = kl_loss + reconstruction_loss

                    val_running_loss += loss.item()
                    val_running_kl_loss += kl_loss.item()
                    val_running_reconstruction_loss += reconstruction_loss.item()
                    val_batches += 1

            val_loss = val_running_loss / val_batches
            val_kl_loss = val_running_kl_loss / val_batches
            val_reconstruction_loss = val_running_reconstruction_loss / val_batches

            print(f"val loss: {val_loss} - "
                  f"val kl loss: {val_kl_loss} - val reconstruction loss: {val_reconstruction_loss}")

            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Loss KL/validation", val_kl_loss, epoch)
            writer.add_scalar("Loss Reconstruction/validation", val_reconstruction_loss, epoch)

            with torch.no_grad():
                generated_imgs = model.generate(imgs_per_validation).numpy()
            generated_imgs = [generated_imgs[i, ...] for i in range(imgs_per_validation)]
            for i, im in enumerate(generated_imgs):
                plt.imshow(im, cmap='gray')
                plt.axis('off')
                plt.savefig(f"./training_generated_imgs_no_quantum/{dir_name}/epoch{epoch}_{i}.png",
                            bbox_inches='tight')
                plt.close()

        writer.flush()
        torch.save(model.state_dict(), f'./trained_models_no_quantum/{dir_name}/{epoch}.pt')

    writer.close()


if __name__ == '__main__':
    main()
