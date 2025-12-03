import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train import (
    BOTTLENECK_DIM,
    Autoencoder,
    CustomDataset,
    load_data,
)

if __name__ == "__main__":

    BATCH_SIZE = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"

    file_csv = "OscGrid_dataset/CSV_format_v1.1/labeled.csv"
    df_train, df_val, df_test = load_data(file_csv)

    dataset = CustomDataset(df_test, 10)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,  # number of subprocess workers
        pin_memory=True,  # speeds up work on CUDA
    )

    run_id = "d28b1f6a5d8944aea4f168cb113d43c1"

    model_path = f"model_epoch_10/model.pt"

    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(model_path, map_location=device))
    print(autoencoder.encoder_conv)

    model = nn.Sequential(
        autoencoder.encoder_conv,
        nn.Flatten(start_dim=1),
        autoencoder.encoder_fc,
    )

    model.eval()

    names_of_latent = [f"l{i}" for i in range(BOTTLENECK_DIM)]

    latent_list = []
    target_list = []

    with torch.no_grad():
        for i, (features, targets) in enumerate(dataloader):
            batch = features.to(device).permute(0, 2, 1)
            latents = model(batch)
            latents = latents.cpu().numpy()
            targets = targets.numpy()
            latent_list.append(latents)
            target_list.append(targets)
            if i % 100 == 0:
                print(f"Processed batch {i}/{len(dataloader)}")

    all_latents = np.vstack(latent_list)
    all_targets = np.concatenate(target_list)
    df_latent = pd.DataFrame(all_latents, columns=names_of_latent)
    df_latent["target"] = all_targets
    df_latent.to_csv("full_latents.csv")
