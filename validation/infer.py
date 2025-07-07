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

    dataset = CustomDataset(df_test)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,  # num of subprocess-workers
        pin_memory=True,  # ускоряет работу на CUDA
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

    # print(model)
    model.eval()

    names_of_latent = [f"l{i}" for i in range(BOTTLENECK_DIM)]

    latent_list = []
    target_list = []

    with torch.no_grad():
        for i, (features, targets) in enumerate(dataloader):
            # features: (batch, features, seq_len) → (batch, seq_len, features) если нужно
            batch = features.to(device).permute(0, 2, 1)
            # forward
            latents = model(batch)  # tensor [batch, BOTTLENECK_DIM]
            latents = latents.cpu().numpy()  # numpy [batch, BOTTLENECK_DIM]
            targets = targets.numpy()  # numpy [batch,]

            latent_list.append(latents)
            target_list.append(targets)
            if i % 100 == 0:
                print(f"Processed batch {i}/{len(dataloader)}")

    # объединяем все батчи
    all_latents = np.vstack(latent_list)  # shape [N_total, BOTTLENECK_DIM]
    all_targets = np.concatenate(target_list)  # shape [N_total,]

    # строим DataFrame один раз
    df_latent = pd.DataFrame(all_latents, columns=names_of_latent)
    df_latent["target"] = all_targets
    df_latent.to_csv("full_latents.csv")
pass
