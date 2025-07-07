import os
import random
import shutil
import sys
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.fft as fft
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

FRAME_SIZE = 32
BOTTLENECK_DIM = 40
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 5e-04
MAX_GRAD_NORM = 10
SEED = 42
# dataset columns used for model training:
SUBSET = ["IA", "IC", "UA BB", "UB BB", "UC BB"]
# new names for selected columns
FEATURE_NAMES = ["IA", "IC", "UA", "UB", "UC"]
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Names of problematic files with duplicate data
# Exclude them from the training dataset:
remove_files = []

class CustomDataset(Dataset):

    def __init__(self, dt: pd.DataFrame):
        self.data = dt
        self.frame_size = FRAME_SIZE
        self.indexes = [
            idx
            for _, group in dt.groupby("block_id")
            for idx in group.index[: 1 - FRAME_SIZE]  # Skip last FRAME_SIZE-1 samples per block
        ]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        start = self.indexes[idx]
        frame = self.data.loc[start : start + self.frame_size - 1]
        sample = frame[["IA", "IC", "UA", "UB", "UC"]]
        x = torch.tensor(
            sample.to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        target = self.data.iloc[start].target
        target = torch.tensor(target, dtype=torch.long)
        return x, target


class SpectralLoss(nn.Module):
    def __init__(self, weights=None):
        super(SpectralLoss, self).__init__()
        self.weights = weights

    def forward(self, input_signal, target_signal, targets=None):
        input_spectrum = torch.abs(fft.rfft(input_signal))
        target_spectrum = torch.abs(fft.rfft(target_signal))

        if self.weights is not None and targets is not None:
            losses = torch.mean(
                torch.abs(input_spectrum - target_spectrum), axis=[1, 2]
            )
            weights = torch.empty(targets.shape, device=targets.device)
            for i, target in enumerate(targets):
                weights[i] = self.weights[target.item()]

            loss = torch.mean(losses * weights)
        else:
            loss = torch.mean(torch.abs(input_spectrum - target_spectrum))

        return loss


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        ch1 = BOTTLENECK_DIM // 2  # 20 when BOTTLENECK_DIM=40
        ch2 = BOTTLENECK_DIM  # 40
        fc_dim = BOTTLENECK_DIM * 12  # 480 when BOTTLENECK_DIM=40
        input_channels = 5
        input_length = 32

        # === Encoder ===
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(
                input_channels,
                ch1,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 32 -> 16
            nn.Conv1d(
                ch1,
                ch2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 16 -> 8
        )

        # After conv: [B, ch2, 8] → flattened: ch2 * 8
        self.encoder_fc = nn.Sequential(
            nn.Linear(ch2 * 8, fc_dim),
            nn.ReLU(True),
            nn.Linear(fc_dim, BOTTLENECK_DIM),  # bottleneck
        )

        # === Decoder ===
        self.decoder_fc = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, fc_dim),
            nn.ReLU(True),
            nn.Linear(fc_dim, ch2 * 8), 
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(
                ch2, ch1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 8→16
            nn.ReLU(True),
            nn.ConvTranspose1d(
                ch1,
                input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # 16→32
            # For precise length restoration (ConvTranspose1d can sometimes add +1)
            nn.Linear(input_length, input_length),
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder_fc(x)
        x = self.decoder_fc(x)
        x = x.view(x.size(0), BOTTLENECK_DIM, 8)
        x = self.decoder_conv(x)
        return x


def seed_everything(seed: int = 42):
    """
    This function is used to maintain repeatability
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(file_csv):
    # Load data from CSV file
    df = pd.read_csv(file_csv)

    df_cleaned = df.dropna(subset=SUBSET).copy()

    ml1_cols = [col for col in df_cleaned.columns if col.startswith("ML_1")]
    ml2_cols = [col for col in df_cleaned.columns if col.startswith("ML_2")]
    ml3_cols = [col for col in df_cleaned.columns if col.startswith("ML_3")]

    df_cleaned["ML1"] = df_cleaned[ml1_cols].eq(1).any(axis=1).astype(int)
    df_cleaned["ML2"] = df_cleaned[ml2_cols].eq(1).any(axis=1).astype(int)
    df_cleaned["ML3"] = df_cleaned[ml3_cols].eq(1).any(axis=1).astype(int)
    df_cleaned["ML0"] = (
        df_cleaned[["ML1", "ML2", "ML3"]].eq(0).all(axis=1).astype(int)
    )

    needed = [
        "sample",
        "file_name",
        "ML0",
        "ML1",
        "ML2",
        "ML3",
    ] + SUBSET

    df_final = df_cleaned[needed].copy()

    # Conditions for each class
    cond0 = (df_final["ML0"] == 1) & (
        df_final[["ML1", "ML2", "ML3"]].eq(0).all(axis=1)
    )
    cond1 = (df_final["ML1"] == 1) & (
        df_final[["ML2", "ML3"]].eq(0).all(axis=1)
    )
    cond2 = (df_final["ML2"] == 1) & (df_final["ML3"] == 0)
    cond3 = df_final["ML3"] == 1

    conds = [cond0, cond1, cond2, cond3]
    choices = [0, 1, 2, 3]

    df_final["target"] = np.select(conds, choices, default=np.nan).astype(int)
    df_final.drop(
        columns=[
            "ML0",
            "ML1",
            "ML2",
            "ML3",
        ],
        inplace=True,
    )

    df_final.rename(
        columns=dict(zip(SUBSET, FEATURE_NAMES)),
        inplace=True,
    )

    # Create boolean mask: True where file_name is not in remove_files
    mask = ~df_final["file_name"].isin(remove_files)

    # Apply mask to DataFrame
    df_final = df_final.loc[mask].copy()

    # Split into blocks, each with only one combination of file_name and target
    # 1) Mark start of new block when file_name or target changes
    df_final["new_block"] = (
        df_final["file_name"] != df_final["file_name"].shift()
    ) | (df_final["target"] != df_final["target"].shift())

    # 2) Assign unique identifier to each block
    df_final["block_id"] = df_final["new_block"].cumsum()

    # 3) Group by block_id and count length of each block
    blocks = df_final.groupby("block_id").agg(
        file_name=("file_name", "first"),
        target=("target", "first"),
        length=("block_id", "size"),
    )

    # 4) Keep only blocks with length >= FRAME_SIZE
    blocks_filtered = blocks[blocks["length"] >= FRAME_SIZE]

    # Shuffle
    blocks_filtered = blocks_filtered.sample(frac=1, random_state=SEED)

    # Dictionaries to accumulate block_ids by dataset
    split_blocks = {"train": [], "val": [], "test": []}

    # For each class separately
    for cls, grp in blocks_filtered.groupby("target"):
        total_len = grp["length"].sum()
        # How many measurements should go to each dataset for this class
        thresholds = {name: total_len * frac for name, frac in RATIOS.items()}
        # Accumulated length counters
        acc = {"train": 0, "val": 0, "test": 0}
        # Iterate through blocks
        for block_id, row in grp.iterrows():
            # Decide which dataset to put this block in
            # Choose first dataset that hasn't reached its threshold
            for name in ["val", "test", "train"]:
                if (
                    acc[name] + row["length"] <= thresholds[name]
                    or name
                    == "train"  
                ):
                    split_blocks[name].append(block_id)
                    acc[name] += row["length"]
                    break

    # Now we have block_id lists for each dataset
    train_ids = set(split_blocks["train"])
    val_ids = set(split_blocks["val"])
    test_ids = set(split_blocks["test"])

    # Mark rows in df_final by block_id
    df_final["split"] = df_final["block_id"].map(
        lambda b: (
            "train"
            if b in train_ids
            else "val" if b in val_ids else "test" if b in test_ids else "del"
            # if not assigned to any dataset - mark as 'del'
        )
    )

    # Check the final proportions by class and by the number of measurements
    result = (
        df_final.groupby(["split", "target"])["sample"]
        .count()
        .rename("n_measurements")
        .reset_index()
    )

    # Add percentage of each class within split
    result["percent"] = (
        result.groupby("split")["n_measurements"].transform(
            lambda x: 100 * x / x.sum()
        )
    ).round(2)

    # Check how data is distributed
    print(result)

    #     split  target  n_measurements  percent
    # 0     del       0              82     3.91
    # 1     del       1            1607    76.63
    # 2     del       2              56     2.67
    # 3     del       3             352    16.79

    # 4    test       0           74354    54.32
    # 5    test       1            7524     5.50
    # 6    test       2           44612    32.59
    # 7    test       3           10399     7.60

    # 8   train       0          347159    47.47
    # 9   train       1           35147     4.81
    # 10  train       2          208341    28.49
    # 11  train       3          140601    19.23

    # 12    val       0           74342    48.93
    # 13    val       1            7501     4.94
    # 14    val       2           44605    29.36
    # 15    val       3           25480    16.77

    cols_to_keep = ["block_id", "target"] + FEATURE_NAMES

    df_train = (
        df_final[df_final["split"] == "train"][cols_to_keep]
        .copy()
        .reset_index(drop=True)
    )
    df_val = (
        df_final[df_final["split"] == "val"][cols_to_keep]
        .copy()
        .reset_index(drop=True)
    )
    df_test = (
        df_final[df_final["split"] == "test"][cols_to_keep]
        .copy()
        .reset_index(drop=True)
    )

    # 1. Fit scaler only on training data
    scaler = StandardScaler()
    df_train_scaled_values = scaler.fit_transform(df_train[FEATURE_NAMES])

    # 2. Apply it to validation and test data
    df_val_scaled_values = scaler.transform(df_val[FEATURE_NAMES])
    df_test_scaled_values = scaler.transform(df_test[FEATURE_NAMES])

    # 3. Create new DataFrames with same index and column names
    df_train_scaled = pd.DataFrame(
        df_train_scaled_values, columns=FEATURE_NAMES, index=df_train.index
    )
    df_val_scaled = pd.DataFrame(
        df_val_scaled_values, columns=FEATURE_NAMES, index=df_val.index
    )
    df_test_scaled = pd.DataFrame(
        df_test_scaled_values, columns=FEATURE_NAMES, index=df_test.index
    )

    # 4. Replace original columns with normalized ones
    df_train.update(df_train_scaled)
    df_val.update(df_val_scaled)
    df_test.update(df_test_scaled)

    # scaler.mean_
    # array([ 0.00234656,  0.00602454,  0.02750021,  0.02089018, -0.01413002])
    # scaler.scale_
    # array([ 1.20476664,  1.19734593, 71.55793895, 74.9240093 , 75.38762804])
    return df_train, df_val, df_test


def interim_plots(inputs, outputs, epoch, step):
    fig, axes = plt.subplots(
        nrows=BATCH_SIZE,
        ncols=len(FEATURE_NAMES),
        figsize=(15, 3 * BATCH_SIZE),
        squeeze=False,
    )

    for idx in range(BATCH_SIZE):
        for j in range(len(FEATURE_NAMES)):
            ax = axes[idx][j]
            ax.plot(inputs[idx, j, :], label="orig")
            ax.plot(
                outputs[idx, j, :],
                linestyle="--",
                label="recon",
            )
            ax.set_ylim(-2, 2)
            ax.set_title(f"{FEATURE_NAMES[j]}_{idx+1}")
            if j == 0:
                ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(f"reconstruction_epoch_{epoch+1}_step_{step}.png")
    plt.close(fig)


def validation(model, data_loader, epoch):
    val_loss = 0.0
    n_val_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            targets = batch[1].squeeze().to(device)
            batch = batch[0].to(device).permute(0, 2, 1)
            output = model(batch)
            loss = criterion(output, batch, targets)
            val_loss += loss.item()
            n_val_batches += 1

    avg_val_loss = val_loss / max(1, n_val_batches)
    print(f"Epoch {epoch+1} - val_loss={avg_val_loss:.4f}")


def save_model(model, epoch):
    model_dir = f"model_epoch_{epoch+1}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))


def collect_params():
    this_mod = sys.modules[__name__]
    params = {k: v for k, v in vars(this_mod).items() if k.isupper()}
    return params


if __name__ == "__main__":
    matplotlib.use("Agg")  # for background plot rendering
    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    file_csv = "OscGrid_dataset/CSV_format_v1.1/labeled.csv"

    df_train, df_val, df_test = load_data(file_csv)

    # Get class labels from training set
    # train_labels = np.array(df_train["target"].values)

    # Compute class weights
    # class_weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=np.unique(train_labels),
    #     y=train_labels,
    # )

    # class_weights
    # array([0.52659444, 5.20135431, 0.87746531, 1.30021835])

    # Convert to tensor and send to target device
    # class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    train_dataset = CustomDataset(df_train)
    val_dataset = CustomDataset(df_val)
    # test_dataset = CustomDataset(df_test)

    model = Autoencoder()
    model.to(device)

    # targets = dataset.data["target"]
    # num_classes = len(np.unique(targets))
    # weights = {
    #     target: len(dataset) / value / num_classes
    #     for target, value in dict(targets.value_counts()).items()
    # }
    # print(f"weights = {weights}")

    criterion = SpectralLoss()  # weights=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataset = train_dataset

    dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,  # number of subprocess workers
            pin_memory=True,  # speeds up CUDA operations
        )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 128,
        shuffle=False,  # no shuffling
        num_workers=4,
        pin_memory=True,
    )

    print(f"Steps in one epoch: {len(dataloader)}")

    for epoch in range(EPOCHS):
        total_loss = 0
        total_batches = len(dataloader)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        for i, batch in enumerate(dataloader):
            progress = (i + 1) / total_batches * 100
            print(f'\rBatch {i+1}/{total_batches} ({progress:.1f}%)', end='', flush=True)

            targets = batch[1].squeeze().to(device)
            batch = batch[0].to(device)
            batch = torch.permute(batch, (0, 2, 1))

            output = model(batch)

            loss = criterion(output, batch, targets)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM
            )
            optimizer.step()

        model.eval()

        validation(model, val_dataloader, epoch)

        save_model(model, epoch)

        model.train()
