import os
import random
import shutil
import sys
import argparse
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.fft as fft
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.models import Autoencoder, GRU, MultiScaleAttentionCNN, MLP, CNN, TransformerClassifier

FRAME_SIZE = 32
BOTTLENECK_DIM = 40
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-03
MAX_GRAD_NORM = 10
SEED = 443
# dataset columns used for model training:
SUBSET = ["IA", "IC", "UA BB", "UB BB", "UC BB"]
# new names for selected columns
FEATURE_NAMES = ["IA", "IC", "UA", "UB", "UC"]
RATIOS = {"train": 0.60, "val": 0.10, "test": 0.30}

# Names of problematic files with duplicate data
# Exclude them from the training dataset:
remove_files = []

class CustomDataset(Dataset):

    def __init__(self, dt: pd.DataFrame, stride: int = 1):
        self.data = dt
        self.frame_size = FRAME_SIZE
        self.stride = stride
        self.indexes = [
            idx
            for _, group in dt.groupby("block_id")
            for idx in group.index[: 1 - FRAME_SIZE : stride]  # Skip last FRAME_SIZE-1 samples per block with stride
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

        # Use weighted majority voting for window labeling
        # Center samples have more influence than edge samples
        frame_targets = frame["target"]
        weights = np.hanning(len(frame_targets))  # Hanning window weights
        
        # Initialize weighted counts as floats
        weighted_counts = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for i, target_val in enumerate(frame_targets):
            weighted_counts[target_val] += weights[i]
        
        # Find the class with maximum weighted count
        target = max(weighted_counts, key=weighted_counts.get)
        
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


def validation(model, data_loader, epoch, model_type="autoencoder", criterion=None, dataset_name="val"):
    val_loss = 0.0
    n_val_batches = 0
    
    if model_type == "classification":
        all_preds = []
        all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            targets = batch[1].squeeze().to(device)
            
            if model_type == "autoencoder":
                inputs = inputs.permute(0, 2, 1)
                output = model(inputs)
                loss = criterion(output, inputs, targets)
            else:  # classification
                output = model(inputs)
                loss = criterion(output, targets)
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            val_loss += loss.item()
            n_val_batches += 1

    avg_val_loss = val_loss / max(1, n_val_batches)
    
    if model_type == "classification":
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        print(f"Epoch {epoch+1} - {dataset_name}_loss={avg_val_loss:.4f}, {dataset_name}_accuracy={accuracy:.4f}, {dataset_name}_f1={f1:.4f}")
        return avg_val_loss, accuracy, f1
    else:
        print(f"Epoch {epoch+1} - {dataset_name}_loss={avg_val_loss:.4f}")
        return avg_val_loss


def test_model(model, test_dataloader, model_type="classification", criterion=None, model_name="model", seed=42):
    """Test model on test dataset and print detailed results"""
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    
    test_loss = 0.0
    n_test_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch[0].to(device)
            targets = batch[1].squeeze().to(device)
            
            if model_type == "autoencoder":
                inputs = inputs.permute(0, 2, 1)
                output = model(inputs)
                loss = criterion(output, inputs, targets)
            else:  # classification
                output = model(inputs)
                loss = criterion(output, targets)
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            test_loss += loss.item()
            n_test_batches += 1

    avg_test_loss = test_loss / max(1, n_test_batches)
    
    if model_type == "classification":
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score (weighted): {f1:.4f}")
        print(f"Test F1 Score (macro): {f1_macro:.4f}")
        
        # Generate classification report
        report = classification_report(all_targets, all_preds, target_names=['ML0', 'ML1', 'ML2', 'ML3'])
        print("\nDetailed Classification Report:")
        print(report)
        
        # Save classification report to file
        report_filename = f"classification_report_{model_name}_seed{seed}.txt"
        with open(report_filename, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Random Seed: {seed}\n")
            f.write(f"Test Loss: {avg_test_loss:.4f}\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Test F1 Score (weighted): {f1:.4f}\n")
            f.write(f"Test F1 Score (macro): {f1_macro:.4f}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(report)
        
        print(f"\nClassification report saved to: {report_filename}")
        
        return avg_test_loss, accuracy, f1
    else:
        print(f"Test Loss: {avg_test_loss:.4f}")
        return avg_test_loss


def save_model(model, epoch):
    model_dir = f"model_epoch_{epoch+1}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))


def collect_params():
    this_mod = sys.modules[__name__]
    params = {k: v for k, v in vars(this_mod).items() if k.isupper()}
    return params


def parse_args():
    parser = argparse.ArgumentParser(description='Train different neural network models')
    parser.add_argument('--model', type=str, choices=['autoencoder', 'gru', 'multi_scale_cnn', 'mlp', 'cnn', 'transformer'],
                       default='autoencoder', help='Model type to train')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--data_path', type=str, default="OscGrid_dataset/CSV_format_v1.2/labeled.csv",
                       help='Path to CSV data file')
    parser.add_argument('--stride', type=int, default=10,
                       help='Stride for dataset sampling (e.g., 10 for every 10th point)')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden size for models (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers for MLP model (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for regularization (default: 0.3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for L2 regularization (default: 1e-4)')
    parser.add_argument('--use_scheduler', action='store_true',
                       help='Use learning rate scheduler')
    parser.add_argument('--oversample', action='store_true',
                       help='Use oversampling for minority classes')
    return parser.parse_args()


if __name__ == "__main__":
    matplotlib.use("Agg")  # for background plot rendering
    args = parse_args()
    
    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    file_csv = args.data_path

    # Check if data file exists
    if not os.path.exists(file_csv):
        print(f"Error: Data file '{file_csv}' not found.")
        print("Please provide the correct path to your data file using --data_path argument.")
        sys.exit(1)

    df_train, df_val, df_test = load_data(file_csv)

    train_dataset = CustomDataset(df_train, stride=args.stride)
    val_dataset = CustomDataset(df_val, stride=args.stride)
    test_dataset = CustomDataset(df_test, stride=args.stride)

    # Initialize model based on type
    if args.model == "autoencoder":
        model = Autoencoder(frame_size=FRAME_SIZE, bottleneck_dim=BOTTLENECK_DIM)
        criterion = SpectralLoss()
        model_type = "autoencoder"
    else:  # classification models
        if args.model == "gru":
            model = GRU(
                frame_size=FRAME_SIZE,
                channel_num=5,
                output_size=4,  # 4 classes
                hidden_size=args.hidden_size,
                num_layers=2,
                dropout=0.4
            )
        elif args.model == "multi_scale_cnn":
            model = MultiScaleAttentionCNN(
                frame_size=FRAME_SIZE,
                channel_num=5,
                hidden_size=args.hidden_size,
                output_size=4  # 4 classes
            )
        elif args.model == "mlp":
            model = MLP(
                frame_size=FRAME_SIZE,
                channel_num=5,
                hidden_size=args.hidden_size,
                output_size=4,  # 4 classes
                num_layers=args.num_layers
            )
        elif args.model == "cnn":
            model = CNN(
                frame_size=FRAME_SIZE,
                channel_num=5,
                hidden_size=args.hidden_size,
                output_size=4  # 4 classes
            )
        elif args.model == "transformer":
            model = TransformerClassifier(
                frame_size=FRAME_SIZE,
                channel_num=5,
                hidden_size=args.hidden_size,
                output_size=4,  # 4 classes
                num_layers=3,
                num_heads=8,
                dropout=args.dropout
            )
        
        # Compute class weights for imbalanced data - use more aggressive weighting
        class_counts = df_train['target'].value_counts().sort_index()
        print(f"Class distribution in training data: {dict(class_counts)}")
        
        # Use inverse frequency weighting with smoothing
        total_samples = len(df_train)
        class_weights = total_samples / (len(class_counts) * class_counts.values)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"Class weights: {class_weights}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        model_type = "classification"

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    else:
        scheduler = None

    dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,  # number of subprocess workers
            pin_memory=True,  # speeds up CUDA operations
        )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Training {args.model} model")
    print(f"Model type: {model_type}")
    print(f"Steps in one epoch: {len(dataloader)}")

    # Training metrics tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    patience_counter = 0
    patience = 10  # Early stopping patience

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_batches = len(dataloader)
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        for i, batch in enumerate(dataloader):
            progress = (i + 1) / total_batches * 100
            print(f'\rBatch {i+1}/{total_batches} ({progress:.1f}%)', end='', flush=True)

            inputs = batch[0].to(device)
            targets = batch[1].squeeze().to(device)

            if model_type == "autoencoder":
                inputs = inputs.permute(0, 2, 1)
                output = model(inputs)
                loss = criterion(output, inputs, targets)
            else:  # classification
                output = model(inputs)
                loss = criterion(output, targets)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM
            )
            optimizer.step()

        avg_train_loss = total_loss / total_batches
        train_losses.append(avg_train_loss)
        
        model.eval()

        if model_type == "autoencoder":
            val_loss = validation(model, val_dataloader, epoch, "autoencoder", criterion)
            val_losses.append(val_loss)
        else:
            val_loss, val_accuracy, val_f1 = validation(model, val_dataloader, epoch, "classification", criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Early stopping based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), "best_model.pt")
                print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Learning rate scheduling
        if scheduler and model_type == "classification":
            scheduler.step(val_loss)
        elif scheduler:
            scheduler.step(val_loss)

        save_model(model, epoch)
        model.train()

    # Load best model for final testing
    if model_type == "classification" and os.path.exists("best_model.pt"):
        print("\nLoading best model for final testing...")
        model.load_state_dict(torch.load("best_model.pt"))
    
    # Final test on test dataset after all epochs
    if model_type == "classification":
        print("\n" + "="*60)
        print("RUNNING FINAL TEST ON TEST DATASET")
        print("="*60)
        model.eval()
        test_model(model, test_dataloader, "classification", criterion, args.model, SEED)
        model.train()
