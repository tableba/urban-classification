from src.dataset.read_data import ReadTifs
import torch
from torch.utils.data import DataLoader
from src.dataset.preprocessing import clean_s2_data, normalize_bands, extract_patches, SentinelDataset
from src.models.simplecnn import SimpleCNN
import numpy as np
import matplotlib.pyplot as plt
import os
import random

reader = ReadTifs()

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#test to split data in 3 parts and select tiles from each part, to get a more representative sample of the data
def prepare_data(max_tiles_per_third=8):
    all_patches = []
    reader = ReadTifs()

    s2_files = {
        os.path.splitext(f)[0].split("_")[1]: os.path.join(reader.s2_dir, f)
        for f in os.listdir(reader.s2_dir)
        if f.lower().endswith(".tif")
    }
    dw_files = {
        os.path.splitext(f)[0].split("_")[1]: os.path.join(reader.dw_dir, f)
        for f in os.listdir(reader.dw_dir)
        if f.lower().endswith(".tif")
    }
    common_tiles = sorted(set(s2_files.keys()) & set(dw_files.keys()))

    # Split into thirds
    third = len(common_tiles) // 3
    first_third = common_tiles[:third]
    second_third = common_tiles[third:2*third]
    last_third = common_tiles[2*third:]

    # Randomly sample from each third
    selected_tiles = (
        random.sample(first_third, min(max_tiles_per_third, len(first_third))) +
        random.sample(second_third, min(max_tiles_per_third, len(second_third))) +
        random.sample(last_third, min(max_tiles_per_third, len(last_third)))
    )

    print(f"Selected tiles: {selected_tiles}")


    tiles_processed = 0
    for key in selected_tiles:
        print(f"Processing tile: {key}")
        
        s2_path = s2_files[key]
        dw_path = dw_files[key]
        s2_data = reader._read_preview(s2_path)
        dw_data = reader._read_preview(dw_path)

        for s2_patch, dw_patch in extract_patches(s2_data, dw_data):
            s2_patch = clean_s2_data(s2_patch)
            s2_patch = normalize_bands(s2_patch)
            all_patches.append((s2_patch, dw_patch))
        tiles_processed += 1
        print(f"Patches extracted from tile {key}: {len(all_patches)} (Total tiles processed: {tiles_processed}/{len(selected_tiles)})")

    return all_patches


def init_model(load_path=None):
    print("Initializing model...")
    model = SimpleCNN().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if load_path:
        model.load_state_dict(torch.load(load_path, map_location=DEVICE))
        print(f"Model loaded from {load_path}")
    else:        
        print("Training model from scratch.")
    

    return model, criterion, optimizer


def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (s2_batch, dw_batch) in enumerate(dataloader):
            s2_batch, dw_batch = s2_batch.to(DEVICE), dw_batch.to(DEVICE)

            outputs = model(s2_batch)
            loss = criterion(outputs, dw_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    print("Training complete.")


def evaluate_model(model, dataloader, criterion):
        print("Evaluating model...")
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i, (s2_batch, dw_batch) in enumerate(dataloader):
                s2_batch, dw_batch = s2_batch.to(DEVICE), dw_batch.to(DEVICE)
                outputs = model(s2_batch)
                loss = criterion(outputs, dw_batch)
                total_loss += loss.item()

                if i % 10 == 9:
                    print(f"Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss


def print_class_distribution(patches):
    labels = []
    for _, dw in patches:
        labels.extend(dw.flatten())
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} pixels")

    
def train_test_split(patches):
    print("Splitting data into train and validation sets...")
    train_patches = patches[:int(0.8 * len(patches))]
    val_patches = patches[int(0.8 * len(patches)):]

    print(f"  Train patches: {len(train_patches)}")
    print_class_distribution(train_patches)

    print(f"  Val patches: {len(val_patches)}")
    print_class_distribution(val_patches)

    train_dataset = SentinelDataset(train_patches)
    val_dataset = SentinelDataset(val_patches)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    return train_dataloader, val_dataloader


def visualize_predictions(model, dataloader, num_samples=3):
    model.eval()
    with torch.no_grad():
        for i, (s2_batch, dw_batch) in enumerate(dataloader):
            if i >= num_samples:
                break
            s2_batch, dw_batch = s2_batch.to(DEVICE), dw_batch.to(DEVICE)
            outputs = model(s2_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = dw_batch.cpu().numpy()

            # Plot
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(labels[0], cmap="jet")
            plt.title("Ground Truth")
            plt.subplot(1, 2, 2)
            plt.imshow(preds[0], cmap="jet")
            plt.title("Prediction")
            plt.show()


if __name__ == "__main__":
    print("=== Starting CNN Testing ===")
    patches = prepare_data()

    """
    dataset = SentinelDataset(patches)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model, criterion, optimizer = init_model()
  
    train_model(model, dataloader, criterion, optimizer, EPOCHS)

    torch.save(model.state_dict(), "simple_cnn.pth")
    print("Model saved as simple_cnn.pth")
    """

    train, val = train_test_split(patches)
    model, criterion, optimizer = init_model()
    
    best_loss = float("inf")
    patience = 3  # Stop after 3 epochs without improvement
    for epoch in range(EPOCHS):
        train_model(model, train, criterion, optimizer, 1)  # Train for 1 epoch
        val_loss = evaluate_model(model, val, criterion)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                break

    visualize_predictions(model, val)

    """

    train_model(model, train, criterion, optimizer, EPOCHS)
    evaluate_model(model, val, criterion)

    visualize_predictions(model, val)

    print("=== CNN Testing Complete ===")
    """