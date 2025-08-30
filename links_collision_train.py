import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random

from links_collision_model import LinksCollisionModel
from links_collision_binary_dataset import LinkCollisionsBinaryDataset
from links_collision_test import test_model

# ===== Config =====
batch_size = 1024
test_batch_size = 1024
num_epochs = 30
learning_rate = 0.002
INPUT_DIM = 1854  # 8 lines: 135+353+135+353+353+135+42+353
PROJ_DIM = 256

# ===== Paths =====
data_dir_path = "data/link_collisions_data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # --- READ ALL FILES FROM ONE DIRECTORY ---
    files = [
        os.path.join(data_dir_path, fn)
        for fn in sorted(os.listdir(data_dir_path))
        if os.path.isfile(os.path.join(data_dir_path, fn))
           and fn.lower().endswith((".txt", ".dat", ".data"))
    ]
    if not files:
        raise RuntimeError(f"No data files (*.txt|*.dat|*.data) found in {data_dir_path}")
    # 80/20 split BY FILES (prevents loading all data at once and avoids leakage within a file)
    random.shuffle(files)
    split_idx = max(1, int(0.8 * len(files)))
    train_files = files[:split_idx]
    test_files = files[split_idx:] if split_idx < len(files) else files[-1:]

    train_dataset = LinkCollisionsBinaryDataset(train_files)
    print(f"Train dataset len: {len(train_dataset)}")
    test_dataset = LinkCollisionsBinaryDataset(test_files)
    print(f"Test dataset len: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    model = LinksCollisionModel(INPUT_DIM, PROJ_DIM, device=device).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9997)

    # --- Initial test before training ---
    print("Initial test evaluation:")
    test_model(model, test_loader, criterion)

    print(f"Training on {len(train_files)} files, testing on {len(test_files)} files. INPUT_DIM={INPUT_DIM}")

    # --- Training loop ---
    os.makedirs("models", exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        train_sample_index = 0
        min_loss = 1.0
        max_loss = 0
        loss_more_than_minus_3 = 0
        loss_less_than_minus_4 = 0
        for features, labels in train_loader:
            features = features.to(device)  # (B, D)
            labels = labels.to(device)  # (B, 8)

            optimizer.zero_grad()
            outputs = model(features)  # (B, 8)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            bs = features.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            if loss.item() < min_loss:
                min_loss = loss.item()
            if loss.item() > max_loss:
                max_loss = loss.item()
            if loss.item() > 0.001:
                loss_more_than_minus_3 += 1
            if loss.item() < 0.0001:
                loss_less_than_minus_4 += 1

            train_sample_index += 1
            if train_sample_index % 200 == 0:
                print(
                    f"Epoch [{epoch:02d}] [{train_sample_index:04d}] [{min_loss:.6f}, {max_loss:.6f}] "
                    f"[{loss_more_than_minus_3:03}  {loss_less_than_minus_4:03}]")
                min_loss = 1.0
                max_loss = 0.0
                loss_more_than_minus_3 = loss_less_than_minus_4 = 0

        epoch_train_loss = total_loss / total_samples if total_samples else 0.0
        print(f"[{epoch:02d}/{num_epochs}] train_loss={epoch_train_loss:.6f}")

        # --- Test after each epoch ---
        test_model(model, test_loader, criterion)

        # save checkpoint each epoch
        torch.save(model.state_dict(), "models/links_collision_kan.pth")

    print("Done.")


if __name__ == "__main__":
    main()
