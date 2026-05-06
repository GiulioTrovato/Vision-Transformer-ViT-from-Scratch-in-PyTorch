import torch
import torch.nn as nn
import torch.optim as optim

from src.vit import ViT
from src.utils import get_dataloaders

# --- hyperparameters ---
BATCH_SIZE = 128
EPOCHS = 20
LR = 3e-4


def train_model():
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"start training on: {device}")

    # data loading
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # model
    model = ViT(img_size=32, patch_size=4, in_channels=3, n_classes=10, emb_size=256, depth=6, num_heads=8)
    model = model.to(device)

    # optimization engine
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

        # --- TRAINING LOOP ---
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # reset gradient to zero

            output = model(images) # predict labels
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

        # --- VALIDATION LOOP ---
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad(): # no more backpropagation
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = outputs.argmax(dim=1) # pick the predicted class

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"accuracy on test set: {accuracy:.2f}%")


if __name__ == "__main__":
    train_model()