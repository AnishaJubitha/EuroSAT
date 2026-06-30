from src.utils import *
import torch

def train_fusion(model, dataloader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for rgb_images, ms_images, labels in dataloader:

        rgb_images = rgb_images.to(device)
        ms_images = ms_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(rgb_images, ms_images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        _, predictions = torch.max(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate_fusion(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for rgb_images, ms_images, labels in dataloader:
            rgb_images = rgb_images.to(device)
            ms_images = ms_images.to(device)
            labels = labels.to(device)

            outputs = model(rgb_images, ms_images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)

            _, predictions = torch.max(outputs, dim=1)
            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss/total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def training_loop_fusion(
        train_loader, val_loader,
        model, criterion, optimizer, 
        path_file=None, scheduler=None
):
    device = get_device()

    history = {
        'train_loss' : [],
        'train_acc' : [],
        'val_loss' : [],
        'val_acc' : [],
        'lr' : []
    }

    best_val_acc = 0.0

    if path_file is not None:
        best_model_path = CHECKPOINT_PATH / path_file
    else:
        best_model_path = None

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_fusion(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate_fusion(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]"
            f"Train Loss: {train_loss:.4f}"
            f"Train Acc: {train_acc:.4f}"
            f"Val Loss: {val_loss:.4f}"
            f"Val Acc: {val_acc:.4f}"
            f"LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)

            print(f"New Best Fusion Model Saved. Val Acc: {val_acc:.4f}")

    print(f"Training Complete. Best Val Accuracy: {best_val_acc:.4f}")

    return history