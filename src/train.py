import torch
from src.config import *
from src.dataloader import *
from src.utils import * 

def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predictions = torch.max(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

    return epoch_loss, epoch_acc
    

def validate_epoch(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total

    return epoch_loss, epoch_acc
        

def training_loop(train_loader, val_loader, model, criterion, optimizer, path_file=None, scheduler=None, early_stop=None):
        
    DEVICE = get_device()

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

        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE
        )

        val_loss, val_acc = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=DEVICE
        )
        
        # Scheduler
        if scheduler is not None:
            scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"Epoch [{epoch + 1}/{EPOCHS}]"
        f" Train Loss : {train_loss:.4f}"
        f" Train ACC : {train_acc:.4f}"
        f" Val Loss : {val_loss:.4f}"
        f" Val ACC : {val_acc:.4f}"
        f" LR : {current_lr:.6f}"
        )


        if early_stop is not None:

            improved = early_stop(val_acc, model)

            if improved:
                print(f'New Best Model Saved. Val Acc : {val_acc:.4f}')

            if early_stop.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}."
                    f"Best Val Accuracy : ' {best_val_acc:.4f}.")
            
                break

        else:
            if val_acc > best_val_acc:
                    best_val_acc = val_acc

                    if best_model_path is not None:
                        torch.save(
                            model.state_dict(),
                            best_model_path
                        )

                    print(f'New Best Model Saved. Val Acc : {val_acc:.4f}')

    if early_stop is not None:
        print(f"Training Complete. Best Val Acc: {early_stop.best_score:.4f}")  
    else:
        print(f"Training Complete. Best Val Accuracy : {best_val_acc:.4f}")             

    return history