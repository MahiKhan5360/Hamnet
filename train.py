import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100, checkpoint_dir='./checkpoints'):
    """Train the HAMNET model"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_loss = float('inf')
    best_val_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (data, targets) in enumerate(progress_bar):
            data = data.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                predictions = model(data)
                loss = criterion(predictions, targets)  # No uncertainty

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            preds = torch.sigmoid(predictions)
            dice = (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)
            train_dice += dice.item()

            progress_bar.set_postfix({
                'loss': loss.item(),
                'dice': dice.item()
            })

            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Batch/Loss", loss.item(), step)
            writer.add_scalar("Batch/Dice", dice.item(), step)

            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

        with torch.no_grad():
            for data, targets in progress_bar:
                data = data.to(device)
                targets = targets.to(device)

                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    predictions = model(data)
                    loss = criterion(predictions, targets)

                val_loss += loss.item()

                preds = torch.sigmoid(predictions)
                dice = (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)
                val_dice += dice.item()

                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'dice': dice.item()
                })

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        writer.add_scalar("Epoch/Train_Loss", avg_train_loss, epoch)
        writer.add_scalar("Epoch/Train_Dice", avg_train_dice, epoch)
        writer.add_scalar("Epoch/Val_Loss", avg_val_loss, epoch)
        writer.add_scalar("Epoch/Val_Dice", avg_val_dice, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_dice': avg_val_dice,
            }, os.path.join(checkpoint_dir, 'hamnet_best_loss.pth'))
            print(f"Saved best model (loss) at epoch {epoch+1}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_dice': best_val_dice,
            }, os.path.join(checkpoint_dir, 'hamnet_best_dice.pth'))
            print(f"Saved best model (dice) at epoch {epoch+1}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_dice': avg_val_dice,
            }, os.path.join(checkpoint_dir, f'hamnet_epoch_{epoch+1}.pth'))

    writer.close()
    return model
