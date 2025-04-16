import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2

try:
    from sklearn.metrics import precision_recall_curve, average_precision_score
except ModuleNotFoundError:
    print("Installing scikit-learn...")
    %pip install --force-reinstall scikit-learn==1.5.2
    from sklearn.metrics import precision_recall_curve, average_precision_score

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    # Convert to numpy arrays
    predictions = torch.sigmoid(predictions).cpu().numpy()  # Apply sigmoid for probabilities
    targets = targets.cpu().numpy()

    # Flatten arrays
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(targets_flat, predictions_flat)

    # Calculate average precision
    ap = average_precision_score(targets_flat, predictions_flat)

    # Calculate Dice at optimal threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_dice = f1_scores[optimal_idx]

    # Calculate IoU at optimal threshold
    predictions_binary = (predictions_flat >= optimal_threshold).astype(np.float32)
    intersection = np.sum(predictions_binary * targets_flat)
    union = np.sum(predictions_binary) + np.sum(targets_flat) - intersection
    iou = intersection / (union + 1e-8)

    return {
        'optimal_threshold': optimal_threshold,
        'dice': optimal_dice,
        'iou': iou,
        'average_precision': ap
    }

def visualize_predictions(images, masks, predictions, save_dir, num_samples=5):
    """Visualize and save model predictions"""
    os.makedirs(save_dir, exist_ok=True)

    # Select random samples
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)

    for i, idx in enumerate(indices):
        # Get data
        image = images[idx].permute(1, 2, 0).cpu().numpy()
        mask = masks[idx].squeeze().cpu().numpy()
        prediction = torch.sigmoid(predictions[idx]).squeeze().cpu().numpy()  # Apply sigmoid

        # Denormalize image for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image * std + mean).clip(0, 1)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot image
        axes[0].imshow(image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Plot ground truth mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # Plot prediction
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{i+1}.png'), dpi=200)
        plt.close()

def evaluate_model(model, test_loader, criterion, device, checkpoint_path=None, save_dir='./results'):
    """Evaluate the HAMNET model on test data"""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load best model if checkpoint path is provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
        print(f"Checkpoint info - Epoch: {checkpoint['epoch']}, Val Dice: {checkpoint.get('val_dice', 'N/A')}")

    # Set model to evaluation mode
    model.eval()

    # Initialize metrics
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    all_images = []

    # Evaluate model
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(data)  # HAMNET returns single output
            loss = criterion(predictions, targets)  # HAMNETLoss returns scalar

            # Update metrics
            test_loss += loss.item()

            # Store predictions and targets for metric calculation
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_images.append(data.cpu())

    # Calculate average loss
    avg_test_loss = test_loss / len(test_loader)

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_images = torch.cat(all_images, dim=0)

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)

    # Print results
    print("\nTest Results:")
    print(f"Loss: {avg_test_loss:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"Dice Coefficient: {metrics['dice']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")

    # Visualize predictions
    visualize_predictions(all_images, all_targets, all_predictions, save_dir)

    # Save metrics to file
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Test Loss: {avg_test_loss:.4f}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    return metrics
