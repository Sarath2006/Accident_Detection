"""
Training script for ResNet50 Accident Detector
Multi-task learning: Accident Detection + Severity Prediction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm
from datetime import datetime
from typing import Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.resnet50_classifier import ResNet50AccidentDetector
from training.data_loader import get_data_loaders


class AccidentDetectorTrainer:
    """Trainer for ResNet50 accident detector with multi-task learning"""
    
    def __init__(
        self,
        model: ResNet50AccidentDetector,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer
        
        Args:
            model: ResNet50AccidentDetector model
            device: 'cuda' or 'cpu'
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.accident_loss_fn = nn.CrossEntropyLoss()  # For binary classification
        self.severity_loss_fn = nn.CrossEntropyLoss()  # For severity class
        self.regression_loss_fn = nn.MSELoss()  # For severity score regression
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir='runs/accident_detector')
    
    def compute_loss(
        self,
        accident_logits: torch.Tensor,
        severity_logits: torch.Tensor,
        severity_score: torch.Tensor,
        accident_labels: torch.Tensor,
        severity_labels: torch.Tensor,
        severity_scores: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-task loss
        
        Loss = alpha * accident_loss + beta * severity_class_loss + gamma * regression_loss
        
        Args:
            accident_logits: (batch_size, 2)
            severity_logits: (batch_size, num_classes)
            severity_score: (batch_size, 1)
            accident_labels: (batch_size,) - 0 or 1
            severity_labels: (batch_size,) - class index 0-3
            severity_scores: (batch_size,) - continuous score 0-100
            alpha, beta, gamma: Loss weights
        
        Returns:
            Total loss and dict of individual losses
        """
        # Accident detection loss (most important)
        loss_accident = self.accident_loss_fn(accident_logits, accident_labels)
        
        # Severity classification loss
        loss_severity_class = self.severity_loss_fn(severity_logits, severity_labels)
        
        # Severity regression loss
        loss_regression = self.regression_loss_fn(
            severity_score.squeeze(-1),
            severity_scores.float()
        )
        
        # Weighted combination
        total_loss = (
            alpha * loss_accident +
            beta * loss_severity_class +
            gamma * loss_regression
        )
        
        return total_loss, {
            'accident': loss_accident.item(),
            'severity_class': loss_severity_class.item(),
            'regression': loss_regression.item(),
            'total': total_loss.item()
        }
    
    def train_epoch(self, train_loader, epoch: int) -> dict:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Epoch number
        
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        total_loss = 0
        losses_by_type = {'accident': 0, 'severity_class': 0, 'regression': 0}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle different batch formats
            if len(batch) == 2:  # Folder structure: (image, accident_label)
                images, accident_labels = batch
                images = images.to(self.device)
                accident_labels = accident_labels.to(self.device)
                severity_labels = accident_labels  # Use accident label as proxy
                severity_scores = accident_labels.float() * 50  # 0->0, 1->50 (rough estimate)
            else:  # CSV format: (image, accident_label, severity_score)
                images, accident_labels, severity_scores = batch
                images = images.to(self.device)
                accident_labels = accident_labels.to(self.device)
                severity_scores = severity_scores.to(self.device)
                # Convert severity scores to class labels
                severity_labels = torch.zeros_like(severity_scores, dtype=torch.long)
                severity_labels[severity_scores < 25] = 0  # Normal
                severity_labels[(severity_scores >= 25) & (severity_scores < 50)] = 1  # Minor
                severity_labels[(severity_scores >= 50) & (severity_scores < 75)] = 2  # Moderate
                severity_labels[severity_scores >= 75] = 3  # Severe
            
            # Forward pass
            self.optimizer.zero_grad()
            accident_logits, severity_logits, severity_score = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.compute_loss(
                accident_logits, severity_logits, severity_score,
                accident_labels, severity_labels, severity_scores
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key in losses_by_type:
                losses_by_type[key] += loss_dict[key]
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc_loss': f"{loss_dict['accident']:.4f}"
            })
        
        # Average losses
        num_batches = len(train_loader)
        avg_loss = {
            'total': total_loss / num_batches,
            'accident': losses_by_type['accident'] / num_batches,
            'severity_class': losses_by_type['severity_class'] / num_batches,
            'regression': losses_by_type['regression'] / num_batches
        }
        
        return avg_loss
    
    def validate(self, val_loader, epoch: int) -> dict:
        """
        Validate on validation set
        
        Args:
            val_loader: Validation data loader
            epoch: Epoch number
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0
        correct_accident = 0
        correct_severity = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            
            for batch in progress_bar:
                # Handle different batch formats
                if len(batch) == 2:
                    images, accident_labels = batch
                    images = images.to(self.device)
                    accident_labels = accident_labels.to(self.device)
                    severity_labels = accident_labels
                    severity_scores = accident_labels.float() * 50
                else:
                    images, accident_labels, severity_scores = batch
                    images = images.to(self.device)
                    accident_labels = accident_labels.to(self.device)
                    severity_scores = severity_scores.to(self.device)
                    severity_labels = torch.zeros_like(severity_scores, dtype=torch.long)
                    severity_labels[severity_scores < 25] = 0
                    severity_labels[(severity_scores >= 25) & (severity_scores < 50)] = 1
                    severity_labels[(severity_scores >= 50) & (severity_scores < 75)] = 2
                    severity_labels[severity_scores >= 75] = 3
                
                # Forward pass
                accident_logits, severity_logits, severity_score = self.model(images)
                
                # Compute loss
                loss, _ = self.compute_loss(
                    accident_logits, severity_logits, severity_score,
                    accident_labels, severity_labels, severity_scores
                )
                
                # Accumulate
                total_loss += loss.item()
                
                # Compute accuracies
                pred_accident = torch.argmax(accident_logits, dim=1)
                pred_severity = torch.argmax(severity_logits, dim=1)
                
                correct_accident += (pred_accident == accident_labels).sum().item()
                correct_severity += (pred_severity == severity_labels).sum().item()
                total_samples += accident_labels.size(0)
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc_acc': f"{correct_accident/total_samples:.2%}"
                })
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accident_accuracy': correct_accident / total_samples,
            'severity_accuracy': correct_severity / total_samples
        }
        
        return metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        save_dir: str = 'models'
    ):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Log to TensorBoard
            self.writer.add_scalars('Loss', {
                'train_total': train_loss['total'],
                'train_accident': train_loss['accident'],
                'val_total': val_metrics['loss']
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'val_accident': val_metrics['accident_accuracy'],
                'val_severity': val_metrics['severity_accuracy']
            }, epoch)
            
            # Print metrics
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss['total']:.4f} (Accident: {train_loss['accident']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: Accident={val_metrics['accident_accuracy']:.2%}, Severity={val_metrics['severity_accuracy']:.2%}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                model_path = os.path.join(save_dir, 'resnet50_accident_best.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"  ✓ Saved best model to {model_path}")
            else:
                patience_counter += 1
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        self.writer.close()


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ResNet50 Accident Detector")
    parser.add_argument('--dataset-root', type=str, default='datasets',
                        help='Root directory of dataset')
    parser.add_argument('--dataset-type', type=str, default='folder', choices=['folder', 'csv'],
                        help='Type of dataset')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='Path to CSV file (required if dataset-type=csv)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet pre-trained weights (requires download)')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = ResNet50AccidentDetector(num_severity_classes=4, pretrained=args.pretrained)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        dataset_root=args.dataset_root,
        dataset_type=args.dataset_type,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True
    )
    
    # Create trainer
    trainer = AccidentDetectorTrainer(
        model=model,
        device=device,
        learning_rate=args.lr
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
