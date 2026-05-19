"""
ResNet50-based Accident Classifier with Severity Prediction
Multi-task learning: Accident Detection + Severity Score Regression
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from typing import Tuple


def _get_resnet50_weights(pretrained: bool):
    return ResNet50_Weights.DEFAULT if pretrained else None


class ResNet50AccidentDetector(nn.Module):
    """
    ResNet50 multi-head network for accident detection and severity prediction
    
    Architecture:
    - Backbone: ResNet50 pre-trained on ImageNet
    - Head 1: Accident vs Non-Accident (Binary Classification)
    - Head 2: Severity Score Regression (0-100 scale)
    """
    
    def __init__(self, num_severity_classes=4, pretrained: bool = False):
        """
        Initialize ResNet50 accident detector
        
        Args:
            num_severity_classes: Number of severity classes (default: 4)
                - 0: Normal/Non-Accident
                - 1: Minor Accident
                - 2: Moderate Accident
                - 3: Major/Severe Accident
            pretrained: Use ImageNet pre-trained weights
        """
        super(ResNet50AccidentDetector, self).__init__()
        self.num_severity_classes = num_severity_classes
        
        # Load ResNet50 backbone
        resnet50 = models.resnet50(weights=_get_resnet50_weights(pretrained))
        
        # Remove the original classification head
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Feature dimension from ResNet50
        self.feature_dim = 2048
        
        # ==================== Accident Detection Head ====================
        # Binary classification: Accident (1) vs Non-Accident (0)
        self.accident_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Binary: [non-accident, accident]
        )
        
        # ==================== Severity Classification Head ====================
        # Multi-class severity: [normal, minor, moderate, major/severe]
        self.severity_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_severity_classes)  # Multi-class severity
        )
        
        # ==================== Severity Regression Head (Alternative) ====================
        # Direct severity score regression (0-100)
        self.severity_regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Single value: severity score
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through ResNet50 with multi-head outputs
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            accident_logits: (batch_size, 2) - logits for [non-accident, accident]
            severity_logits: (batch_size, num_severity_classes) - class logits
            severity_score: (batch_size, 1) - continuous severity score (0-100)
        """
        # Extract features from ResNet50 backbone
        features = self.backbone(x)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 2048)
        
        # Get predictions from each head
        accident_logits = self.accident_head(features)  # (batch_size, 2)
        severity_logits = self.severity_head(features)  # (batch_size, num_severity_classes)
        severity_score = self.severity_regression_head(features)  # (batch_size, 1)
        
        # Clamp severity score to [0, 100]
        severity_score = torch.clamp(severity_score, min=0.0, max=100.0)
        
        return accident_logits, severity_logits, severity_score
    
    def predict(self, x: torch.Tensor) -> dict:
        """
        Get human-readable predictions from input images
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Dictionary with predictions:
            - is_accident: Boolean tensor (batch_size,) - whether accident detected
            - accident_confidence: Float tensor (batch_size,) - confidence of accident (0-1)
            - severity_class: Integer tensor (batch_size,) - 0=normal, 1=minor, 2=moderate, 3=severe
            - severity_score: Float tensor (batch_size,) - severity score (0-100)
            - severity_label: List of strings - ['Normal', 'Minor', 'Moderate', 'Severe']
        """
        self.eval()
        with torch.no_grad():
            accident_logits, severity_logits, severity_score = self.forward(x)
            
            # Get accident probability
            accident_probs = torch.softmax(accident_logits, dim=1)
            is_accident = accident_probs[:, 1] > 0.5  # Threshold at 0.5
            accident_confidence = accident_probs[:, 1]  # Probability of accident
            
            # Get severity class
            severity_class = torch.argmax(severity_logits, dim=1)
            
            # Severity labels
            severity_labels = ['Normal', 'Minor', 'Moderate', 'Severe']
            severity_label = [severity_labels[int(sc)] for sc in severity_class]
            
        return {
            'is_accident': is_accident,
            'accident_confidence': accident_confidence,
            'severity_class': severity_class,
            'severity_score': severity_score.squeeze(-1),
            'severity_label': severity_label
        }


class ResNet50MultiTask(nn.Module):
    """
    Simplified ResNet50 for inference with combined loss during training
    Focuses on accident detection as primary task
    """
    
    def __init__(self, pretrained: bool = False):
        super(ResNet50MultiTask, self).__init__()
        
        # Load ResNet50 backbone
        resnet50 = models.resnet50(weights=_get_resnet50_weights(pretrained))
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Single unified head for accident classification
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # Binary: accident vs non-accident
        )
    
    def forward(self, x):
        """Forward pass through ResNet50 backbone and classifier"""
        features = self.backbone(x).view(x.size(0), -1)
        return self.classifier(features)


def load_resnet50_model(model_path: str, device='cpu', architecture='multi_task'):
    """
    Load a pre-trained ResNet50 model from file
    
    Args:
        model_path: Path to saved model weights
        device: Device to load model on ('cpu' or 'cuda')
        architecture: 'multi_task' for multi-head or 'simple' for single head
    
    Returns:
        Model loaded on specified device
    """
    if architecture == 'multi_task':
        model = ResNet50AccidentDetector(pretrained=False)
    else:
        model = ResNet50MultiTask(pretrained=False)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded ResNet50 model from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
    
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing ResNet50AccidentDetector...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50AccidentDetector(num_severity_classes=4, pretrained=False)
    model = model.to(device)
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    accident_logits, severity_logits, severity_score = model(dummy_input)
    
    print(f"✓ Model initialized successfully on {device}")
    print(f"  Accident logits shape: {accident_logits.shape}")
    print(f"  Severity logits shape: {severity_logits.shape}")
    print(f"  Severity score shape: {severity_score.shape}")
    
    # Test prediction
    predictions = model.predict(dummy_input)
    print(f"\n✓ Predictions:")
    print(f"  Is accident: {predictions['is_accident']}")
    print(f"  Accident confidence: {predictions['accident_confidence']}")
    print(f"  Severity class: {predictions['severity_class']}")
    print(f"  Severity labels: {predictions['severity_label']}")
