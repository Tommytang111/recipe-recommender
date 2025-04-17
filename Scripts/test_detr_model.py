import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from transformers import DetrForObjectDetection, DetrConfig

# Enhanced DETR model reinitializer
def reinitialize_detr_model(model, num_classes, device):
    """
    Properly reinitialize DETR model with correct number of classes
    and handle the ImageLoss cross-entropy weight tensor issues.
    """
    from torch import nn
    import torch
    
    print(f"Reinitializing model for {num_classes} classes...")
    
    # 1. Replace the classification head
    model.class_labels_classifier = nn.Linear(
        model.class_labels_classifier.in_features, 
        num_classes + 1  # +1 for the "no object" class
    )
    
    # 2. Fix the empty_weight in the loss function
    empty_weight = torch.ones(num_classes + 1)
    empty_weight[-1] = 0.1  # Lower weight for the "no-object" class
    model.config.num_labels = num_classes + 1
    
    # 3. Update all modules with empty_weight attribute - simplify to always use direct assignment
    for module in model.modules():
        if hasattr(module, 'empty_weight'):
            # Always use direct assignment for testing consistency
            module.empty_weight = empty_weight.to(device)
            module.num_classes = num_classes
    
    # 4. Special case for loss function in DETR
    if hasattr(model, 'criterion'):
        if hasattr(model.criterion, 'empty_weight'):
            model.criterion.empty_weight = empty_weight.to(device)
            model.criterion.num_classes = num_classes
    
    # 5. Move model to device
    model = model.to(device)
    return model

class TestDETRModelReinitialization(unittest.TestCase):
    
    def setUp(self):
        # Set up mock objects for testing
        self.device = torch.device("cpu")
        self.num_classes = 10
        
        # Create a minimal config for DETR
        self.config = DetrConfig()
        self.config.num_labels = 91  # Original COCO classes
        
        # Create mock model with proper return values
        self.model = MagicMock()
        self.model.class_labels_classifier = nn.Linear(256, 91)
        self.model.config = self.config
        
        # Create a mock module with empty_weight
        self.mock_module = MagicMock()
        # Create a mock tensor with a mock to() method
        mock_tensor = MagicMock()
        mock_tensor.to = MagicMock(return_value=mock_tensor)
        self.mock_module.empty_weight = mock_tensor
        self.mock_module.num_classes = 90
        
        # Set up modules method to return our mock_module
        self.model.modules.return_value = [self.mock_module]
        
        # Make to() return the model itself for device moves
        self.model.to.return_value = self.model
    
    def test_classifier_replacement(self):
        """Test if the classifier is replaced with correct output dimension."""
        # Run the function
        result = reinitialize_detr_model(self.model, self.num_classes, self.device)
        
        # Check that a Linear layer with correct output dimension was created
        self.assertEqual(result.class_labels_classifier.out_features, self.num_classes + 1)
    
    def test_empty_weight_update(self):
        """Test if the empty_weight tensor is correctly updated."""
        # Run the function
        reinitialize_detr_model(self.model, self.num_classes, self.device)
        
        # Verify the module was accessed and updated
        self.mock_module.empty_weight.to.assert_called_with(self.device)
        self.assertEqual(self.mock_module.num_classes, self.num_classes)
    
    def test_config_update(self):
        """Test if the config is updated with the new number of classes."""
        # Before
        self.assertEqual(self.config.num_labels, 91)
        
        # Run the function
        reinitialize_detr_model(self.model, self.num_classes, self.device)
        
        # After - config should be modified directly
        self.assertEqual(self.config.num_labels, self.num_classes + 1)
    
    def test_device_placement(self):
        """Test if model is moved to the correct device."""
        # Run the function
        reinitialize_detr_model(self.model, self.num_classes, self.device)
        
        # Check if to() was called with the correct device
        self.model.to.assert_called_with(self.device)
    
    def test_with_criterion(self):
        """Test handling of model with criterion attribute."""
        # Add criterion with empty_weight
        criterion = MagicMock()
        # Create a mock tensor with a mock to() method
        mock_tensor = MagicMock()
        mock_tensor.to = MagicMock(return_value=mock_tensor)
        criterion.empty_weight = mock_tensor
        criterion.num_classes = 90
        self.model.criterion = criterion
        
        # Run the function
        reinitialize_detr_model(self.model, self.num_classes, self.device)
        
        # Verify the criterion was updated
        self.model.criterion.empty_weight.to.assert_called_with(self.device)
        self.assertEqual(self.model.criterion.num_classes, self.num_classes)