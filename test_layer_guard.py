import unittest
import os
import torch
import torch.nn as nn
from Layer_guard import LayerGuardEnhanced

class TestLayerGuard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create the dummy model for testing if it doesn't exist.
        """
        cls.model_path = "dummy_model.pth"
        if not os.path.exists(cls.model_path):
            # Define a simple model
            class DummyModel(nn.Module):
                def __init__(self):
                    super(DummyModel, self).__init__()
                    self.linear1 = nn.Linear(10, 5)
                    self.linear2 = nn.Linear(5, 2)

                def forward(self, x):
                    x = torch.relu(self.linear1(x))
                    x = self.linear2(x)
                    return x

            # Instantiate the model
            model = DummyModel()

            # Save the model's state_dict
            torch.save(model.state_dict(), cls.model_path)

    def test_initialization(self):
        """
        Test that LayerGuardEnhanced can be initialized with a model path.
        """
        self.assertTrue(os.path.exists(self.model_path), f"Dummy model not found at {self.model_path}")
        
        try:
            detector = LayerGuardEnhanced(model_path=self.model_path)
            self.assertIsInstance(detector, LayerGuardEnhanced)
        except Exception as e:
            self.fail(f"LayerGuardEnhanced initialization failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
