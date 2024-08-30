import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class V2MIDIModel(nn.Module):
    def __init__(self):
        super(V2MIDIModel, self).__init__()
        # Load a pre-trained R3D_18 model, using the recommended approach with weights
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        
        # Remove the last fully connected layer
        self.base_model = nn.Sequential(*list(model.children())[:-1])
        
        # Replace the last layer with a new one that outputs 16 predictions for MIDI
        self.fc = nn.Linear(model.fc.in_features, 16)

    def forward(self, x):
        x = self.base_model(x)
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)  # Using sigmoid since we're dealing with a binary classification problem

    def initialize_weights(self):
        # Initialize weights for the fully connected layer
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)


if __name__ == "__main__":
    model = V2MIDIModel().to(torch.device("cuda"))  # Ensure model is on CUDA
    dummy_video_data = torch.rand(1, 3, 16, 112, 112).to(torch.device("cuda"))  # Also move dummy data to CUDA
    output = model(dummy_video_data)
    print(f"Output shape from dummy data: {output.shape}")