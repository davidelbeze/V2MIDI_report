import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class V2MIDIModel(nn.Module):
    def __init__(self):
        super(V2MIDIModel, self).__init__()
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.base_model = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, 6 * 16)  # 6 classes for each of the 16 frames

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(-1, 16, 6)  # Reshape to (batch, frames, classes)
        return torch.log_softmax(x, dim=2)  # Use log_softmax for numerical stability with NLLLoss

if __name__ == "__main__":
    model = V2MIDIModel().to(torch.device("cuda"))
    dummy_video_data = torch.rand(1, 3, 16, 112, 112).to(torch.device("cuda"))
    output = model(dummy_video_data)
    print(f"Output shape from dummy data: {output.shape}")
