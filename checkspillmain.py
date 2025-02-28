import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label2cat = {0: "Oil Spill", 1: "No Oil Spill"}

# Load the pretrained MobileNetV2 model and modify it for your output size
class CustomMobilenetV2(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.mnet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.mnet.classifier = nn.Sequential(
            nn.Linear(1280, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.mnet(x)

# Load the model and move it to the appropriate device
model = CustomMobilenetV2(output_size=len(label2cat))  # Replace with your actual number of classes
pretrained_dict = torch.load('weights_best.pth')
model_dict = model.state_dict()

# Filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)

# Load the updated state dict
model.load_state_dict(model_dict)
model.to(device)
model.eval()

# Transformation for the input image
test_transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load and preprocess an image
def image_loader(image_name):
    image = Image.open(image_name)
    image = test_transform(image).float()
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Load and predict on a new image
image_path = r""  # Replace with your actual image path
images = image_loader(image_path)

with torch.no_grad():
    output = model(images)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = probabilities.argmax().item()
    predicted_label = f"There is {label2cat[predicted_class]} with probability {probabilities[predicted_class]:.2f}"

print(predicted_label)