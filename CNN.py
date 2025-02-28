# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import jcopdl
from torch import nn, optim
from jcopdl.callback import Callback, set_config

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0")

bs = 16
crop_size = 128

train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(crop_size, scale = (0.9, 1.0)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(140),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder("train/", transform=train_transform)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)

test_set = datasets.ImageFolder("test/", transform=test_transform)
testloader = DataLoader(test_set, batch_size=bs, shuffle=True)

feature, target = next (iter(trainloader))
feature.shape

label2cat = train_set.classes
label2cat


# %%
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            #Convolutional block 1
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            #Convolutional block 2
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            #Convolutional block 3
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            #Convolutional block 4
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            #Convolutional block 5
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            #Flattening
            nn.Flatten()
        ) 
        
        #Fully connected layer
        self.fc = nn.Sequential(
            #Linear 1024 -> 256
            nn.Linear(2048,256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            #Linear 256 -> 2
            nn.Linear(256, 2),
            nn.LogSoftmax()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
        

# %%
config = set_config({
    "batch_size" : bs,
    "crop_size": crop_size
})

# %%
"""
# Training Preparation --> MCOC
"""

# %%
model = CNN().to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
callback = Callback(model, config, outdir="model", early_stop_patience=  10)

# %%
"""
# Training
"""

# %%
from tqdm.auto import tqdm

def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device):
    if mode == "train":
        model.train()
    elif mode == "test":
        model.eval()
    cost = correct = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)
        
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()
    cost = cost / len(dataset)
    acc = correct / len(dataset)
    return cost, acc

# %%
while True:
    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
    with torch.no_grad():
        test_cost, test_score = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)
    
    # Logging
    callback.log(train_cost, test_cost, train_score, test_score)

    # Checkpoint
    callback.save_checkpoint()
        
    # Runtime Plotting
    callback.cost_runtime_plotting()
    callback.score_runtime_plotting()
    
    # Early Stopping
    if callback.early_stopping(model, monitor="test_score"):
        callback.plot_cost()
        callback.plot_score()
        break

# %%
feature, target = next(iter(testloader))
feature, target = feature.to(device), target.to(device)

# %%
with torch.no_grad():
    model.eval()
    output = model (feature)
    preds = output

# %%
"""
# Probability and classification
"""

preds = preds.cpu().numpy()
np.exp(preds), preds.argmax(1)

# %%
max(np.exp(preds[0]))

# %%
"""
# Sanity check
"""

# %%

fig, axes = plt.subplots(4,4, figsize = (24,24))

for img, label, pred, ax in zip (feature, target, preds, axes.flatten()):
    ax.imshow(img.permute(1,2,0).cpu())
    font = {"color": 'r'} if label != pred.argmax(0) else {"color":'g'}
    label, pred, prob = label2cat [label.item()], label2cat[pred.argmax(0).item()], "{:.2f}".format(max(np.exp(pred)).item())
    ax.set_title(f"Label: {label} | Pred: {pred} | Prob: {prob}", fontdict= font)
    ax.axis("off")

# %%
fig.savefig("oilspill.png")

# %%
"""
# new image
- https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411/2
"""

# %%
crop_size = 128
loader = transforms.Compose([
            transforms.Resize(70),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()])

# %%
from PIL import Image
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

# %%
images = image_loader("test.jpeg")
images.shape

# %%
with torch.no_grad():
    model.eval()
    output = model (images)
    print(f"there is %s with probability %s" %(label2cat[output.argmax(1).item()], "{:.2f}".format(max(np.exp(output)[0]))))