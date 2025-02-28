from multiprocessing import freeze_support
import jcopdl
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from jcopdl.callback import Callback, set_config
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from jcopdl.utils.dataloader import MultilabelDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bs = 16
crop_size = 224

train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(crop_size, scale = (0.9, 1.0)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder("train/", transform=train_transform)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)

test_set = datasets.ImageFolder("test/", transform=test_transform)
testloader = DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=0)


label2cat = train_set.classes
label2cat

from torchvision.models import mobilenet_v2


mnet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

for param in mnet.parameters():
    param.requires_grad = False

mnet.classifier = nn.Sequential(
    nn.Linear(1280, 5),
    nn.LogSoftmax()
)

class CustomMobilenetV2(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.mnet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.freeze()
        self.mnet.classifier = nn.Sequential(
            nn.Linear(1280, output_size),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.mnet(x)

    def freeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = True

config = set_config({
    "output_size" : len(train_set.classes),
    "batch_size" : bs,
    "crop_size" : crop_size
})


model = CustomMobilenetV2(config.output_size).to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
callback = Callback(model, config, early_stop_patience = 5, outdir="model")


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

feature, target = next(iter(testloader))
feature, target = feature.to(device), target.to(device)

with torch.no_grad():
    model.eval()
    output = model (feature)
    preds = output

preds = preds.cpu().numpy()
np.exp(preds), preds.argmax(1)

max(np.exp(preds[0]))

fig, axes = plt.subplots(4,4, figsize = (24,24))

for img, label, pred, ax in zip (feature, target, preds, axes.flatten()):
    ax.imshow(img.permute(1,2,0).cpu())
    font = {"color": 'r'} if label != pred.argmax(0) else {"color":'g'}
    label, pred, prob = label2cat [label.item()], label2cat[pred.argmax(0).item()], "{:.2f}".format(max(np.exp(pred)).item())
    ax.set_title(f"Label: {label} | Pred: {pred} | Prob: {prob}", fontdict= font)
    ax.axis("off")

fig.savefig("oilspill.jpg")

from PIL import Image
def image_loader(image_name):
    image = Image.open(image_name)
    image = test_transform(image).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image

images = image_loader("test.jpeg")
images = images.to(device)

with torch.no_grad():
    model.eval()
    output = model (images)
    print(f"there is %s with probability %s" %(label2cat[output.argmax(1).item()], "{:.2f}".format(max(np.exp(output)[0]))))

model.unfreeze()
optimizer = optim.AdamW (model.parameters(), lr= 1e-5)

#mengatur ulang early stop patience
callback.reset_early_stop()
callback.early_stop_patience = 5

if __name__ == '__main__':
    freeze_support()
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

    while True:
        train_cost = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
        with torch.no_grad():
            test_cost = loop_fn("test", test_set, testloader, model, criterion, optimizer, device)

        # Logging
        callback.log(train_cost, test_cost)

        # Checkpoint
        callback.save_checkpoint()

        # Runtime Plotting
        callback.cost_runtime_plotting()

        # Early Stopping
        if callback.early_stopping(model, monitor="test_cost"):
            callback.plot_cost()
            break


feature, target = next (iter(testloader))
feature, target = feature.to(device), target.to(device)

with torch.no_grad():
    model.eval()
    output = model(feature)
    preds = (output > 0.5 ).to(torch.float32)

def convert_to_label(x):
    return [label for pred, label in zip (x, label2cat) if pred == 1]
def inverse_norm(img):
    img[0,:,:] = img[0,:,:] * 0.229 + 0.485
    img[1,:,:] = img[1,:,:] * 0.224 + 0.456
    img[2,:,:] = img[2,:,:] * 0.225 + 0.406
    return img

fig, axes = plt.subplots(6,6, figsize = (24,24))

for img, label, pred, ax in zip (feature, target, preds, axes.flatten()):
    ax.imshow(inverse_norm(img).permute(1,2,0).cpu())
    font = {"color": 'r'} if (pred != label).any() else {"color":'g'}
    label, pred = convert_to_label(label), convert_to_label(pred)
    ax.set_title(f"Label: {label} | Pred: {pred}", fontdict= font)
    ax.axis("off")