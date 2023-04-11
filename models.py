import torch
import torch.nn as nn
import torchvision.models as models


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(1000, 512*7*7)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 7, 7)
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = torch.sigmoid(self.conv6(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        #print(encoded.shape)
        #print(f"encoded shape = {encoded.view(encoded.shape[0],-1).shape}")
        decoded = self.decoder(encoded)
        return decoded

# Load pretrained ResNet50 model
encoder = models.resnet50(pretrained=True)
decoder = ConvDecoder()

def get_baseline_clf(finetuned = True):
    # UC Merced 90% accuracy 40 epochs

    baseline_clf = models.resnet50(pretrained = True)
    baseline_clf.fc = nn.Sequential(nn.Linear(2048,3),)
    for name, param in baseline_clf.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    if finetuned:
        baseline_clf.load_state_dict(torch.load('baseline_clf_ucmerced.pth'))
    return baseline_clf

# Create autoencoder with ResNet50 encoder and decoder
autoencoder = Autoencoder(encoder, decoder)

def evaluate_model(model,data_loader,device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    model.eval()
    model.to(device)

    with torch.no_grad():
        for data in data_loader:
            images, labels, _ = data
            images,labels = images.to(device),labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the validation images: {100 * correct // total} %')
    return accuracy
