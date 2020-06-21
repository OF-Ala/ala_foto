import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, n_classes):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(8)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 96, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(96)
        )

        self.dense1 = nn.Sequential(
            nn.Linear(96 * 5 * 5, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p = 0.3)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p = 0.3)
        )
        self.out = nn.Linear(128, n_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = self.out(x)
        return logits
    
class Classifier_model():

    def __init__(self, path_name):
        # все изображения будут масштабированы к размеру 256x256 px
        self.RESCALE_SIZE = 256
        self.save_path = path_name
        #self.model = models.resnet18(pretrained=False)
        #self.model.fc = nn.Linear(512, 3)
        #self.model.load_state_dict(torch.load(self.save_path + 'classification.pt', map_location='cpu'))
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(self.save_path + 'classification-simple.pt', map_location='cpu'))

        self.model.eval()

    def img_to_tensor(self, file_name):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(self.save_path + file_name)
        image.load()

        image = image.resize((self.RESCALE_SIZE, self.RESCALE_SIZE))

        image = np.array(image)

        image = np.array(image / 255, dtype='float32')
        A = transform(image)

        return A

    def predict_one_sample(self, img_name):
        """Предсказание, для одной картинки"""
        img = self.img_to_tensor(img_name)
        img = img.unsqueeze(0)
        with torch.no_grad():
            self.model.eval()
            logit = self.model(img)
            probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
        return  np.argmax(probs), probs[0][np.argmax(probs)]
