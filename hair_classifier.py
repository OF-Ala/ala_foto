import torchvision.models as models
import torch

class Classifier_model():

    def __init__(self, path_name):
        # все изображения будут масштабированы к размеру 256x256 px
        self.RESCALE_SIZE = 256
        self.save_path = path_name
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 3)
        self.model.load_state_dict(torch.load(self.save_path + 'classification.pt', map_location='cpu'))

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