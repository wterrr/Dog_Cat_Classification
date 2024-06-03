import torchvision
import torch.nn as nn
import io
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, send_from_directory, url_for
import torch
import os

app = Flask(__name__)
app.config['SAMPLES_FOLDER'] = 'samples'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load pre-trained model
class my_Model(torch.nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    def forward(self, image_tensor):
        return self.model(image_tensor)

# Create model instance
model = my_Model('cat_dog_model.pt')

@app.route('/')
def home():
    samples = os.listdir(app.config['SAMPLES_FOLDER'])
    return render_template('index.html', samples=samples)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            upload_folder = 'static/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            img_path = os.path.join(upload_folder, file.filename)
            file.save(img_path)

            tensor = model.transform(Image.open(img_path)).float() / 255.0
            class_result, confidence = predictByImage(tensor)

            if class_result == "Cat":
                result = f"Cat with {confidence}% confidence"
            else:
                result = f"Dog with {100-confidence}% confidence"

            samples = os.listdir('samples')
            return render_template('index.html', result=result, samples=samples, uploaded_image=img_path)

def predictByImage(image_tensor):
    m_image = image_tensor.to("cpu").unsqueeze(0)
    preds = model(m_image)
    confidence = (preds*100).round().item()
    class_result = "Cat" if preds.round().item() == 1.0 else "Dog"
    return class_result, confidence

@app.route('/predict_sample', methods=['POST'])
def predict_sample():
    if request.method == 'POST':
        sample = request.form['sample']
        img_path = os.path.join('./samples', sample)

        image = model.transform(Image.open(img_path)).float() / 255.0
        class_result, confidence = predictByImage(image)

        if class_result == "Cat":
            result = f"Cat with {confidence}% confidence"
        else:
            result = f"Dog with {100-confidence}% confidence"

        samples = os.listdir('./samples')
        return render_template('index.html', result=result, samples=samples, sample_image=img_path)
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/samples/<filename>')
def sample_file(filename):
    return send_from_directory(app.config['SAMPLES_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
