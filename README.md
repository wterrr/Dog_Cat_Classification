# Cat and Dog Classification

This project allows you to classify images as either cats or dogs using a pre-trained ResNet18 model.

## Installation

### Clone the repository

```bash
git clone https://github.com/wterrr/DCC.git
cd DCC
```

### Install virtualenv if you haven't already
```bash
pip install virtualenv
```

### Create a virtual environment
```bahs
virtualenv venv
```

#### Activate the virtual environment
##### Windows
```bash
venv\Scripts\activate
```
##### MacOS/Linux
```bash
source venv/bin/activate
```

#### Install dependencies
```bash
pip install -r requirements.txt
```
## Usage
1. Run the Flask web application:
   ``` bash
   python app.py
2. Open a web browser and go to `http://192.168.1.6:80` to use the application.
3. Upload an image of a cat or a dog or use sample image to classify it.

## If you'd like to contribute to this project, please follow these steps:
1. Fork the repository on GitHub.
2. Create a new branch with a descriptive name (`git checkout -b feature/add-new-feature`).
3. Make your changes.
4. Commit your changes (git commit -am 'Add new feature').
5. Push to the branch (git push origin feature/add-new-feature).
6. Create a new pull request on GitHub.
