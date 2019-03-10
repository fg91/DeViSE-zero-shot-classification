import matplotlib
matplotlib.use("tkagg")  # so that import pyplot does not try to pull in a GUI
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from flask import Flask, request, send_file
from flasgger import Swagger
import numpy as np
from PIL import Image
from annoy import AnnoyIndex
import io
import os.path
from torchvision.models import resnet34

app = Flask(__name__)
app.config['SWAGGER'] = {
    "swagger_version": "2.0",
    "title": "DeViSE",
    "description": "Zero-shot predictions based on the DeViSE - Deep Visual-Semantic Embedding Model by Fromme et al. (2013)"}

swagger = Swagger(app)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def create_index(vecs, dim = 300):
    annoy_index = AnnoyIndex(dim)
    for i, wordvec in enumerate(vecs):
        annoy_index.add_item(i, wordvec)
    annoy_index.build(10)
    return annoy_index

tfms = transforms.Compose([
    torchvision.transforms.Resize(224, interpolation=2),
    torchvision.transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tfms2 = transforms.Compose([
    torchvision.transforms.Resize(500, interpolation=2),
    torchvision.transforms.CenterCrop(500),
    transforms.ToTensor(),
])

data = torchvision.datasets.ImageFolder(root='pictures/', transform=tfms)
data_not_normalized = torchvision.datasets.ImageFolder(root='pictures/', transform=tfms2)

data_loader = torch.utils.data.DataLoader(data, batch_size=10)

# Load classifier
model = torch.load('devise_trained_full_imagenet.pth', map_location='cpu')
#model.to(device)
model.eval()

# Load WordNet wordvectors
classId2wordvec = pickle.load(open('className_2_wordvec_without_dups.pkl', 'rb'))
classes, wordvecs = zip(*classId2wordvec)
classId2wordvec_dict = dict([(c,vw) for c, vw in zip(classes, wordvecs)])
classes = np.array(classes)

# Create nearest neighbour index for WordNet vectors
WordNet_index = create_index(wordvecs)

# Predict the wordvectors for the test dataset
fname = 'predicted_wordvecs_testdata.pkl'
if not os.path.isfile(fname):
    predictions = np.ndarray(shape=(len(data), 300))
    num_batches = len(data_loader)
    bs = data_loader.batch_size
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            print(i)
#            x = x.to(device)
            preds = model(x)
            
            start = i * bs
            end = start + bs if i != num_batches - 1 else len(data_loader.dataset)
            
            predictions[start:end] = preds
    predictions = np.split(predictions, len(data_loader.dataset), axis=0)
    predictions = [np.squeeze(pred, 0) for pred in predictions]
    try:
        pickle.dump(predictions, open(fname, 'wb'))
    except:
        print("Could not save predicted wordvectors.")
else:
    predictions = pickle.load(open(fname, 'rb'))

# Create nearest neighbour index for the predicted wordvectors (test dataset)
predicted_index = create_index(predictions)

def wordvec_to_images(wordvec, index):
    indcs = index.get_nns_by_vector(wordvec, 4)
    grid = torchvision.utils.make_grid(torch.tensor(np.stack([data_not_normalized[i][0] for i in indcs])))
    return Image.fromarray(np.uint8(grid.numpy()*255).transpose(1,2,0))

@app.route('/wordvec_2_image')
def wordvec_to_image():
    """Returns the four images from the stored dataset that are semantically most similar to the chosen word (or words). If two words are chosen, their respective word vectors are averaged.
    ---
    tags:
    - DeViSE
    parameters:                                                      
    - name: category_1
      in: query                                                   
      type: string                                                     
      required: true
    - name: category_2
      in: query                                                   
      type: string                                                     
    responses:                                                       
        200:                                                         
            description: "string"                                     
    """
    category_1 = request.args.get("category_1")
    category_2 = request.args.get("category_2")
    if category_2 is None:
        try:
            wordvec = classId2wordvec_dict[category_1]
        except:
            return "No word vector found for this input. Please try another input."
    else:
        try:
            wordvec = (classId2wordvec_dict[category_1] + classId2wordvec_dict[category_2]) / 2;
        except:
            return "No word vector found for one of the two chosen inputs. Please try other inputs."
    grid = wordvec_to_images(wordvec, predicted_index);
    imgByteArr = io.BytesIO()
    grid.save(imgByteArr, format='JPEG')

    return send_file(io.BytesIO(imgByteArr.getvalue()),
                     attachment_filename='return.jpeg',
                     mimetype='image/jpeg')

@app.route('/zero_shot_prediction_images', methods=["POST"])
def zero_shot_prediction_images():
    """Returns the four images from the stored dataset that are semantically most similar to the input image.
    ---
    tags:
    - DeViSE
    parameters:                                                      
    - name: input_image                                              
      in: formData                                                   
      type: file                                                     
      required: true                                                 
    responses:                                                       
        200:                                                         
            description: "image"                                     
    """
    try:
        img = Image.open(request.files.get("input_image"))
    except:
        return "Could not open chosen image. Please choose a JPEG file."
    try:
        transformed_img = tfms(img)
        pred = model(transformed_img.unsqueeze(0))
        grid = wordvec_to_images(pred[0], predicted_index);
        imgByteArr = io.BytesIO()
        grid.save(imgByteArr, format='JPEG')
    except:
        return "Prediction unsuccessful. Please choose a JPEG file."

    return send_file(io.BytesIO(imgByteArr.getvalue()),
                     attachment_filename='return.jpeg',
                     mimetype='image/jpeg')

@app.route('/zero_shot_prediction_categories', methods=["POST"])
def zero_shot_prediction_categories():
    """ Returns the top five zero-shot predictions for the input image.
    ---
    tags:
    - DeViSE
    parameters:                                                      
    - name: input_image                                              
      in: formData                                                   
      type: file                                                     
      required: true                                                 
    responses:                                                       
        200:                                                         
            description: "image"                                     
    """
    try:
        img = Image.open(request.files.get("input_image"))
    except:
        return "Could not open chosen image. Please choose a JPEG file."
    try:
        transformed_img = tfms(img)
        pred = model(transformed_img.unsqueeze(0))
        indcs = WordNet_index.get_nns_by_vector(pred[0], 5)
        categories = classes[indcs]
    except:
        return "Prediction unsuccessful. Please choose a JPEG file."

    return ' '.join([c for c in categories])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
