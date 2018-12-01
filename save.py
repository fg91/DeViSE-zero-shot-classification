import pickle
from flask import Flask, request, send_file
from flasgger import Swagger
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import nmslib
import io

app = Flask(__name__)
swagger = Swagger(app)

tfms = transforms.Compose([
    torchvision.transforms.Scale(224, interpolation=2),
    torchvision.transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

@app.route('/zero_shot_prediction_image', methods=["POST"])
def zero_shot_prediction_image():
    """Returns the five most likely categories for an input image based on the DeViSE model that makes semantically relevant predictions [...] and generalizes to classes outside of its labeled training set, i.e. zero-shot learning (Frome et al. 2013)
    ---
    parameters:
    - name: input_image
      in: formData
      type: file
      required: true
    responses:
        200:
            description: "image"
    """
    img = Image.open(request.files.get("input_image"))
    transformed_img = tfms(img)
    pred = model(transformed_img.unsqueeze(0))
    nn_indcs, _ = get_knns(nearest_neighbour_index, pred.data.numpy())
    return ' '.join(o for o in classes[nn_indcs])

# def serve_pil_image(pil_img):
#     img_io = StringIO()
#     pil_img.save(img_io, 'JPEG', quality=70)
#     img_io.seek(0)
#     return send_file(img_io, mimetype='image/jpeg')

@app.route('/search_string_to_image', methods=["POST"])
def search_string_to_image():
    """Returns the five most likely categories for an input image based on the DeViSE model that makes semantically relevant predictions [...] and generalizes to classes outside of its labeled training set, i.e. zero-shot learning (Frome et al. 2013)
    ---
    parameters:
    - name: input_image
      in: formData
      type: file
      required: true
    responses:
        200:
            description: "image"
    """
    img = Image.open(request.files.get("input_image"))

    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='JPEG')
    
    
    return send_file(io.BytesIO(imgByteArr.getvalue()),
              attachment_filename='return.jpeg',
              mimetype='image/jpeg')


def create_index(a):
    index = nmslib.init(space='angulardist')
    index.addDataPointBatch(a)
    index.createIndex()
    return index

def get_knns(index, vecs):
    return zip(*index.knnQueryBatch(vecs, k=5, num_threads=1))

if __name__ == "__main__":
    # Load classifier
    # device = torch.device("cpu")
    # model = torch.load('model.pth')
    # model.to(device)
    # model.eval()

    # # Load word vectors
    # classId2wordvec = pickle.load(open('classId2wordvec.pkl', 'rb'))
    # classes, wordvecs = zip(*classId2wordvec)
    # classes = np.array(classes)

    # # Create nearest neighbour index
    # nearest_neighbour_index = create_index(wordvecs)
    
    app.run(host='0.0.0.0', port=5000)
