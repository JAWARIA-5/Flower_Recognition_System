import numpy as np
import torch
from PIL import Image
def process_image(image):
    im = Image.open(image)
    width, height = im.size 
    if width > height: 
        height = 256
        im.thumbnail((50000, height), Image.LANCZOS)
    else: 
        width = 256
        im.thumbnail((width, 50000), Image.LANCZOS)
 
    width, height = im.size 
    reduce = 224
    left = (width - reduce) / 2 
    top = (height - reduce) / 2
    right = left + 224 
    bottom = top + 224
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im) / 255  # to make values from 0 to 1
    np_image -= np.array([0.485, 0.456, 0.406]) 
    np_image /= np.array([0.229, 0.224, 0.225])
    
    np_image = np_image.transpose((2, 0, 1))
    return np_image


def predict(image_file, model, topk):
    image = process_image(image_file)
    tensor_image = torch.from_numpy(image).type(torch.FloatTensor)
    tensor_image = tensor_image.unsqueeze(dim=0)
    with torch.no_grad():
        output = model.forward(tensor_image)
    output_prob = torch.exp(output)
    top_probs, top_indices = output_prob.topk(topk)
    top_probs = top_probs.numpy()[0]
    top_indices = top_indices.numpy()[0]
    mapping = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [mapping[item] for item in top_indices]
    
    return top_probs, top_classes