"""import torch
import torch.nn.functional as F

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


def predict(model, image):

    x = transform(image).unsqueeze(0)

    x = x.to(next(model.parameters()).device)

    with torch.no_grad():

        output = model(x)

        prob = F.softmax(output, dim=1)

        conf, pred = prob.max(dim=1)

        embedding = model.extract_features(x)

    return (
        pred.item(),
        conf.item(),
        embedding.cpu()
    )"""

import torch
import torch.nn.functional as F

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


def predict(model, image):

    x = transform(image).unsqueeze(0)

    x = x.to(next(model.parameters()).device)

    with torch.no_grad():

        output = model(x)

        probs = F.softmax(output, dim=1)

        conf, pred = probs.max(dim=1)

        embedding = model.extract_features(x)

    return (
        pred.item(),
        conf.item(),
        embedding.cpu()
    )