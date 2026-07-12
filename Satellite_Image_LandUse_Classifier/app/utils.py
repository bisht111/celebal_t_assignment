import torch.nn.functional as F
import torch.nn.functional as F


THRESHOLD = 0.6384


def similarity(emb1, emb2):

    score = F.cosine_similarity(
        emb1,
        emb2
    )

    return score.item()


def detect(score):

    return score < THRESHOLD

import numpy as np
import cv2


def create_heatmap(img1, img2):

    img1 = np.array(img1.resize((224,224)))

    img2 = np.array(img2.resize((224,224)))

    diff = cv2.absdiff(img1,img2)

    gray = cv2.cvtColor(
        diff,
        cv2.COLOR_RGB2GRAY
    )

    heat = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_JET
    )

    return heat

import torch.nn.functional as F


def similarity(emb1, emb2):

    score = F.cosine_similarity(
        emb1,
        emb2
    )

    return score.item()