import torch
import torch.nn.functional as F


def cosine_similarity(emb1, emb2):
    """
    Cosine similarity between two embedding vectors.
    """

    return F.cosine_similarity(
        emb1,
        emb2,
        dim=1
    )


def euclidean_distance(emb1, emb2):

    return torch.norm(
        emb1 - emb2,
        dim=1
    )