import torch


def extract_embeddings(model, loader, device):
    """
    Extract embeddings and labels from a dataloader.

    Returns:
        embeddings : Tensor (N,512)
        labels     : Tensor (N,)
    """

    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():

        for images, target in loader:

            images = images.to(device)

            features = model.extract_features(images)

            embeddings.append(features.cpu())

            labels.append(target)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    return embeddings, labels