import random
import torch

def create_real_pairs(embeddings, labels, num_pairs=5000):

    labels = labels.numpy()

    class_dict = {}

    for idx, cls in enumerate(labels):
        class_dict.setdefault(cls, []).append(idx)

    classes = list(class_dict.keys())

    emb1 = []
    emb2 = []

    idx1_list = []
    idx2_list = []

    targets = []

    # No Change
    for _ in range(num_pairs // 2):

        cls = random.choice(classes)

        idx1, idx2 = random.sample(class_dict[cls], 2)

        emb1.append(embeddings[idx1])
        emb2.append(embeddings[idx2])

        idx1_list.append(idx1)
        idx2_list.append(idx2)

        targets.append(0)

    # Change
    for _ in range(num_pairs // 2):

        c1, c2 = random.sample(classes, 2)

        idx1 = random.choice(class_dict[c1])
        idx2 = random.choice(class_dict[c2])

        emb1.append(embeddings[idx1])
        emb2.append(embeddings[idx2])

        idx1_list.append(idx1)
        idx2_list.append(idx2)

        targets.append(1)

    return (
        torch.stack(emb1),
        torch.stack(emb2),
        torch.tensor(targets),
        idx1_list,
        idx2_list
    )