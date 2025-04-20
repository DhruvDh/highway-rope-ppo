import numpy as np

def add_positional_embeddings(cars, method="rank", embed_dim=1):
    # cars: shape [N, features]
    x_positions = cars[:, 0]  # assume x is forward distance
    ranks = x_positions.argsort().argsort()  # rank each car

    if method == "rank":
        embeddings = ranks.reshape(-1, 1) / len(cars)  # normalized rank
    elif method == "sinusoidal":
        embeddings = np.array([
            [np.sin(rank / 10000**(i / embed_dim)) for i in range(embed_dim)]
            for rank in ranks
        ])
    else:
        raise ValueError("Unknown method: choose 'rank' or 'sinusoidal'")

    return np.concatenate([cars, embeddings], axis=1)
