import numpy as np

# Compute squared Euclidean distance between data points and codebook

def _distance_matrix(data, codebook):
    diff = data[:, None, :] - codebook[None, :, :]
    return np.sum(diff * diff, axis=2)


# Update codebook means based on cluster assignments

def _update_codebook(data, labels, codebook):
    new_codebook = np.copy(codebook)
    counts = np.bincount(labels, minlength=len(codebook))
    for i in range(len(codebook)):
        if counts[i] > 0:
            new_codebook[i] = np.mean(data[labels == i], axis=0)
        else:
            # Reuse the vector with the largest distortion and add small noise
            farthest = np.argmax(np.sum((data - codebook[i]) ** 2, axis=1))
            new_codebook[i] = data[farthest] + 1e-3 * np.random.randn(data.shape[1])
    return new_codebook


# Lloyd iteration to refine the codebook

def _lloyd_iterations(data, codebook, max_iter, tol):
    prev_distortion = np.inf
    for _ in range(max_iter):
        distances = _distance_matrix(data, codebook)
        labels = np.argmin(distances, axis=1)
        distortion = np.mean(distances[np.arange(len(data)), labels])
        codebook = _update_codebook(data, labels, codebook)
        if abs(prev_distortion - distortion) / max(distortion, 1e-12) < tol:
            break
        prev_distortion = distortion
    return codebook


# Linde-Buzo-Gray vector quantizer training

def lbg_vq(data, codebook_size, epsilon=0.01, max_iter=50, tol=1e-4):
    if codebook_size < 1:
        raise ValueError("codebook_size must be >= 1")
    vectors = np.asarray(data, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("data must be a 2D array (num_vectors x dimension)")
    num_vectors, dimension = vectors.shape
    if num_vectors == 0:
        raise ValueError("data must contain at least one vector")

    codebook = [np.mean(vectors, axis=0)]

    while len(codebook) < codebook_size:
        current_size = len(codebook)
        if current_size * 2 <= codebook_size:
            expanded = []
            for vector in codebook:
                expanded.append(vector * (1.0 + epsilon))
                expanded.append(vector * (1.0 - epsilon))
            codebook = np.vstack(expanded)
        else:
            needed = codebook_size - current_size
            split_vectors = codebook[:needed]
            keep_vectors = codebook[needed:]
            left = split_vectors * (1.0 + epsilon)
            right = split_vectors * (1.0 - epsilon)
            codebook = np.vstack((left, right, keep_vectors))

        codebook = _lloyd_iterations(vectors, codebook, max_iter, tol)

    if len(codebook) > codebook_size:
        codebook = codebook[:codebook_size]

    return codebook


# Assign each vector to the nearest codeword

def vq_encode(data, codebook):
    vectors = np.asarray(data, dtype=np.float64)
    codebook = np.asarray(codebook, dtype=np.float64)
    distances = _distance_matrix(vectors, codebook)
    labels = np.argmin(distances, axis=1)
    return labels, distances[np.arange(len(vectors)), labels]


# Decode indices back to codewords

def vq_decode(labels, codebook):
    codebook = np.asarray(codebook, dtype=np.float64)
    return codebook[np.asarray(labels, dtype=np.int32)]
