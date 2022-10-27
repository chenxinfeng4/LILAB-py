import numpy as np

def create_randsparse(dense_tensor, missing_rate, nside):
    [nseq, nbody, ndim] =  dense_tensor.shape
    nseqround = int((nseq // nside) * nside)
    nseqslice = nseqround // nside
    random_tensor = np.zeros((nseq, nbody, ndim))
    random_tensor_slice = np.random.rand(nseqslice, nbody) < missing_rate
    for i in range(nside):
        random_tensor[i:nseqround:nside,:,:] = random_tensor_slice[:,:,np.newaxis]

    binary_tensor = random_tensor > 0
    return binary_tensor