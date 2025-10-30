# beluga_model.py
import torch
from torch import nn
import numpy as np
from Bio import SeqIO
import math
import pickle

print('starting to generate embeddings from Beluga!', flush=True)

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

def load_beluga_model(model_path, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Beluga().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def encodeSeqs(seqs, inputsize=2000):
    """Convert sequences to 0-1 encoding and truncate to the input size.
    The output concatenates the forward and reverse complement sequence
    encodings.

    Args:
        seqs: list of sequences (e.g. produced by fetchSeqs)
        inputsize: the number of basepairs to encode in the output

    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize

    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """
    seqsnp = np.zeros((len(seqs), 4, inputsize), np.bool_)

    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}

    n = 0
    
    for line in seqs:
        #cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        cline=line
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = mydict[c]
        n = n + 1

    # get the complementary sequences
    dataflip = seqsnp[:, ::-1, ::-1]
    seqsnp = np.concatenate([seqsnp, dataflip], axis=0)
    return seqsnp

def sliding_windows(seq, window=2000, step=200):
    return [seq[i:i + window] for i in range(0, len(seq) - window + 1, step)]
 
def get_beluga_embeddings(model, seq_around_tss, batch_size=32):
    device = next(model.parameters()).device
    windows = sliding_windows(seq_around_tss, window=2000, step=200)
    encoded = encodeSeqs(windows, inputsize=2000).astype(np.float32)

    outputs = []
    for i in range(0, encoded.shape[0], batch_size):
        batch = torch.from_numpy(encoded[i:i+batch_size]).unsqueeze(2).to(device)
        with torch.no_grad():
            out = model(batch).cpu().numpy()  # shape: (B, 2002)
        outputs.append(out)

    output_matrix = np.vstack(outputs)  # shape: (400, 2002) -- 400 because positive & negative

    half = output_matrix.shape[0] // 2
    forward = output_matrix[:half]
    reverse = output_matrix[half:]
    average_embedding = (forward + reverse) / 2

    # run exponential function on the beluga raw output
    shifts = np.arange(-19900, 20000, 200) / 200  # 200 shifts, centered around TSS
    #weights = np.array([np.exp(-p * np.abs(shifts)) for p in 
    #                    [0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.02, 0.05, 0.1, 0.2]])  # shape: (10, 200)

    p = 0.2 # apply only one exponential function to keep the dimensionality of the embedding low
    weights = np.exp(-p * np.abs(shifts))
    weights = weights / weights.sum() # normalization step

    # beluga predictions should be shape (200, 2002)
    condensed = np.dot(weights, average_embedding)  # shape: (10, 2002) --> now (1, 2002)

    return torch.from_numpy(condensed).float()