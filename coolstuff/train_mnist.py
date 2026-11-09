"""
Train a single layer MLP with MNIST, 
"""
import argparse
import gzip
import os
import time
import urllib.request
import nvtx

import numpy as np
from photon import Tensor
from photon.backend_dispatcher import get_backend
try:
    rng = lambda name, color: nvtx.annotate(name, color=color)
except ImportError:
    rng = lambda name, color: contextlib.nullcontext()

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}


def load_mnist(data_dir="data/mnist"):
    os.makedirs(data_dir, exist_ok=True)
    out = {}
    for key, fname in FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(MIRROR + fname, path)
        with gzip.open(path, "rb") as f:
            raw = f.read()
        if key.endswith("_x"):
            out[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784)
        else:                               
            out[key] = np.frombuffer(raw, np.uint8, offset=8)
    return out


def to_np(t):
    return np.array(t._data) if t._device == "cpu" else t._data.numpy()


def main():
    # NN args for configurable teesting 
    ap = argparse.ArgumentParser()
    ap.add_argument("device", nargs = "?", default = "cuda")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-limit", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()
    dev, B = args.device, args.batch
    backend = get_backend(dev)

    d = load_mnist()
    norm = lambda x: ((x.astype(np.float32) / 255.0 - 0.131) / 0.308) # normalises mnist
    n_train = (min(args.train_limit, 60000) // B) * B

    Xtr, ytr = norm(d["train_x"][:n_train]), d["train_y"][:n_train]
    Xte, yte = norm(d["test_x"]), d["test_y"]
    Ytr_onehot = np.eye(10, dtype=np.float32)[ytr]
    
    # annotation for profiler
    with nvtx.annotate("MNIST dataset upload", color = "orange"):
        X_all = backend.NDArray(Xtr.ravel(), list(Xtr.shape))
        Y_all = backend.NDArray(Ytr_onehot.ravel(), list(Ytr_onehot.shape))

    np.random.seed(42)
    
    # Init parameters 
    def param(*shape, scale):
        return Tensor((np.random.randn(*shape) * scale).astype(np.float32), device=dev)

    # Weights scaled back down for std = 1 
    W1, b1 = param(784, args.hidden, scale=784 ** -0.5), param(args.hidden, scale=0.0)
    W2, b2 = param(args.hidden, 10, scale=args.hidden ** -0.5), param(10, scale=0.0)
    params = [W1, b1, W2, b2]


    # TODO: build out nn library
    def forward(x):
        h = (x @ W1 + b1).tanh()
        return h @ W2 + b2                                 

    def cross_entropy(logits, y_onehot):
        """crossentropy via logsumexp 

        NB: we don't have comparison kernel yet so the max for c is kinda a hack to prevent having to backprop
        """
        
        # TODO: implement max op on tensors directly
        c = Tensor._from_ndarray(logits._data.max([1], True), device=dev, requires_grad=False) 
        zs = logits - c
        lse = zs.exp().sum(axes=[1], keepdims=True).log() 
        picked = (zs * y_onehot).sum(axes=[1], keepdims=True)  
        return (lse - picked).sum() / float(logits.shape[0])

    def evaluate():
        correct = 0
        for i in range(0, len(Xte), 1000):
            xb = Tensor(Xte[i:i + 1000], device=dev, requires_grad=False)
            correct += int((to_np(forward(xb)).argmax(1) == yte[i:i + 1000]).sum())
        return correct / len(Xte)

    steps_per_epoch = n_train // B
    print(f"device={dev} train={n_train} hidden={args.hidden} batch={B} "
          f"steps/epoch={steps_per_epoch}")
    for epoch in range(args.epochs):
        t0, running = time.time(), 0.0
        with nvtx.annotate(f"epoch {epoch}", color="purple"):
            for s in range(steps_per_epoch):
                with nvtx.annotate("step", color="white"):
                    # TODO: Better api for detaching from compgraph
                    xb = Tensor._from_ndarray(X_all[s * B:(s + 1) * B], device=dev, requires_grad=False)
                    yb = Tensor._from_ndarray(Y_all[s * B:(s + 1) * B], device=dev, requires_grad=False)
                    # TODO: zero grad api
                    for p in params:
                        p.grad = None
                    with nvtx.annotate("forward", color="blue"):
                        loss = cross_entropy(forward(xb), yb)
                    with nvtx.annotate("backward",color="red"):
                        loss.backward()
                    with nvtx.annotate("sgd", color="green"):
                        for p in params:
                            p._data = p._data - args.lr * p.grad._data
                if s % 50 == 0 or s == steps_per_epoch - 1: 
                    running = float(to_np(loss)[0])
                    print(f"  epoch {epoch} step {s:4d}  loss {running:.4f}")
        with nvtx.annotate("eval", color="yellow"):
            acc = evaluate()
        print(f"epoch {epoch}: {time.time() - t0:.3f}s  test acc {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
