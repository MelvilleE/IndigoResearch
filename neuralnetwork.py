from typing import List, Callable, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
import os
import argparse
import time



# -----------------------------
# Activation functions & utils
# -----------------------------
class Activation:
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    @staticmethod
    def relu_deriv(z):
        return (z > 0).astype(z.dtype)
    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    @staticmethod
    def leaky_relu_deriv(z, alpha=0.01):
        return np.where(z > 0, 1.0, alpha)
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    @staticmethod
    def sigmoid_deriv(z):
        s = Activation.sigmoid(z)
        return s * (1 - s)
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    @staticmethod
    def tanh_deriv(z):
        return 1 - np.tanh(z) ** 2
    @staticmethod
    def linear(z):
        return z
    @staticmethod
    def linear_deriv(z):
        return np.ones_like(z)
    @staticmethod
    def softmax(z):
        # numerically stable softmax along last axis
        z_max = np.max(z, axis=1, keepdims=True)
        exp = np.exp(z - z_max)
        return exp / np.sum(exp, axis=1, keepdims=True)

# mapping names to functions
_ACTS = {
    "relu": (Activation.relu, Activation.relu_deriv),
    "leaky_relu": (lambda z: Activation.leaky_relu(z, 0.01), lambda z: Activation.leaky_relu_deriv(z, 0.01)),
    "sigmoid": (Activation.sigmoid, Activation.sigmoid_deriv),
    "tanh": (Activation.tanh, Activation.tanh_deriv),
    "linear": (Activation.linear, Activation.linear_deriv),
    "softmax": (Activation.softmax, None),  # derivative handled in loss
}

# -----------------------------
# Layer & Neuron abstractions
# -----------------------------
class Neuron:
    def __init__(self, weights_row: np.ndarray, bias_scalar: np.ndarray, index: int):
        self._w = weights_row  
        self._b = bias_scalar  
        self.index = index

    @property
    def weights(self) -> np.ndarray:
        return self._w

    @weights.setter
    def weights(self, v: Sequence[float]):
        self._w[:] = v

    @property
    def bias(self) -> float:
        return float(self._b)

    @bias.setter
    def bias(self, v: float):
        self._b[...] = v

class Layer:

    def __init__(self, n_inputs: int, n_units: int, activation: str = "relu", name: Optional[str] = None, seed: Optional[int] = None):
        rng = np.random.RandomState(seed)
        # He init for relu variants, Xavier for others
        if activation in ("relu", "leaky_relu"):
            std = np.sqrt(2.0 / max(1, n_inputs))
        else:
            std = np.sqrt(2.0 / (n_inputs + n_units))
        self.W = rng.randn(n_units, n_inputs) * std  
        self.b = np.zeros(n_units)
        self.activation = activation
        self.name = name or f"Layer_{n_units}"
        
        self._last_z = None
        self._last_a = None

    def neurons(self) -> List[Neuron]:
        return [Neuron(self.W[i], self.b[i:i+1], i) for i in range(self.W.shape[0])]

    def forward(self, X: np.ndarray) -> np.ndarray:
        
        z = X.dot(self.W.T) + self.b
        self._last_z = z
        act_fn = _ACTS[self.activation][0]
        a = act_fn(z)
        self._last_a = a
        return a

    def set_weights(self, W: np.ndarray, b: np.ndarray):
        assert W.shape == self.W.shape
        assert b.shape == self.b.shape
        self.W[:] = W
        self.b[:] = b


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(0.5 * (y_pred - y_true) ** 2))

def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    
    m = y_true.shape[0]
    return (y_pred - y_true) / m

def cross_entropy_loss_softmax(y_pred: np.ndarray, y_true: np.ndarray) -> float:

    eps = 1e-12
    m = y_true.shape[0]
    clipped = np.clip(y_pred, eps, 1 - eps)
    return float(-np.sum(y_true * np.log(clipped)) / m)

def cross_entropy_grad_softmax(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    
    m = y_true.shape[0]
    return (y_pred - y_true) / m


class NeuralNetwork:
    def __init__(self, seed: Optional[int] = None):
        self.layers: List[Layer] = []
        self.seed = seed

    def add_layer(self, n_units: int, activation: str = "relu", input_dim: Optional[int] = None, name: Optional[str] = None):
        if not self.layers and input_dim is None:
            raise ValueError("First layer must specify input_dim")
        n_inputs = input_dim if not self.layers else self.layers[-1].W.shape[0]
        layer = Layer(n_inputs, n_units, activation=activation, name=name, seed=self._seed_for_layer())
        self.layers.append(layer)

    def _seed_for_layer(self):
        if self.seed is None:
            return None
        
        return (self.seed + len(self.layers)) & 0xFFFFFFFF

    def forward(self, X: np.ndarray) -> np.ndarray:
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, loss: str = "mse") -> float:
        if loss == "mse":
            return mse_loss(y_pred, y_true)
        elif loss == "cross_entropy":
            return cross_entropy_loss_softmax(y_pred, y_true)
        else:
            raise ValueError("Unsupported loss")

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01, batch_size: int = 32, loss: str = "mse", verbose: bool = True):
        n_samples = X.shape[0]
        history = []
        for epoch in range(1, epochs + 1):
            
            perm = np.random.permutation(n_samples)
            Xs = X[perm]
            ys = y[perm]
            for i in range(0, n_samples, batch_size):
                xb = Xs[i:i+batch_size]
                yb = ys[i:i+batch_size]
                # forward
                activations = [xb]
                for layer in self.layers:
                    activations.append(layer.forward(activations[-1]))
                y_pred = activations[-1]
                # gradient
                if loss == "mse":
                    delta = mse_grad(y_pred, yb)  # dL/dA_last
                    
                    last_layer = self.layers[-1]
                    if last_layer.activation == "softmax":
                        
                        deriv_fn = _ACTS[last_layer.activation][1]
                        if deriv_fn is None:
                            raise ValueError("Softmax derivative not provided for MSE scenario")
                        delta = delta * deriv_fn(last_layer._last_z)
                    else:
                        deriv_fn = _ACTS[last_layer.activation][1]
                        delta = delta * deriv_fn(last_layer._last_z)
                elif loss == "cross_entropy":
                   
                    y_pred_probs = activations[-1]
                    delta = cross_entropy_grad_softmax(y_pred_probs, yb)  # already dL/dz_last when softmax used in forward
                else:
                    raise ValueError("Unsupported loss")
                
                for l_idx in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[l_idx]
                    A_prev = activations[l_idx]
                    # delta shape: (batch, n_units)
                    m = A_prev.shape[0]
                    dW = (delta.T @ A_prev)  # shape (n_units, n_inputs)
                    db = np.sum(delta, axis=0)
                    # compute delta for previous layer using current weights BEFORE updating
                    if l_idx > 0:
                        dA_prev = delta @ layer.W  # use current W (not yet updated)
                        prev_layer = self.layers[l_idx - 1]
                        deriv_fn = _ACTS[prev_layer.activation][1]
                        if prev_layer.activation == "softmax":
                            # softmax derivative is complicated; softmax in hidden layers is unusual
                            raise NotImplementedError("Softmax derivative in hidden layers is not implemented")
                        next_delta = dA_prev * deriv_fn(prev_layer._last_z)
                    else:
                        next_delta = None
                    # update parameters (SGD)
                    layer.W -= lr * dW
                    layer.b -= lr * db
                    if next_delta is not None:
                        delta = next_delta
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs):
                y_pred = self.predict(X)
                loss_val = self.compute_loss(y_pred, y, loss=loss)
                print(f"Epoch {epoch}/{epochs} loss={loss_val:.6f}")
            # record full-dataset loss at end of epoch
            y_pred_full = self.predict(X)
            history.append(self.compute_loss(y_pred_full, y, loss=loss))
        return history


    def get_layer(self, idx: int) -> Layer:
        return self.layers[idx]

    def get_layer_neuron(self, layer_idx: int, neuron_idx: int) -> Neuron:
        return self.layers[layer_idx].neurons()[neuron_idx]

    def set_layer_weights(self, layer_idx: int, W: np.ndarray, b: np.ndarray):
        self.layers[layer_idx].set_weights(W, b)

    def save_weights(self, filepath: str):

        to_save = {}
        for i, layer in enumerate(self.layers):
            to_save[f"W{i}"] = layer.W
            to_save[f"b{i}"] = layer.b
            to_save[f"act{i}"] = np.array(layer.activation, dtype='<U16')
        np.savez(filepath, **to_save)

    def load_weights(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        layers = []
        i = 0
        while f"W{i}" in data:
            W = data[f"W{i}"]
            b = data[f"b{i}"]
            act = str(data[f"act{i}"].tolist())
           
            layer = Layer(W.shape[1], W.shape[0], activation=act)
            layer.set_weights(W.copy(), b.copy())
            layers.append(layer)
            i += 1
        self.layers = layers

def load_csv(filepath: str, input_cols: Sequence[int], output_cols: Sequence[int], delimiter: str = ",", has_header: bool = True, dtype=float) -> Tuple[np.ndarray, np.ndarray]:

    header = 0 if has_header else None
    df = pd.read_csv(filepath, sep=delimiter, header=header)

    def _select_columns(df_local, cols):
        if len(cols) == 0:
            return np.empty((len(df_local), 0))
        
        if all(isinstance(c, int) for c in cols):
            return df_local.iloc[:, list(cols)].to_numpy(dtype=float)
        
        return df_local.loc[:, list(cols)].to_numpy(dtype=float)

    if df.shape[0] == 0:
        X = np.empty((0, len(input_cols)))
        y = np.empty((0, len(output_cols)))
    else:
        X = _select_columns(df, input_cols)
        y = _select_columns(df, output_cols)
    return X, y

def parse_tle_lines(line1: str, line2: str) -> np.ndarray:

    incl = raan = ecc = argp = mean_anom = mean_motion = np.nan
    try:
        
        incl = float(line2[8:16].strip())
        raan = float(line2[17:25].strip())
        ecc_str = line2[26:33].strip()
        ecc = float("0." + ecc_str) if ecc_str != "" else 0.0
        argp = float(line2[34:42].strip())
        mean_anom = float(line2[43:51].strip())
        mean_motion = float(line2[52:63].strip())
    except Exception:
        pass

    incl_r = np.deg2rad(incl) if not np.isnan(incl) else 0.0
    raan_r = np.deg2rad(raan) if not np.isnan(raan) else 0.0
    argp_r = np.deg2rad(argp) if not np.isnan(argp) else 0.0
    mean_anom_r = np.deg2rad(mean_anom) if not np.isnan(mean_anom) else 0.0


    mu = 398600.4418 
    a_km = np.nan
    try:
        mm_rad_s = mean_motion * 2.0 * np.pi / 86400.0
        if mm_rad_s > 0:
            a_km = (mu / (mm_rad_s ** 2)) ** (1.0 / 3.0)
    except Exception:
        a_km = np.nan

    feats = np.array([
        0.0 if np.isnan(ecc) else ecc,
        0.0 if np.isnan(mean_motion) else mean_motion,
        0.0 if np.isnan(a_km) else a_km,
        np.sin(incl_r), np.cos(incl_r),
        np.sin(raan_r), np.cos(raan_r),
        np.sin(argp_r), np.cos(argp_r),
        np.sin(mean_anom_r), np.cos(mean_anom_r)
    ], dtype=float)
    print(feats)
    return feats

def load_tle_csv(filepath: str, name_col=0, line1_col=1, line2_col=2, delimiter: str = ",", has_header: bool = True) -> Tuple[np.ndarray, np.ndarray]:

    header = 0 if has_header else None
    df = pd.read_csv(filepath, sep=delimiter, header=header, dtype=str, keep_default_na=False)
    features = []
    names = []

    def get_cell(row, col):
        return row.iloc[col] if isinstance(col, int) else row[col]

    for _, row in df.iterrows():
        try:
            l1 = str(get_cell(row, line1_col)).rstrip()
            l2 = str(get_cell(row, line2_col)).rstrip()
            nm = str(get_cell(row, name_col))
        except Exception:
            continue
        feats = parse_tle_lines(l1, l2)
        features.append(feats)
        names.append(nm)
    if len(features) == 0:
        X = np.empty((0, 11))
    else:
        X = np.vstack(features)
    return X, np.array(names)

def heuristic_risk_from_features(X: np.ndarray) -> np.ndarray:

    if X.shape[0] == 0:
        return np.empty((0, 1))

    a_km = X[:, 2]
    R_earth = 6371.0
    alt_km = a_km - R_earth
    alt_score = np.exp(-((alt_km - 700.0) / 300.0) ** 2)  # peak ~700 km
    ecc_score = np.clip(X[:, 0] * 5.0, 0.0, 1.0)
    incl = np.arctan2(X[:, 3], X[:, 4])
    inc_target = np.deg2rad(98.0)
    inc_score = np.exp(-((incl - inc_target) / np.deg2rad(10.0)) ** 2)
    risk = 0.6 * alt_score + 0.2 * ecc_score + 0.2 * inc_score
    return np.clip(risk, 0.0, 1.0).reshape(-1, 1)


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B

def transpose(A: np.ndarray) -> np.ndarray:
    return A.T

def _train_test_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: Optional[int] = None):
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    t = int(n * (1 - test_frac))
    train_idx = perm[:t]
    test_idx = perm[t:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def _standardize_fit(X: np.ndarray):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def _standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma

def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    n = y_true.shape[0]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float('nan')
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def train_and_compare_physics(X: np.ndarray, y: np.ndarray, physics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None, seed: Optional[int] = 42):
    """
    Split, normalize, train NN, and compare to physics_fn baseline on test set.
    physics_fn takes features X and returns predictions in same shape as y.
    """
    if X.shape[0] < 4:
        print("Not enough samples to split for train/test. Need >=4 rows.")
        return

    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_frac=0.2, seed=seed)

    mu, sigma = _standardize_fit(X_train)
    X_train_s = _standardize_apply(X_train, mu, sigma)
    X_test_s = _standardize_apply(X_test, mu, sigma)

    # build a small network
    nn = NeuralNetwork(seed=seed)
    nn.add_layer(32, activation="relu", input_dim=X.shape[1])
    nn.add_layer(16, activation="relu")
    nn.add_layer(1, activation="sigmoid")

    # train on the training split
    nn.train(X_train_s, y_train, epochs=200, lr=0.01, batch_size=min(32, X_train_s.shape[0]), loss="mse", verbose=False)

    # predictions
    ypred_nn = nn.predict(X_test_s).flatten()
    if physics_fn is not None:
        ypred_phys = physics_fn(X_test).flatten()
    else:
        ypred_phys = None

    # metrics
    m_nn = _metrics(y_test, ypred_nn)
    print("NN on test set:", m_nn)
    if ypred_phys is not None:
        m_phys = _metrics(y_test, ypred_phys)
        print("Physics baseline on test set:", m_phys)
        # compare
        print("Delta MAE (NN - Phys):", round(m_nn["MAE"] - m_phys["MAE"], 6))
    else:
        print("No physics function provided; only NN evaluated.")

    # show sample rows for quick inspection
    print("\nSample (y_true, y_nn, y_phys):")
    for i in range(min(8, X_test.shape[0])):
        phys_val = f"{ypred_phys[i]:.3f}" if ypred_phys is not None else "N/A"
        print(f"{float(y_test.flatten()[i]):.3f}   {ypred_nn[i]:.3f}   {phys_val}")

    # return trained model and normalizer so you can reuse them
    return {"nn": nn, "mu": mu, "sigma": sigma, "metrics_nn": m_nn, "metrics_phys": (m_phys if ypred_phys is not None else None)}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train TLE risk model")
    p.add_argument("--file", "-f", default="tle_file.csv", help="TLE CSV path")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42, help="If -1, use random seeds per run")
    p.add_argument("--runs", type=int, default=1, help="Train this many independent runs (different seeds)")
    p.add_argument("--loss", choices=["mse", "cross_entropy"], default="mse")
    p.add_argument("--normalize", action="store_true", help="Standardize features using training split")
    args = p.parse_args()

    # locate file
    tle_path = args.file
    if not os.path.exists(tle_path):
        print(f"File not found: {tle_path}")
        print("PWD:", os.getcwd())
        raise SystemExit(1)

    X, names = load_tle_csv(tle_path, name_col=0, line1_col=1, line2_col=2, has_header=True)
    if X.shape[0] == 0:
        print("No TLEs parsed; exiting.")
        raise SystemExit(1)

    y = heuristic_risk_from_features(X) 
    mu = sigma = None
    if args.normalize:
        mu, sigma = _standardize_fit(X)
        X_proc = _standardize_apply(X, mu, sigma)
    else:
        X_proc = X.copy()

    runs = args.runs
    seeds = []
    if args.seed == -1:
        seeds = [int(time.time()) + i for i in range(runs)]
    else:
        seeds = [args.seed + i for i in range(runs)]

    all_histories = []
    for run_i, s in enumerate(seeds, 1):
        print(f"\nRun {run_i}/{runs} seed={s}")
        nn = NeuralNetwork(seed=s if s is not None else None)
        nn.add_layer(16, activation="relu", input_dim=X_proc.shape[1], name="hidden1")
        nn.add_layer(8, activation="relu", name="hidden2")
        nn.add_layer(1, activation="sigmoid", name="risk_out")

        history = nn.train(X_proc, y, epochs=args.epochs, lr=args.lr, batch_size=min(args.batch_size, X_proc.shape[0]), loss=args.loss, verbose=True)
        all_histories.append(history)

        preds = nn.predict(X_proc).flatten()
        print("Sample preds:", preds[:10])

    if runs > 1:
        finals = [h[-1] if len(h) else float("nan") for h in all_histories]
        print("\nFinal losses across runs:", finals)