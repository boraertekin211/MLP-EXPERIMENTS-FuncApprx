
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# ---------------------------------------------------------
# 1) True (nonlinear) function
# ---------------------------------------------------------
def f(x):
    return np.sin(2 * x) + 0.5 * x

X_MIN, X_MAX = -3, 3

# Fixed, dense set of points used for testing and plotting
x_test = np.linspace(X_MIN, X_MAX, 500).reshape(-1, 1)
y_test = f(x_test).ravel()
plt.figure(); plt.plot(x_test, y_test, "b-"); plt.title("True function: y = sin(2x) + 0.5x"); plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.savefig("true_function.png", dpi=150)

def generate_data(n_samples, noise=0.05, seed=0):
    """Creates the (X, y) table: single input, single output."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, n_samples).reshape(-1, 1)
    y = f(x).ravel() + rng.normal(0, noise, n_samples)
    return x, y


# ---------------------------------------------------------
# 2) Function that runs a single experiment
# ---------------------------------------------------------
def experiment(n_neurons, n_samples, activation):
    x_train, y_train = generate_data(n_samples)

    model = MLPRegressor(
        hidden_layer_sizes=(n_neurons,),  # single hidden layer
        activation=activation,            # 'logistic' (sigmoid) or 'relu'
        solver="lbfgs",                   # fast and stable for small datasets
        max_iter=5000,
        random_state=1,
    )
    model.fit(x_train, y_train)           # output layer is linear by default

    y_pred = model.predict(x_test)

    # Squared errors
    sse_train = np.sum((model.predict(x_train) - y_train) ** 2)
    mse_train = sse_train / len(y_train)
    sq_err_test = (y_pred - y_test) ** 2
    sse_test = np.sum(sq_err_test)
    mse_test = np.mean(sq_err_test)

    return dict(x_train=x_train, y_train=y_train, y_pred=y_pred,
                sse_train=sse_train, mse_train=mse_train,
                sse_test=sse_test, mse_test=mse_test)


# ---------------------------------------------------------
# 3) Plots several cases side by side and prints a results table
# ---------------------------------------------------------
def compare(cases, title, file_name):
    """cases: list of (label, n_neurons, n_samples, activation)"""
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    print(f"\n=== {title} ===")
    print(f"{'Case':<24}{'SSE(train)':>12}{'MSE(train)':>12}{'SSE(test)':>12}{'MSE(test)':>12}")

    for ax, (label, n_neurons, n_samples, act) in zip(axes, cases):
        s = experiment(n_neurons, n_samples, act)
        print(f"{label:<24}{s['sse_train']:>12.4f}{s['mse_train']:>12.5f}"
              f"{s['sse_test']:>12.4f}{s['mse_test']:>12.5f}")

        ax.plot(x_test, y_test, "b-", label="True function")
        ax.plot(x_test, s["y_pred"], "r--", label="MLP output")
        ax.scatter(s["x_train"], s["y_train"], s=10, c="gray", alpha=0.6,
                   label="Training data")
        ax.set_title(f"{label}\nTest MSE = {s['mse_test']:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(file_name, dpi=150)


# ---------------------------------------------------------
# EXPERIMENT 1: Number of hidden neurons (few / suitable / many)
# ---------------------------------------------------------
compare(
    [("Few (1 neuron)",       1,   200, "logistic"),
     ("Suitable (10 neurons)", 10,  200, "logistic"),
     ("Many (200 neurons)",    200, 200, "logistic")],
    "Experiment 1: Effect of the number of neurons (200 samples, sigmoid)",
    "experiment1_neurons.png",
)

# ---------------------------------------------------------
# EXPERIMENT 2: Number of samples (few / suitable / many)
# ---------------------------------------------------------
compare(
    [("Few (8 samples)",        20, 8,    "logistic"),
     ("Suitable (200 samples)", 20, 200,  "logistic"),
     ("Many (5000 samples)",    20, 5000, "logistic")],
    "Experiment 2: Effect of the number of samples (20 neurons, sigmoid)",
    "experiment2_samples.png",
)

# ---------------------------------------------------------
# EXPERIMENT 3: Sigmoid vs ReLU
# ---------------------------------------------------------
compare(
    [("Sigmoid (10 neurons)", 10, 200, "logistic"),
     ("ReLU (10 neurons)",    10, 200, "relu")],
    "Experiment 3: Activation function (10 neurons, 200 samples)",
    "experiment3_activation.png",
)

plt.show()
