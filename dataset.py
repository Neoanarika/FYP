import numpy as np
import jax.numpy as jnp
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

def make_linearly_seperable_dataset():
    X, y = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=0)
    X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=0.5, random_state=42)

    subspace1 = np.logical_and(np.logical_and(X_1[:, 0] < 0, X_1[:, 1] < 0.8), X_1[:, 1]> 0.3)
    subspace2 = np.logical_and(np.logical_and(X_2[:, 0] > 0, X_2[:, 1] < 1), X_2[:, 1]> 0.8)

    X_task_1 = X_1[subspace1]
    X_task_2 = X_2[subspace2]

    y_task_1 = y_1[subspace1]
    y_task_2 = y_2[subspace2]

    X_task_1 = jnp.array(X_task_1)
    X_task_2 = jnp.array(X_task_2)

    y_task_1 = jnp.array(y_task_1)
    y_task_2 = jnp.array(y_task_2)

    X_task_total = jnp.concatenate([X_task_1, X_task_2], axis=0)
    y_task_total = jnp.concatenate([y_task_1, y_task_2], axis=0)

    return X_task_1, y_task_1, X_task_2, y_task_2, X_task_total, y_task_total