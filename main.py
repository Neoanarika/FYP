from jax import random
from optimiser import optimiser
from model import Phi
from kprior import selectM
from dataset import make_linearly_seperable_dataset
from utils import plot_decision_boundary, plot_leverge_points

class hyperparameters():
    n_iter=100000
    lr = 0.001
    topk = 10
    delta = 0.2
    seed = 0 

if  __name__ == "__main__":
    param = hyperparameters()
    X_task_1, y_task_1, X_task_2, y_task_2, X_task_total, y_task_total = make_linearly_seperable_dataset()

    X1 = Phi(X_task_1)
    X2 = Phi(X_task_2)
    Xtotal = Phi(X_task_total) 

    key = random.PRNGKey(param.seed)
    key, W_key = random.split(key)
    W0 = random.normal(W_key, (X1.shape[1],))

    W1 = optimiser(X1, y_task_1, W0, n_iter=param.n_iter, lr=param.lr, delta=param.delta)
    Wall = optimiser(Xtotal, y_task_total, W0, n_iter=param.n_iter, lr=param.lr, delta=param.delta)

    M, X_minus_M_id, scores = selectM(X1, W1, topk=param.topk)
    W2 = optimiser(X2, y_task_2, W1, W1, M, n_iter=param.n_iter, lr=param.lr, delta=param.delta)  

    data = [("Old data points", X_task_1, y_task_1)]
    plot_decision_boundary(data, W1)

    data = [("Old data points", X_task_1, y_task_1), ("New data points", X_task_2, y_task_2)]
    plot_decision_boundary(data, Wall

    data = [("Old data points", X_task_1, y_task_1), ("New data points", X_task_2, y_task_2)]
    plot_leverge_points(data, W2, W1, M, X_minus_M_id, 50*(scores/jnp.min(scores)))