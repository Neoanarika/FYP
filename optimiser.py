from tqdm.notebook import tqdm
from model import loss_map_grad, loss_grad

def optimiser(X, Y, W_0, W_prior=None, M=None, lr=0.01, n_iter=1000, delta= 0.2, error_level=0.001):
  """
  Trains the model using gradient descent
  """
  for i in tqdm(range(n_iter)):
    if M is None and W_prior is None:
      W_0 = W_0 - lr*loss_map_grad(X, Y, W_0, delta=delta)
    else:
      out = loss_grad(X, Y, M, W_0, W_prior, delta=delta)
      W_0 = W_0 - lr*out

  return W_0