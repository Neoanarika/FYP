import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import grad, jit

def f(params, x):
  return params.T @ x

def h(params, x):
  """
  Logistic regression model

  Parameters
  ----------
  X: jnp.array
     The input data
  w: jnp.array
     The weights of the mdoel

  Returns
  -------
  jnp.array 
    probability of it being class 0 or 1. 
  """
  return sigmoid(f(params, x))

def phi(x):
  """
  Makes feature for the logistic regression model

  Parameters
  ----------
  X: jnp.array
     The input data

  Returns
  -------
  jnp.array 
    The data with features 
  """
  power_1 = x
  out = jnp.insert(power_1, 0 , 1)
  return out

Phi = jit(jax.vmap(phi))
F = jit(jax.vmap(f, (None, 0)))
H = jit(jax.vmap(h, (None, 0)))

def celoss(preds, Y):
  """
  Binary cross entropy loss 

  Parameters
  ----------
  preds: jnp.array
     The predictions from the model

  Y: jnp.array
    The lables

  Returns
  -------
  jnp.array 
    The cross entropy scores
  """
  return -jnp.sum(Y * jnp.log(preds) + (1 - Y) * jnp.log(1 - preds)) 

loss_map = lambda X, Y, W, delta=0.2: celoss(H(W, X), Y) + (delta/2)*W.T@W

def kprior(W, M, W_prior, delta=0.2, debug=False):
  """
  Knowledge adaption prior implementation

  Parameters
  ----------
  W: jnp.array
     The new weights of model for current task.

  M: jnp.array
    The memory

  W_prior: jnp.array
     Prior weights from previous task.

  Returns
  -------
  jnp.array 
    The kprior value
  """
  assert W.shape == (3, )
  assert W_prior.shape == (3, )
  
  loss_1 = celoss(H(W, M), H(W_prior, M))
  loss_2 = (delta/2)*(W-W_prior).T@(W-W_prior)

  if debug:
    return loss_1, loss_2, loss_1+loss_2

  return  loss_1+loss_2

grad_f = jit(jax.vmap(grad(f, argnums=0), (None, 0)))

def dsigmoid(s):
  """
  Derivative of sigmoid
  """  
  return s - s*s

def G(W,X, delta=0.2):
  J_x = grad_f(W, X)
  assert jnp.sum(J_x - X) == 0, "J_x is not equal to X in the glm case"
  lambda_x = jnp.diag(score(W,X))
  return J_x.T @ lambda_x  @ J_x + delta*jnp.eye(X.shape[1])

def score(W,X):
  preds = H(W, X)
  lamb = dsigmoid(preds)
  return lamb

loss = lambda X, Y, M, W, W_prior, delta=0.2: celoss(H(W, X), Y) + kprior(W, M, W_prior, delta=delta)
loss_map_grad = jit(grad(loss_map, argnums=2))
loss_grad = jit(grad(loss, argnums=3))
loss_kprior_grad = jit(grad(kprior, argnums=0))

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