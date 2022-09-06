import jax.numpy as jnp
from kprior.model import kprior, loss_map_grad, loss_kprior_grad

def test_train_to_converge(X, Y, W1, lvl=0.001):
  grad = loss_map_grad(X, Y, W1)
  return grad.T @ grad < lvl
  
def test_kprior(W, M, err=0.01):
  loss1, loss2, total = kprior(W, M, W, debug = True)

  assert jnp.abs(loss1 - total) < err, f"The error is {jnp.abs(loss1 - total)}"

def test_full_gradient_recovery(X1, Y1, W1, W2, error_level = 0.001):
  true_grad = loss_map_grad(X1, Y1, W2) 
  corr_factor = loss_map_grad(X1, Y1, W1) 
  kprior_grad = loss_kprior_grad(W2, X1, W1)
  out = kprior_grad - (true_grad-corr_factor)
  assert out.T @ out < error_level, f"The current error level is {out.T @ out: 0.2f} which is above the error level of {error_level}"

def evaluate_error_gradient_recovery(X1, Y1, M, W1, W2):
  true_grad = loss_map_grad(X1, Y1, W2) 
  kprior_grad = loss_kprior_grad(M, W1, W2)
  return jnp.mean(0.5*(true_grad - kprior_grad)**2)