import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.numpy.linalg import cholesky #looks like numpy cholesky is more stable
from kprior.model import score, grad_f

def levergeScore(W,X, delta=1):
  J_x = grad_f(W, X) 
  lambda_x = jnp.diag(score(W,X))
  Gmat =  J_x.T @ lambda_x  @ J_x + delta*jnp.eye(X.shape[1])
  # assert jnp.sum(grad_X - X) == 0, "J_x is not equal to X in the glm case"
  leverage = jnp.diagonal(J_x @ jnp.linalg.pinv(Gmat) @ J_x.T @ lambda_x)
  return leverage
  # J_x = grad_f(W, X)
  # lambda_x = jnp.diag(score(W,X))
  # return jnp.diag(J_x @ J_x.T @ jnp.linalg.pinv(J_x @ J_x.T + jnp.linalg.pinv(lambda_x)))

def leverage_scores_nn(gram: jnp.array, lambdas: jnp.array, delta: jnp.array, epsilon: float=1e-8):
    """Scalar output only
    gram : shp (n,n)
    lambdas : shp (n,)
    epsilon : avoid divide-by-zero
    """
    gram_plus_ridge = gram + delta * (1/lambdas) * jnp.eye(gram.shape[0])
    L = cholesky(gram_plus_ridge)
    L_inv = solve_triangular(L.T, jnp.eye(L.shape[0]))
    gram_inv = L_inv.dot(L_inv.T)
    print("triggered")
    return jnp.einsum("ij,ji->i", gram, gram_inv)

def selectM(X, W, topk= 10):
  """
  Select form X based on topk highest leverge scores
  """  
  # K = X @ X.T
  # lambda_x = score(W, X)
  # scores = leverage_scores_nn(K,lambda_x, delta=1)
  scores = levergeScore(W, X)
  id1 = jnp.argsort(scores)[-topk:]
  id2 = jnp.argsort(scores)[:len(X)-topk]
  return X[id1], id2, scores[id1]