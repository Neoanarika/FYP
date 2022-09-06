import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(data, W, M=None):
  b = W[0]
  w1, w2 = W[1], W[2]
  # Calculate the intercept and gradient of the decision boundary.
  c = -b/w2
  m = -w1/w2

  xmin, xmax = -1, 1
  ymin, ymax = -0.5, 1.5

  point_size = 50
  point_alpha = 0.7

  background_alpha = 0.2

  xd = np.array([xmin, xmax])
  yd = m*xd + c
  fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8,8))
  axs.plot(xd, yd, 'k', lw=1, ls='--')
  axs.fill_between(xd, yd, ymin, color='tab:orange', alpha=background_alpha)
  axs.fill_between(xd, yd, ymax, color='tab:blue', alpha=background_alpha)
  
  for (name, X, y) in data:
    axs.scatter(*X[y==0].T, s=point_size, alpha=point_alpha, label=f"{name} label 1")
    axs.scatter(*X[y==1].T, s=point_size, alpha=point_alpha, label=f"{name} label 2")

  axs.set_xlim(xmin, xmax)
  axs.set_ylim(ymin, ymax)
  axs.set_title("Classification of datapoints", fontsize=15)
  axs.set_ylabel(r'$x_2$', fontsize=15)
  axs.set_xlabel(r'$x_1$', fontsize=15)
  plt.legend(loc=2, fontsize="large")

def plot_leverge_points(data, W, W_prior, M, X_minus_M_id, S):
  b = W[0]
  w1, w2 = W[1], W[2]
  # Calculate the intercept and gradient of the decision boundary.
  c = -b/w2
  m = -w1/w2

  b_p = W_prior[0]
  w_p1, w_p2 = W_prior[1], W_prior[2]
  # Calculate the intercept and gradient of the decision boundary.
  c_p = -b_p/w_p2
  m_p = -w_p1/w_p2

  xmin, xmax = -1, 1
  ymin, ymax = -0.5, 1.5

  point_size = 50
  point_alpha = 0.7

  background_alpha = 0.2

  xd = np.array([xmin, xmax])
  yd = m*xd + c
  yd_p = m_p*xd + c_p
  fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(20,10))
  axs[0].plot(xd, yd_p, 'k', lw=1, ls='--')
  axs[0].fill_between(xd, yd_p, ymin, color='tab:orange', alpha=background_alpha)
  axs[0].fill_between(xd, yd_p, ymax, color='tab:blue', alpha=background_alpha)
  axs[1].plot(xd, yd, 'k', lw=1, ls='--')
  axs[1].fill_between(xd, yd, ymin, color='tab:orange', alpha=background_alpha)
  axs[1].fill_between(xd, yd, ymax, color='tab:blue', alpha=background_alpha)

  for (name, X, y) in data:
    axs[0].scatter(*X[y==0].T, s=point_size, alpha=point_alpha, label=f"{name} label 1")
    axs[0].scatter(*X[y==1].T, s=point_size, alpha=point_alpha, label=f"{name} label 2")
    if "Old" in name:
      X, y = X[X_minus_M_id], y[X_minus_M_id]
      axs[1].scatter(*X[y==0].T, s=point_size, alpha=point_alpha, label=f"{name} label 1")
      axs[1].scatter(*X[y==1].T, s=point_size, alpha=point_alpha, label=f"{name} label 2")
    else:
      axs[1].scatter(*X[y==0].T, s=point_size, alpha=point_alpha, label=f"{name} label 1")
      axs[1].scatter(*X[y==1].T, s=point_size, alpha=point_alpha, label=f"{name} label 2")

  axs[0].scatter(M[:, 1], M[:, 2], color="red", s = S, label=f"memorable points",edgecolors='black')
  axs[1].scatter(M[:, 1], M[:, 2], color="red", s = S, label=f"memorable points",edgecolors='black')

  axs[0].set_xlim(xmin, xmax)
  axs[0].set_ylim(ymin, ymax)
  axs[0].set_ylabel(r'$x_2$', fontsize=15)
  axs[0].set_xlabel(r'$x_1$', fontsize=15)
  axs[0].set_title("Without leverge points")
  axs[0].legend(loc=2, fontsize="large")

  axs[1].set_xlim(xmin, xmax)
  axs[1].set_ylim(ymin, ymax)
  axs[1].set_ylabel(r'$x_2$', fontsize=15)
  axs[1].set_xlabel(r'$x_1$', fontsize=15)
  axs[1].set_title("With leverge points")
  axs[1].legend(loc=2, fontsize="large")