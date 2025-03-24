import numpy as np
import deepxde as dde
from copy import deepcopy
import scipy.io
import sys
import math
import os
import scipy.interpolate

reference_data = np.loadtxt("Datasets/surfaceFields.txt", skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
x_cell_centroid = reference_data[:, 2]
y_cell_centroid = reference_data[:, 3]
u_mean_x = reference_data[:, 7]
u_mean_y = reference_data[:, 8]
t_mean = reference_data[:, 6]
vol_weights = reference_data[:, 11]
Re = 150
Pr = 1

points = np.zeros((len(x_cell_centroid), 2))
for i in range(len(x_cell_centroid)):
    points[i][0] = x_cell_centroid[i]
    points[i][1] = y_cell_centroid[i]
    
for k in range(1, 11):

  # integer before 'j' = (number of points in x or y) + 1    
      
  grid_x, grid_y = np.mgrid[-14.75:39.85:(55j*k)+1j, -9.9:9.9:(20j*k)+1j]
  num_x = np.shape(grid_x)[0]
  num_y = np.shape(grid_y)[1]
  
  gridu = scipy.interpolate.griddata(points, u_mean_x, (grid_x, grid_y)).T
  gridv = scipy.interpolate.griddata(points, u_mean_y, (grid_x, grid_y)).T
  gridt = scipy.interpolate.griddata(points, t_mean, (grid_x, grid_y)).T
  
  # now need to generate new x and y points using linspace
  
  x_line = np.linspace(-14.75, 39.85, num_x)
  y_line = np.linspace(-9.9, 9.9, num_y)
      
  # reformat new_grid u data to be same shape as x and y
  
  current_y = np.zeros(num_x)
  new_x = []
  new_y = []
  new_u = []
  new_v = []
  new_t = []
  for i in range(num_y):
      for j in range(num_x):
          current_y[j] = y_line[i]
      new_y = np.concatenate((new_y, current_y))
      new_x = np.concatenate((new_x, x_line))
      new_u = np.concatenate((new_u, gridu[i]))
      new_v = np.concatenate((new_v, gridv[i]))
      new_t = np.concatenate((new_t, gridt[i]))
  
  new_x = np.array(new_x)
  new_y = np.array(new_y)
  new_u = np.array(new_u)
  new_v = np.array(new_v)
  new_t = np.array(new_t)
  
  vertex_botleft = [-3, -4]
  vertex_topright = [15, 4]
  geom = dde.geometry.Rectangle(vertex_botleft, vertex_topright) - dde.geometry.Disk([0, 0], 0.5)
  
  def pde(X, V):
      # put quantities into arrays of single-value arrays
      u = V[:, 0:1]
      v = V[:, 1:2]
      p = V[:, 2:3]
      t = V[:, 3:4]
      fx = V[:, 4:5]
      fy = V[:, 5:6]
      g = V[:, 6:7]
      
      # compute partial derivatives. components i, j: i is the component from the V matrix (0 - 6 for respective properties),
      # j is component for spatial coordinate (0 - 1 for x or y)
      du_x = dde.grad.jacobian(V, X, i=0, j=0)
      dv_y = dde.grad.jacobian(V, X, i=1, j=1)
      du_y = dde.grad.jacobian(V, X, i=0, j=1)
      dv_x = dde.grad.jacobian(V, X, i=1, j=0)
      dp_x = dde.grad.jacobian(V, X, i=2, j=0)
      dp_y = dde.grad.jacobian(V, X, i=2, j=1)
      dt_x = dde.grad.jacobian(V, X, i=3, j=0)
      dt_y = dde.grad.jacobian(V, X, i=3, j=1)
      # compute partial second derivatives. First parameter is variable rather than matrix.
      # i, j are denominator spatial coordinate indices, e.g dy/dxidxj
      du_xx = dde.grad.hessian(u, X, i=0, j=0)
      dv_xx = dde.grad.hessian(v, X, i=0, j=0)
      du_yy = dde.grad.hessian(u, X, i=1, j=1)
      dv_yy = dde.grad.hessian(v, X, i=1, j=1)
      dt_xx = dde.grad.hessian(t, X, i=0, j=0)
      dt_yy = dde.grad.hessian(t, X, i=1, j=1)
      # compute partial derivatives of forcing term
      dfx_x = dde.grad.jacobian(V, X, i=4, j=0)
      dfy_y = dde.grad.jacobian(V, X, i=5, j=1)
      
      return [
          du_x + dv_y,
          u * du_x + v * du_y + dp_x - (1.0 / Re) * (du_xx + du_yy) + fx,
          u * dv_x + v * dv_y + dp_y - (1.0 / Re) * (dv_xx + dv_yy) + fy,
          dfx_x + dfy_y, 
          u * dt_x + v * dt_y - (1.0 / (Re * Pr)) * (dt_xx + dt_yy) + g
      ]
  
  def generate_training_points(x, y, u, v, t):
      x_t = []
      y_t = []
      u_t = []
      v_t = []
      t_t = []
      
      x_t = np.array(x_t)
      y_t = np.array(y_t)
      u_t = np.array(u_t)
      v_t = np.array(v_t)
      t_t = np.array(t_t)
      
      x_t = x[0::1].reshape(-1, 1)
      y_t = y[0::1].reshape(-1, 1)
      u_t = u[0::1].reshape(-1, 1)
      v_t = v[0::1].reshape(-1, 1)
      t_t = t[0::1].reshape(-1, 1)
      
      X = []
      
      for i in range(x_t.shape[0]):
          if geom.inside([x_t[i, 0], y_t[i, 0]]) \
                  and x_t[i, 0] > vertex_botleft[0] and x_t[i, 0] < vertex_topright[0] \
                  and y_t[i, 0] > vertex_botleft[1] and y_t[i, 0] < vertex_topright[1]:
              X.append([x_t[i, 0], y_t[i, 0], u_t[i, 0], v_t[i, 0], t_t[i, 0]])
      
      X = np.array(X)
   
      return np.hsplit(X, 5)
  
  [x_training, y_training, u_training, v_training, t_training] = \
      generate_training_points(new_x, new_y, new_u, new_v, new_t)
  
  training_points = np.hstack((x_training, y_training))
  
  def func_zeros(X):
      x = X[:, 0:1]
      return x * 0
  
  def func_temp(X):
      x = X[:, 0:1]
      return 1
  
  def boundary(x, on_boundary):
      return on_boundary and not(
          np.isclose(x[0], vertex_botleft[0]) or np.isclose(x[0], vertex_topright[0]) 
          or np.isclose(x[1], vertex_botleft[1]) or np.isclose(x[1], vertex_topright[1])
      )
  
  # u, v dirichlet BCs at each point in training_points
  u_training_points = dde.PointSetBC(training_points, u_training, component=0)
  v_training_points = dde.PointSetBC(training_points, v_training, component=1)
  t_training_points = dde.PointSetBC(training_points, t_training, component=3)
  # dirichlet BCs at walls, component indices from pde function
  bc_wall_u = dde.DirichletBC(geom, func_zeros, boundary, component=0)
  bc_wall_v = dde.DirichletBC(geom, func_zeros, boundary, component=1)
  bc_wall_t = dde.DirichletBC(geom, func_temp, boundary, component=3)
  bc_wall_fx = dde.DirichletBC(geom, func_zeros, boundary, component=4)
  bc_wall_fy = dde.DirichletBC(geom, func_zeros, boundary, component=5)
  bc_wall_g = dde.NeumannBC(geom, func_zeros, boundary, component=6)
  
  data = dde.data.PDE(
      geom, pde, 
      [bc_wall_u, bc_wall_v, bc_wall_t, bc_wall_fx, bc_wall_fy, bc_wall_g, u_training_points, v_training_points, t_training_points], 
      0, 1500, anchors=np.loadtxt("Datasets/collocation_points.txt"), solution=None, num_test=50, 
  )
  
  layer_size = [2] + [100] * 7 + [7]
  activation = "tanh"
  initializer = "Glorot uniform"
  net = dde.maps.FNN(layer_size, activation, initializer)
  
  model = dde.Model(data, net)
  
  loss_weights = [1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10]
  model.compile("adam", lr=0.001, loss_weights=loss_weights)
  
  losshistory, train_state = model.train(25000, display_every=1000)
  
  model.compile("L-BFGS-B", loss_weights=loss_weights)
  dde.optimizers.config.set_LBFGS_options(maxiter=35000)
  losshistory, train_state = model.train()
  
  dde.saveplot(losshistory, train_state, issave=True, isplot=False, loss_fname="surface_data_loss_"+str(k), output_dir="Losses")
  model.save("models/surface_data_" + str(k))
