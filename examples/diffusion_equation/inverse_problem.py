import numpy as np
import deepxde as xde
from deepxde.backend import tensorflow as tf
import tensorflow as tf2

if __name__ == '__main__':

    C = xde.Variable(2.0)

    """" Define Domain """
    x_domain = xde.geometry.Interval(-1, 1)
    time_domain = xde.geometry.TimeDomain(0, 1)
    full_domain = xde.geometry.GeometryXTime(x_domain, time_domain)

    """" Define PDE """

    def diffusion_equation(x, y):
        # Derivatives
        dy_dt = xde.gradients.jacobian(y, x, i=0, j=1)
        d2y_dx2 = xde.gradients.hessian(y, x, i=0, j=0)
        # Equation
        pde = dy_dt - C * d2y_dx2 + tf2.math.exp(-x[:, 1:]) * (tf.sin(np.pi * x[:, 0:1]) - (np.pi**2)*tf.sin(np.pi*x[:, 0:1]))
        return pde

    def forward_solution(x):
        solution = np.sin(np.pi*x[:, 0:1])*np.exp(-x[:, 1:])
        return solution

    """" Initial and Boundary conditions """
    boundary_conditions = xde.DirichletBC(full_domain, forward_solution, lambda _, on_boundary: on_boundary)
    initial_conditions = xde.IC(full_domain, forward_solution, lambda _, on_initial: on_initial)

    observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T
    observe_y = xde.PointSetBC(observe_x, forward_solution(observe_x), component=0)

    """" Data for PINN """
    data = xde.data.TimePDE(
        full_domain,
        diffusion_equation,
        [boundary_conditions, initial_conditions, observe_y],
        num_domain=40,
        num_initial=10,
        anchors=observe_x,
        solution=forward_solution,
        num_test=10000
    )

    layer_size = [2] + [32]*3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = xde.maps.FNN(layer_size, activation, initializer)

    model = xde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=C)
    variable = xde.callbacks.VariableValue(C, period=1000)
    loss_history, train_state = model.train(epochs=50000, callbacks=[variable])

    xde.saveplot(loss_history, train_state, issave=True, isplot=True)
