"""
Harmonic Oscillator: Implementation of a PINN to calculate the force acting on a spring when extended to a location 'x'
at time 't'. x and t are related as x = sin(t). We assume the underlying relation is given by Hooke's law as
F(x) = -kx - (0.1)sin(x) where k = 0.1 and compare the performance of a normal neural network with a
physics informed neural network in terms of predicting the force at different time instants.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

if __name__ == '__main__':

    tf.random.set_seed(468)

    """" Global parameters """
    EPOCHS = 50
    k = 1

    """" Define the harmonic oscillator process """

    def get_positions(time_instants):
        return np.sin(time_instants)

    def get_force(positions):
        return -k*positions + 0.1*np.sin(positions)

    t = np.linspace(0, 10, num=101)
    x = get_positions(t)
    F = get_force(x)
    plt.figure(1)
    plt.plot(t, F, 'b')
    plt.legend(["Actual force"])
    plt.xlabel("Time (t)")
    plt.ylabel("Force")
    plt.title("Force acting on the spring vs. time")
    plt.show()

    """" Collect data samples from the forward process """

    ts = [0, 3, 6.3, 9.3]
    xs = get_positions(ts)
    F_data = get_force(xs)
    plt.figure(2)
    plt.plot(t, F, 'b')
    plt.plot(ts, F_data, 'o', color="black")
    plt.legend(["Actual force", "Measured force"])
    plt.xlabel("Time (t)")
    plt.ylabel("Force")
    plt.title("Force acting on the spring vs. time")
    plt.show()

    """" Simple Network to predict force using only measured data """

    simple_model = Sequential()
    simple_model.add(Dense(32, activation="tanh"))
    simple_model.add(Dense(1))
    simple_model.compile(optimizer='adam', loss='mse')
    simple_model.fit(xs, F_data, epochs=EPOCHS)

    F_pred = simple_model.predict(x)
    plt.figure(3)
    plt.plot(t, F, 'b')
    plt.plot(ts, F_data, 'o', color="black")
    plt.plot(t, F_pred, '--', color="red")
    plt.legend(["Actual force", "Measured force", "NN prediction"])
    plt.xlabel("Time (t)")
    plt.ylabel("Force")
    plt.title("Force predicted by simple Neural Network")
    plt.show()

    """" Physics Informed Network to predict force using data and PDE loss """

    pinn_model = Sequential()
    pinn_model.add(Dense(32, activation="tanh"))
    pinn_model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam()

    def pinn_loss(x, F_act, F_pred):
        reg_param = 0.1
        mse_loss = tf.reduce_mean((F_act-F_pred)**2)
        temp = -k*x + 0.1*tf.sin(x)
        temp = tf.cast(temp, tf.float32)
        pde_loss = tf.reduce_mean((F_pred - temp)**2)
        return mse_loss + reg_param*pde_loss

    @tf.function
    def train_step(x, F_act):
        with tf.GradientTape() as pinn_tape:
            F_pinn = pinn_model(x, training=True)
            loss = pinn_loss(x, F_act, F_pinn)

        gradients = pinn_tape.gradient(loss, pinn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pinn_model.trainable_variables))

    for epoch in range(0, EPOCHS):
        for input, F_act in zip(xs, F_data):
            input = input.reshape((1, 1))
            F_act = np.float32(F_act)
            train_step(input, F_act)
    print("PINN training complete")
    F_pinn_pred = pinn_model.predict(x)

    """" Performance: NN vs. PINN """
    plt.figure(4)
    plt.plot(t, F, color="blue")
    plt.plot(ts, F_data, 'o', color="black")
    plt.plot(t, F_pred, '--', color="red")
    plt.plot(t, F_pinn_pred, '-', color="green")
    plt.legend(["Actual force", "Measured force", "NN prediction", "PINN prediction"])
    plt.xlabel("Time (t)")
    plt.ylabel("Force")
    plt.title("NN vs. PINN")
    plt.show()

    """" Extrapolation performance: NN vs. PINN """
    t_new = np.linspace(10, 20, num=101)
    x_new = get_positions(t_new)
    F_new_act = get_force(x_new)
    F_new_NN = simple_model.predict(x_new)
    F_new_PINN = pinn_model.predict(x_new)
    plt.figure(5)
    plt.plot(t_new, F_new_act, color="blue")
    plt.plot(t_new, F_new_NN, "--", color="red")
    plt.plot(t_new, F_new_PINN, '-', color="green")
    plt.legend(["Actual force", "NN prediction", "PINN prediction"])
    plt.xlabel("Time (t)")
    plt.ylabel("Force")
    plt.title("Extrapolation: NN vs. PINN")
    plt.show()
