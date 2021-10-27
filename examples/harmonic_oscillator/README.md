Python implementation of the example mentioned in https://mitmath.github.io/18337/lecture3/sciml.html

The example describes a harmonic oscillator which is a system that when displaced from its original position will experience a restoring force `F` proportional to the displacement `x` as `F = -kx`. A Physics Informed Neural Network is implemented to predict the amount of force acting on a spring when it is stretched by a distance `x`. In this case the force is represented as `F(x) = -kx + 0.1*sin(x)`.

The actual force acting on the spring as a function of time `t` can be given as:

<img src="https://user-images.githubusercontent.com/5306916/139063290-6d490e49-542c-49c0-ae11-62b3486674b6.png" width="350" height="250">

In order to estimate this function using a neural network, we select 4 data points on this curve, which more of less lie around zero.

<img src="https://user-images.githubusercontent.com/5306916/139063346-ca2a5130-bd31-4f9e-bf0a-cb7ed79ab766.png" width="350" height="250">

A simple neural network minimizing the mean squared error of a curve fitted to these points ends up undershooting at the peaks of the curve due to lack of data for those points.

<img src="https://user-images.githubusercontent.com/5306916/139063356-8b12be5d-c0b6-4352-87b4-6f52dd50401d.png" width="350" height="250">

However, a PINN implementation where the mean squared error loss is regularized with another loss term representing the error in the output as a solution of the PDE leads the function to fit well even at the peaks in spite of lack of data.

<img src="https://user-images.githubusercontent.com/5306916/139063365-fef6e7ea-23ff-49b2-8ccc-8e12bc16fa9b.png" width="350" height="250">
