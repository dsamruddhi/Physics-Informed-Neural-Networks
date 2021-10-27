Python implementation of the example mentioned in https://mitmath.github.io/18337/lecture3/sciml.html

A Physics Informed Neural Network is implemented to predict the amount of force acting on a spring when it is stretched by a distance `x`. According to Hooke's law, `F(x) = -kx` where `k` is a characteristic of the spring. 

The actual force acting on the spring as a function of time `t` can be given as:

<img src="https://user-images.githubusercontent.com/5306916/139063290-6d490e49-542c-49c0-ae11-62b3486674b6.png" width="400" height="300">

In order to estimate this function using a neural network, we select 4 data points on this curve, which more of less lie around zero.

<img src="https://user-images.githubusercontent.com/5306916/139063346-ca2a5130-bd31-4f9e-bf0a-cb7ed79ab766.png" width="400" height="300">

A simple neural network minimizing the mean squared error of a curve fitted to these points ends up undershooting at the peaks of the curve due to lack of data for those points.

<img src="https://user-images.githubusercontent.com/5306916/139063356-8b12be5d-c0b6-4352-87b4-6f52dd50401d.png" width="400" height="300">

However, a PINN implementation where the mean squared error loss is regularized with another loss term representing the error in the output as a solution of the PDE leads the function to fit well even at the peaks in spite of lack of data.

<img src="https://user-images.githubusercontent.com/5306916/139063365-fef6e7ea-23ff-49b2-8ccc-8e12bc16fa9b.png" width="400" height="300">
