# Snake Algorithm 
# By Hadjerci Mohammed Allaeddine


import cv2 
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from matplotlib import animation
import skimage.filters as filters


# Resources
# Active Contour Implementation

# We Minimize Internal Energy to smoothen the contour

# Active Contour Defining
# 1 - as First Methode we Can Use a list of points to define an Active contour
#     Internal energy is minimizing an approximation of the first and second derivitive of the contour ??

# 2 - Second methode is by using splines or parametric equations to represent the active controur
#     The internal energy of the active contour is then created from the constraints placed on it by the spline ??

# We will use seconde methode by defining a curve like a circle
# and then Use this equation
# E internal = integral(a, b) [ alpha * | lambda(s)' |^2 + beta * | lambda(s)'' |^2 ] ds
# lambda(s) is the equation of the curve
# i think the notion of the integral want to say that we need minimize the internal enrgy
# or a=b because of closed contour so we don't need to calculate integral hmmm


# All we need here is Declaring the curve (Curve Declared Below)
# This Curve will be modified by minimizing the internal Energy of this curve using Euler-Lagrange



#################################################################################################

# External Energy
# the external force pushes the active contour points to the edge of the object of intrest
# Marr method
# E external(x, y) = - | gradient [G(x, y) * I(x, y)]|^2
# I(x, y) is the image
# G(x, y) is a gaussian blur
# * is the 2d convolution operation
# gradient is vector gradient operation
# |.| magnitude of the vector 
# Note :
#       if we remove Gaussian blur the equation would be same
#       as the negative edge map

# Gaussian blur will spread out the sharp changes in gradient so that active contour points
# that are further away from the edge will be "pulled" toward that edge
# if the image is slightly blurry the edge can span over multiple pixels with a weaker gradient
# the gaussian blur will combine these gradients in such a way that their total energy will be near the same
# as sharp edge that has a large gradient between a few pixels

# now we can use this definition of the external energy to find an equation for the external force by
# taking the negative gradient of the external energy
# this will "push" the active contour points towards the minima, which is hopefully is an edge in the image
# F external(x, y) = gradient | gradient [G(x, y) * I(x, y)]|^2

# since the external forne is a vector it can be broken up into x and y
# F external X (x, y) is the x componenet of the external force vector
# F external Y (x, y) is the y componenet of the external force vector

# Note that Marr methode 
# the range is limited means in certain cases the external force would be near 0
# we can fix that by applying a wide gaussian blur

def MiniMaxNorm(img, mi, ma):
    return (img - mi) / (ma - mi)

def getExternalForce(img, sigma=30):

    # Image Normalization Minimax
    NormalizedImg = MiniMaxNorm(img, img.min(), img.max())

    # Gaussian Blur
    # We use the skimage library to calculate 
    # a gaussian with a size relative to sigma
    blurred = filters.gaussian(NormalizedImg , sigma )

    # Compute Gradtient "Derivative"
    Gx, Gy = np.gradient(blurred)

    # Compute magnitude of the image and Normalize it
    # good tip math.sqrt will not work cause it returns a scalar
    mag = (Gx**2 + Gy**2) ** 0.5
    # Normalization
    mag = MiniMaxNorm(mag, mag.min(), mag.max())

    # mag is the external energy
    # now we compute the externale force F external X and F external Y
    Gmagx, Gmagy = np.gradient(mag)

    return Gmagx, Gmagy


# Euler Lagrange Minimization
# Euler Lagrange Methode minimize energy using higher order derivatives
# In euler lagrange we have second and fourth order derivatives of the curve to compute
# to compute them we can use discrete derivative filters using pascal coefficients
# We can Obtain these coefficients using convolutions
# Paper : CS6640: Image Processing Final Project Active Contours Models

def getD2(N):
    t = np.zeros((N, N))
    row = np.r_[-2, 1, np.zeros(N-3), 1]
    for i in range(N):
        row = np.roll(row, 1)
        t[i] = row.copy()

    return t


def getD4(N):
    t = np.zeros((N, N))
    row = np.r_[-6, 4, -1, np.zeros(N-5), -1, 4]
    for i in range(N):
        row = np.roll(row, 1)
        t[i] = row.copy()
    return t


# https://www.crisluengo.net/archives/217
# this is the matrix that will help us minimize the internal energy
# Euler Lagrange
def getA(alpha, beta, N):
    return (alpha*getD2(N) + beta*getD4(N))



# interpolate to get pixels that are in line of the circle "BOUNDING BOX"
def interpolation(F, x, y):
    x[x < 0] = 0
    y[y < 0] = 0

    h, w = F.shape

    x[x > w-1] = w-1
    y[y > h-1] = h-1

    x = x.round().astype(int)
    y = y.round().astype(int)

    return F[y, x]


# HYPER PARAMETRES
alpha = 0.01
beta = 0.05
delta = 100
iterations = 500

# Circle Curve Coordinates
# we will minimize the internal energy of this circle
t = np.arange(0, 2*np.pi, 0.05)
x = 180+200*np.cos(t)
y = 160+200*np.sin(t)


##############################################################################################

# Snake Algorithm Steps
# 1 - Calculate External Energy this push the contour the edge
#     it is responsible for finding edges
# 2 - Declare the parametric Curve our bounding box or our snake
# 3 - Minimize this equation E = E int + E ext
#     E internal responsible for the modification of the shape of the snake
# 4 - To minimize we use Euler Lagrange

img = cv2.imread('lenna.jpg', 0)
A = getA(alpha, beta, x.shape[0])
A_inv = np.linalg.inv(np.eye(x.shape[0]) - delta*A)

snakes = []

Fx, Fy = getExternalForce(img, sigma=2.4)

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(img, cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, img.shape[1])
ax.set_ylim(img.shape[0], 0)

for i in range(iterations):
    newX = np.dot(A_inv, x + delta*interpolation(Fx, x, y))
    newY = np.dot(A_inv, y + delta*interpolation(Fy, x, y))

    x = newX.copy()
    y = newY.copy()

    snakes.append((x.copy(), y.copy()))


j = 0
snake = snakes[j]
line, = ax.plot(np.r_[snake[0], snake[0][0]], np.r_[snake[1], snake[1][0]], c=(0,0,1), lw=2)

j = 1

def animate(i):
    global j
    if (j < len(snakes)):
        snake = snakes[j]
        line.set_xdata(np.r_[snake[0], snake[0][0]])
        line.set_ydata(np.r_[snake[1], snake[1][0]])

        fig.canvas.draw()
        fig.canvas.flush_events()

        j+=1
    return im

anim = animation.FuncAnimation(fig, animate, interval=50)

plt.show()
