#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse Euler beam example

Governing equation: u_{xxxx}(x) = f(x) for x \in (0,1)

Inputs: u(x) observed at equally spaced data points
Outputs: u(x) at arbitrary points
         f(x) at arbitrary points
         
"""
import tensorflow as tf
import numpy as np
import time
from utils.tfp_loss import tfp_function_factory
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
tf.random.set_seed(42)

class model(tf.keras.Model): 
    def __init__(self, loss_weights, layers_u, layers_f, train_op, num_epoch,
                 print_epoch, data_type):
        super(model, self).__init__()
        self.model_layers_u = layers_u
        self.model_layers_f = layers_f
        self.train_op = train_op
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.loss_weights = loss_weights
        self.adam_loss_hist = []

            
    def call(self, X):
        return tf.concat((self.u(X), self.force(X)), axis=1)
    
    # Running the model       
    def u(self, X):
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers_u:
            X = l(X)
        return X
    
    def force(self, X):
        X = 2.0*(X - self.bounds["lb"])/(self.bounds["ub"] - self.bounds["lb"]) - 1.0
        for l in self.model_layers_f:
            X = l(X)
        return X
    
    # Return the first derivative
    def du(self, X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            u_val = self.u(X)
        du_val = tape.gradient(u_val, X)
        return du_val
    
    # Return the second derivative
    def d2u(self, X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            du_val = self.du(X)
        d2u_val = tape.gradient(du_val, X)
        return d2u_val
    
    # Return the third derivative
    def d3u(self, X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            d2u_val = self.d2u(X)
        d3u_val = tape.gradient(d2u_val, X)
        return d3u_val
    
    # Return the fourth derivative
    def d4u(self, X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            d3u_val = self.d3u(X)
        d4u_val = tape.gradient(d3u_val, X)
        return d4u_val
             
    #Custom loss function
    def get_all_losses(self, Xint, Xbnd0, Ybnd0, Xbnd1, Ybnd1, Xdata, Ydata):
        u_val_bnd = self.u(Xbnd0)
        d2u_val_bnd = self.force(Xbnd1)
        
        d4u_val_int = self.d4u(Xint)
        f_val_int = self.force(Xint)
        
        u_val_int = self.u(Xdata)
        
        int_loss = tf.reduce_mean(tf.math.square(d4u_val_int - f_val_int))
        bnd_loss = tf.reduce_mean(tf.math.square(u_val_bnd - Ybnd0))
        bnd_loss += tf.reduce_mean(tf.math.square(d2u_val_bnd - Ybnd1))
        data_loss = tf.reduce_mean(tf.math.square(u_val_int - Ydata))
        
        
        return int_loss, bnd_loss, data_loss
    
    def get_loss(self, Xint, Xbnd0, Ybnd0, Xbnd1, Ybnd1, Xdata, Ydata):
        int_loss, bnd_loss, data_loss = self.get_all_losses(Xint, Xbnd0, 
                                                            Ybnd0, Xbnd1, Ybnd1, 
                                                            Xdata, Ydata)
        return self.loss_weights[0]*int_loss + self.loss_weights[1]*bnd_loss \
            + self.loss_weights[2]*data_loss
      
    # get gradients
    def get_grad(self, Xint, Xbnd0, Ybnd0, Xbnd1, Ybnd1, Xdata, Ydata):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(Xint, Xbnd0, Ybnd0, Xbnd1, Ybnd1, Xdata, Ydata)
        g = tape.gradient(L, self.trainable_variables)
        return L, g
    
    @tf.function
    def train_step(self, Xint, Xbnd0, Ybnd0, Xbnd1, Ybnd1, Xdata, Ydata):
        loss, grad_theta = self.get_grad(Xint, Xbnd0, Ybnd0, Xbnd1, Ybnd1, Xdata, Ydata)
        
        # Perform gradient descent step
        self.train_op.apply_gradients(zip(grad_theta, self.trainable_variables))
        return loss
      
    # perform gradient descent
    def network_learn(self, Xint, Xbnd0, Ybnd0, Xbnd1, Ybnd1, Xdata, Ydata):
        self.bounds = {"lb" : tf.math.reduce_min(Xint),
                       "ub" : tf.math.reduce_max(Xint)}
        for i in range(self.num_epoch):
            L = self.train_step(Xint, Xbnd0, Ybnd0, Xbnd1, Ybnd1, Xdata, Ydata)
            #self.adam_loss_hist = self.adam_loss_hist[i].assign(L)
            self.adam_loss_hist.append(L)
            if i%self.print_epoch==0:
                tf.print("Epoch {} loss: {}".format(i, L))

 
if __name__ == "__main__":
    
    #define the force function f(x)
    k = 1
    def rhs_fun(x):
        f = k**4*np.pi**4*np.sin(k*np.pi*x)
        return f
    
    def exact_sol(x):
        y = np.sin(k*np.pi*x)
        return y
    
    def second_deriv_exact_sol(x):
        output = -k**2*np.pi**2*np.sin(k*np.pi*x)
        return output
    
    #define the input and output data set
    xmin = 0.
    xmax = 1.
    numIntPts = 100
    numDataPts = 10
    data_type = "float64"
    
    weight_interior = 1.
    weight_boundary = 5.
    weight_data = 100.
    loss_weights = [weight_interior, weight_boundary, weight_data]
    
    
    # data for the interior (collocation points)
    Xint = np.linspace(xmin, xmax, numIntPts+2)[1:-1].astype(data_type)
    Xint = np.array(Xint)[np.newaxis].T
    Yint = rhs_fun(Xint)
    
    # data for the boundary
    Xbnd0 = np.array([[xmin],[xmax]]).astype(data_type)
    Ybnd0 = exact_sol(Xbnd0)
    
    Xbnd1 = np.array([[xmin],[xmax]]).astype(data_type)
    Ybnd1 = second_deriv_exact_sol(Xbnd1)
    
    # observed data
    Xdata = np.linspace(xmin, xmax, numDataPts+2)[1:-1].astype(data_type)
    Xdata = np.array(Xdata)[np.newaxis].T
    Ydata = exact_sol(Xdata)
    
    #define the model 
    tf.keras.backend.set_floatx(data_type)
    lu1 = tf.keras.layers.Dense(10, "swish")
    lu2 = tf.keras.layers.Dense(10, "swish")
    lu3 = tf.keras.layers.Dense(1, None)
    
    lf1 = tf.keras.layers.Dense(10, "swish")
    lf2 = tf.keras.layers.Dense(10, "swish")
    lf3 = tf.keras.layers.Dense(1, None)
    
    train_op = tf.keras.optimizers.Adam()
    num_epoch = 5000
    print_epoch = 100
    pred_model = model(loss_weights, [lu1, lu2, lu3], [lf1, lf2, lf3], train_op,
                       num_epoch, print_epoch, data_type)
    
    #convert the training data to tensors
    Xint_tf = tf.convert_to_tensor(Xint[1:-1])
    Yint_tf = tf.convert_to_tensor(Yint[1:-1])
    Xbnd0_tf = tf.convert_to_tensor(Xbnd0)
    Ybnd0_tf = tf.convert_to_tensor(Ybnd0)
    Xbnd1_tf = tf.convert_to_tensor(Xbnd1)
    Ybnd1_tf = tf.convert_to_tensor(Ybnd1)
    Xdata_tf = tf.convert_to_tensor(Xdata)
    Ydata_tf = tf.convert_to_tensor(Ydata)
    
   
    #training
    print("Training (ADAM)...")
    t0 = time.time()
    pred_model.network_learn(Xint_tf, Xbnd0_tf, Ybnd0_tf, 
                             Xbnd1_tf, Ybnd1_tf, Xdata_tf, Ydata_tf)
    t1 = time.time()
    print("Time taken (ADAM)", t1-t0, "seconds")
    print("Training (LBFGS)...")
    
    loss_func = tfp_function_factory(pred_model, Xint_tf, Xbnd0_tf, 
                                     Ybnd0_tf, Xbnd1_tf, Ybnd1_tf, Xdata_tf, Ydata_tf)

    # convert initial model parameters to a 1D tf.Tensor
    init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)#.numpy()
    # train the model with L-BFGS solver
    results = tfp.optimizer.bfgs_minimize(
        value_and_gradients_function=loss_func, initial_position=init_params,
              max_iterations=5000, tolerance=1e-14)  
 
    # after training, the final optimized parameters are still in results.position
    # so we have to manually put them back to the model
    loss_func.assign_new_model_parameters(results.position)

    t2 = time.time()
    print("Time taken (LBFGS)", t2-t1, "seconds")
    print("Time taken (all)", t2-t0, "seconds")
    print("Testing...")
    numPtsTest = 1001
    x_test = np.linspace(xmin, xmax, numPtsTest).astype(data_type)
    x_test = np.array(x_test)[np.newaxis].T
    x_tf = tf.convert_to_tensor(x_test)

    y_test = pred_model.u(x_tf)    
    y_exact = exact_sol(x_test)
    f_test = pred_model.d4u(x_tf)

    plt.plot(x_test, y_test, label = 'Predicted')
    plt.plot(x_test, y_exact, label = 'Exact')
    plt.legend()
    plt.show()
    plt.plot(x_test, y_exact-y_test)
    plt.title("Error for displacement")
    plt.show()
    err = y_exact - y_test
    print("L2-error norm: {}".format(np.linalg.norm(err)/np.linalg.norm(y_exact)))
    
    f_exact = rhs_fun(x_test)
    plt.plot(x_test, f_test, label = 'Predicted')
    plt.plot(x_test, f_exact, label = 'Exact')
    plt.legend()
    plt.title("Force")
    plt.show()
    
    err_force = f_exact - f_test
    plt.plot(x_test, err_force)
    plt.title("Error for force")
    plt.show()
    print("Relative error for force: {}".format(np.linalg.norm(err_force)/np.linalg.norm(f_exact)))
    
    # plot the loss convergence
    num_iter_bfgs = len(loss_func.history)
    plt.semilogy(range(num_epoch), pred_model.adam_loss_hist, label='Adam')
    plt.semilogy(range(num_epoch, num_epoch+num_iter_bfgs), loss_func.history, 
                 label = 'BFGS')
    plt.legend()
    plt.title('Loss convergence')
    plt.show()