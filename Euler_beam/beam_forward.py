import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from time import time
from matplotlib.gridspec import GridSpec
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

# Set random seed for reproducible results
tf.random.set_seed(5)

# Set number of collocation points
N_c = 100

#Hyperparameters
N_h = 2 # number of hidden layers
N_n = 20 # number of neurons per layer
loss_fac = [1., 1.] # loss scaling factors: [PDE, BC]

# Set boundary
xmin = 0.0
xmax = 1.0

# Lower bounds
lb = tf.constant([xmin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([xmax], dtype=DTYPE)

# Boundary data
X_b = tf.constant([[xmin],[xmax]], dtype=DTYPE)

# Draw uniformly sampled collocation points
#X_c = tf.random.uniform((N_c,1), lb[0], ub[0], dtype=DTYPE)
X_c = tf.reshape(tf.linspace(lb[0],ub[0],N_c+2)[1:-1],[-1,1])


def beam_u(x):
    
    return tf.sin(np.pi*x)


def force(x,amp):
    
    return amp*tf.sin(np.pi*x)

# Define model architecture
class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self, lb, ub, 
            output_dim=1,
            num_hidden_layers=N_h, 
            num_neurons_per_layer=N_n,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lb = lb
        self.ub = ub
        
        # Define NN architecture
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        
    def call(self, X):
        """Forward-pass through neural network."""
        Z = self.scale(X)
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)
    
class PINNSolver():
    def __init__(self, model, X_c):
        self.model = model
        
        # Store data points
        self.x = X_c
        #force amplitude
        self.amp = np.pi**4
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.weights = []
        self.iter = 0
        
    def get_r(self):
        
        with tf.GradientTape() as tape1:
            tape1.watch(self.x)
            with tf.GradientTape() as tape2:
                tape2.watch(self.x)
                with tf.GradientTape() as tape3:
                    tape3.watch(self.x)
                    with tf.GradientTape() as tape4:
                        tape4.watch(self.x)
                        u = self.model(self.x)[:,0]
                    u_x = tape4.gradient(u, self.x)
                u_xx = tape3.gradient(u_x, self.x)
            u_xxx = tape2.gradient(u_xx, self.x)
        u_xxxx = tape1.gradient(u_xxx, self.x)

        res_PDE = u_xxxx - force(self.x, self.amp)
        
        return res_PDE
    
    def loss_fn(self, X):
        
        # Compute phi_r
        r_PDE = self.get_r()
        loss_PDE = tf.reduce_mean(tf.square(r_PDE))
        

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            with tf.GradientTape() as tape2:
                tape2.watch(X)    
                u = self.model(X)[:,0]
            u_x = tape2.gradient(u, X)
        u_xx = tape.gradient(u_x, X)
        loss_BC = tf.reduce_mean(tf.square(u[0])+tf.square(u[1])+tf.square(u_xx[0])+tf.square(u_xx[1]))
        del tape

        return loss_fac[0]*loss_PDE + loss_fac[1]*loss_BC
    
    def get_grad(self, X):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(X)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, g
    
    def solve_with_TFoptimizer(self, optimizer, X, N=1001):
        """This method performs a gradient descent type optimization."""
        
        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad(X)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss
        
        for i in range(N):
            
            loss = train_step()
            
            self.current_loss = loss.numpy()
            self.current_weights = self.model.get_weights()
            self.callback()
            
    def callback(self, xr=None):
        if self.iter % 100 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter,self.current_loss))
        self.hist.append(self.current_loss)
        self.weights.append(self.current_weights)
        self.iter+=1
        
    def plot_solution(self, **kwargs):
        N = 1000
        xspace = tf.linspace(self.model.lb[0], self.model.ub[0], N+1)
        xspace = tf.reshape (xspace, [-1,1])
        #choose model with minimum loss
        best_weights= np.argmin(self.hist)
        tf.print('minimum loss epoch = ',best_weights)
        self.model.set_weights(self.weights[best_weights])
        prediction = self.model(xspace)

        
        fig = plt.figure(figsize=(15,10))
        gs = GridSpec(2, 2, height_ratios=[0.8,1], width_ratios=[1, 1])
        
        ax4 = fig.add_subplot(gs[0,:])
        ax4.semilogy(range(len(self.hist)), self.hist,'k-')
        ax4.set_xlabel('$n_{epoch}$')
        ax4.set_ylabel('$\\phi^{n_{epoch}}$')
        ax4.set_title('min loss = %.1e' % np.amin(self.hist))
        
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(xspace, prediction[:,0],color = 'b', linestyle='dashed', label='PINN')
        ax1.plot(xspace,beam_u(xspace),color='r',label='analytic')
        ax1.legend()
        ax1.set_title('displacement')
        
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(xspace,force(xspace,1),color='r')
        ax2.set_title('normalized force')
        
        #ax3 = fig.add_subplot(gs[1, 2])
        #ax3.plot(xspace, prediction[:,2])
        #ax3.set_title('modulus')
        
        
        #fig.subplots_adjust(wspace=0.05)
        
        plt.show()
        

# Initialize model
model = PINN_NeuralNet(lb, ub)
#model.build(input_shape=(None,1))

# Initilize PINN solver
solver = PINNSolver(model, X_c)

# Start timer
t0 = time()

#lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([950,970],[1e-2,1e-3,5e-4])
optim = tf.keras.optimizers.Adam(learning_rate=3e-3)
#optim = tf.keras.optimizers.Adam()
solver.solve_with_TFoptimizer(optim, X_b, N=5000)


# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))

solver.plot_solution();