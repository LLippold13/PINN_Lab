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
tf.random.set_seed(7)

#network parameters
N_h = 2     # number of hidden layers
N_n = 20    # number of hidden layer neurons
loss_fac = [1., 1., 1e4] # loss scaling factors

# Set number of data/ collocation points
N_d = 10

# Set boundary
xmin = 0.0
xmax = 1.0

# Lower bounds
lb = tf.constant([xmin], dtype=DTYPE)
# Upper bounds
ub = tf.constant([xmax], dtype=DTYPE)

# Boundary data
X_data = tf.constant([[xmin],[xmax]], dtype=DTYPE)

# 
#X_r = tf.random.uniform((N_d,1), lb[0], ub[0], dtype=DTYPE)
X_r = tf.reshape(tf.linspace(lb[0],ub[0],N_d+2)[1:-1],[-1,1])


# =============================================================================
# Defining an the analytic solutions for a beam with linear elasticity
# =============================================================================
def beam_u(x):
    
    return 0.5*tf.math.log(x+1)

def beam_eps(x):
    
    return 0.5/(x+1)

def beam_E(x):
    
    return 20*x + 10*(1-x)

# Define model architecture
class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self, lb, ub, 
            output_dim=2,
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
        # scale input coordinate to tanh range
        self.scale_in = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        #scaling output to expected magnitude
        self.scale_out = tf.keras.layers.Lambda(lambda x: tf.multiply(x, [0.5,10]))
    
    def call(self, X):
        """Forward-pass through neural network."""
        Z = self.scale_in(X)
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        Z = self.out(Z)
        
        return self.scale_out(Z)
    
class PINNSolver():
    
    def __init__(self, model, X_r):
        self.model = model
        
        # Store data points
        self.x = X_r[:,0:1]
        
        #stress BC at x=1
        self.sig_0 = 5.
        
        # Initialize history of losses, weights and global iteration counter
        self.hist = [] # training progress of total loss
        self.losses = [] # training progress of individual loss contributions
        self.weights = [] 
        self.iter = 0
        
    def get_r(self):
        
        with tf.GradientTape(persistent=True) as tape:
            
            # Computing derivatives for PDE residual
            tape.watch(self.x)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(self.x)
            
                pred = self.model(self.x)
                u = pred[:,0]
                E = pred[:,1]
            u_x = tape2.gradient(u, self.x)
            E_x = tape2.gradient(E, self.x)
        u_xx = tape.gradient(u_x, self.x)

        res_data = u_x - beam_eps(self.x) # data residual
        res_PDE = E_x*u_x + E*u_xx        # PDE residual
        
        del tape
        del tape2
        
        return res_PDE, res_data
    
    def loss_fn(self, X):
        
        # Computing loss contributions
        
        res_PDE, res_data = self.get_r()
        loss_PDE = tf.reduce_mean(tf.square(res_PDE))
        loss_data = tf.reduce_mean(tf.square(res_data))
        loss_BC = 0
        
        # Computing BC loss

        with tf.GradientTape() as tape3:
            tape3.watch(X)
            pred_b = self.model(X)
            u = pred_b[:,0]
            E = pred_b[:,1]
        u_x = tape3.gradient(u, X)
        loss_BC += tf.reduce_mean(tf.square(u[0])+tf.square(E[1]*u_x[1]-self.sig_0)) # two boundary conditions on left and right end of beam

        return loss_fac[0]*loss_PDE, loss_fac[1]*loss_BC, loss_fac[2]*loss_data
    
    def get_grad(self, X):
        with tf.GradientTape(persistent=True) as tape:

            tape.watch(self.model.trainable_variables)
            losses = self.loss_fn(X) 
            total_loss = losses[0] + losses[1] + losses[2]
            
        g = tape.gradient(total_loss, self.model.trainable_variables)
        del tape
        
        return total_loss, g, losses
    
    def solve_with_TFoptimizer(self, optimizer, X, N=1001):
        """This method performs a gradient descent type optimization."""
        
        @tf.function
        def train_step():
            loss, grad_theta, losses = self.get_grad(X)
            
            # Perform gradient descent step
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, losses
        
        for i in range(N): 
            
            loss, losses = train_step()
            
            self.current_loss = loss.numpy()
            self.current_losses = np.array([losses[0].numpy(),losses[1].numpy(),losses[2].numpy()])
            self.current_weights = self.model.get_weights()
            self.callback()
            
    def callback(self, xr=None):
        if self.iter % 100 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter,self.current_loss))
        
        # storing values
        self.hist.append(self.current_loss)
        self.losses.append(self.current_losses)
        self.weights.append(self.current_weights)
        self.iter+=1
        
    def plot_solution(self, **kwargs):
        N = 1000
        xspace = tf.linspace(self.model.lb[0], self.model.ub[0], N+1)
        xspace = tf.reshape (xspace, [-1,1])
        best_weights= np.argmin(self.hist) #choose model with minimum loss
        tf.print('minimum loss epoch = ',best_weights)
        self.model.set_weights(self.weights[best_weights])
        with tf.GradientTape() as tape4:
            tape4.watch(xspace)
            prediction = self.model(xspace)
            u = prediction[:,0]
        u_x = tape4.gradient(u, xspace)
        
        fig = plt.figure(figsize=(15,10))
        gs = GridSpec(2, 3, height_ratios=[1,1], width_ratios=[1, 1, 1])
        
        ax4 = fig.add_subplot(gs[0,0:2])
        ax4.loglog(range(len(self.hist)), self.hist,'k-', label = 'total')
        ax4.loglog(range(len(self.hist)), np.array(self.losses)[:,0],'r-', alpha=0.5, label='PDE')
        ax4.loglog(range(len(self.hist)), np.array(self.losses)[:,1],'b-', alpha=0.5, label='BC')
        ax4.loglog(range(len(self.hist)), np.array(self.losses)[:,2],'g-', alpha=0.5, label='data')
        ax4.set_xlim(100,)
        ax4.set_ylim(None,np.amax(self.hist[100:]))
        ax4.set_xlabel('$n_{epoch}$')
        ax4.set_ylabel('$\\phi^{n_{epoch}}$')
        ax4.set_title('min loss = %.1e' % np.amin(self.hist))
        ax4.legend()
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(xspace, u_x, color = 'royalblue', linestyle='dashed', label='PINN')
        ax2.plot(xspace, beam_eps(xspace),color='crimson',label='analytic')
        ax2.set_title('strain')
        
        ax1 = fig.add_subplot(gs[0, 2])
        ax1.plot(xspace, prediction[:,0],color = 'royalblue', linestyle='dashed', label='PINN')
        ax1.plot(xspace,beam_u(xspace),color='crimson',label='analytic')
        ax1.legend()
        ax1.set_title('displacement')
        
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.plot(xspace,prediction[:,1]*u_x[:,0],color = 'royalblue', linestyle='dashed', label='PINN')
        ax3.axhline(y=5.,color='crimson')
        ax3.set_title('stress')
        
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(xspace, prediction[:,1], color = 'royalblue', linestyle='dashed', label='PINN')
        ax2.plot(xspace,beam_E(xspace),color='crimson',label='analytic')
        ax2.set_title('Young modulus')

        #fig.subplots_adjust(wspace=0.05)
        
        plt.show()
        

# Initialize model
model = PINN_NeuralNet(lb, ub)
#model.build(input_shape=(None,1))

# Initilize PINN solver
solver = PINNSolver(model, X_r)

# Start timer
t0 = time()

#lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([950,970],[1e-2,1e-3,5e-4])
optim = tf.keras.optimizers.Adam(learning_rate=3e-3)
#optim = tf.keras.optimizers.Adam()
solver.solve_with_TFoptimizer(optim, X_data, N=2000)


# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))

solver.plot_solution();