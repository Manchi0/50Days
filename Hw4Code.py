import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def rand_weight_bias(inputL,next_layer): # list with weights
    weights=np.random.randn(inputL, next_layer) * 0.01
    bias=np.zeros((next_layer,))
    return weights, bias


# 1. Load and normalize the MNIST dataset using PyTorch
def load_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.view(-1))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    x_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0
    x_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
    return x_train, x_test

# Load data
x_train, x_test = load_data()

# 2. Initialize weights and biases (Students need to implement)
W1, b1 = rand_weight_bias(784, 512)
W2, b2 = rand_weight_bias(512, 256)
W3, b3 = rand_weight_bias(256, 128)
W4, b4 = rand_weight_bias(128, 256)
W5, b5 = rand_weight_bias(256, 512)
W6, b6 = rand_weight_bias(512, 784)

# 3. Define activation functions (Students need to implement)
def relu(x):
    return np.maximum(0,x)

def diff_relu(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    x = np.clip(x, -500, 500)  # Clip input to avoid overflow
    return 1/(1+np.exp(-x))

def diff_sigmoid(x,sigx):
    sig = sigx
    return sig * (1 - sig)

# 4. Implement forward pass (Encoder + Decoder) - Students need to implement
def forward(x):
    # Encoder: multiple layers with ReLU activation
    h1_1b = np.dot(x, W1) + b1
    h1_1a = relu(h1_1b)

    h1_2b = np.dot(h1_1a, W2) + b2
    h1_2a = relu(h1_2b)

    z_b = np.dot(h1_2a, W3) + b3
    z = relu(z_b)

    # Decoder: Multiple layers with ReLU/Sigmoid activation for reconstruction
    h2_1b = np.dot(z, W4) + b4
    h2_1a = relu(h2_1b)

    h2_2b = np.dot(h2_1a, W5) + b5
    h2_2a = relu(h2_2b)

    x_reconstructed_b = np.dot(h2_2a, W6) + b6
    x_reconstructed = sigmoid(x_reconstructed_b)

    # Output the encoded representation and the reconstructed input
    # as well as intermediate activations for layer output tracking
    #print(x.shape, z.shape, x_reconstructed.shape, h1_1a.shape, h1_2a.shape, h2_1a.shape, h2_2a.shape)
    # Expected shapes: (60000, 784), (60000, 128), (60000, 784), (60000, 512), (60000, 256), (60000, 256), (60000, 512)
    return z, x_reconstructed, h1_1a, h1_2a, h2_1a, h2_2a, h1_1b,h1_2b,h2_1b,h2_2b,x_reconstructed_b,z_b


# 5. Compute Mean Squared Error Loss (Students need to implement)
def mse_loss(x, x_reconstructed):
    return np.mean((x-x_reconstructed)**2)

# 6. Implement backpropagation and weight updates - Students need to implement
def backward(x, h1_1, h1_2, z, h2_1, h2_2, x_reconstructed,h1_1b,h1_2b,h2_1b,h2_2b, x_reconstructed_b,z_b,sigx,lr):
    global W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6
    # Compute gradients and update weights and biases
    error= 2*(x_reconstructed-x)*diff_sigmoid(x,sigx)/ x.shape[0]
    d_w6=np.dot(h2_2.T, error)
    d_b6=np.sum(error,axis=0)

    error_l5=np.dot(error,W6.T)*diff_relu(h2_2b)
    d_w5=np.dot(h2_1.T,error_l5)
    d_b5=np.sum(error_l5,axis=0)

    error_l4=np.dot(error_l5,W5.T)*diff_relu(h2_1b)
    d_w4=np.dot(z.T, error_l4)
    d_b4=np.sum(error_l4,axis=0)

    error_l3=np.dot(error_l4,W4.T)*diff_relu(z_b)
    d_w3=np.dot(h1_2.T, error_l3)
    d_b3=np.sum(error_l3,axis=0)

    error_l2=np.dot(error_l3,W3.T)*diff_relu(h1_2b)
    d_w2=np.dot(h1_1.T, error_l2)
    d_b2=np.sum(error_l2,axis=0)

    error_l1=np.dot(error_l2,W2.T)*diff_relu(h1_1b)
    d_w1=np.dot(x.T, error_l1)
    d_b1=np.sum(error_l1,axis=0)        

    W1 -= d_w1*lr
    b1 -= d_b1*lr
    W2 -= d_w2*lr
    b2 -= d_b2*lr
    W3 -= d_w3*lr
    b3 -= d_b3*lr
    W4 -= d_w4*lr
    b4 -= d_b4*lr
    W5 -= d_w5*lr
    b5 -= d_b5*lr
    W6 -= d_w6*lr
    b6 -= d_b6*lr


# 7. Implement training loop - Students need to implement
def train(x_train, epochs=500, lr=0.01):
    # Train the network and collect loss and latent codes
    losses=[]
    sigx=sigmoid(x_train)
    for epoch in range(epochs):
        z,x_reconstructed,h1_1,h1_2,h2_1,h2_2,h1_1b,h1_2b,h2_1b,h2_2b,x_reconstructed_b,z_b=forward(x_train)
        losses.append(mse_loss(x_train,x_reconstructed))
        backward(x_train,h1_1,h1_2,z,h2_1,h2_2,x_reconstructed,h1_1b,h1_2b,h2_1b,h2_2b,x_reconstructed_b,z_b,sigx,lr)
        if epoch%5==0:
            print(f'Epoch: {epoch}, Loss: {losses[-1]:.4f}')




    return losses, z, x_reconstructed

# Train the model (Uncomment once students complete the train function)
losses, latent_codes, reconstructed_data = train(x_train)

# 8. Plot the loss curve
def plot_loss_curve(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label='MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

# Uncomment after training to plot the loss curve
plot_loss_curve(losses)

# 9. Visualize latent codes using PCA
def visualize_latent_space(latent_codes):
    pca = PCA(n_components=2)
    reduced_codes = pca.fit_transform(latent_codes)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_codes[:, 0], reduced_codes[:, 1], s=5, alpha=0.6)
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# Uncomment after training to visualize latent space
visualize_latent_space(latent_codes)

# 10. Show original and reconstructed images side by side
def show_images(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

# Uncomment after training to show images
show_images(x_train, reconstructed_data)
