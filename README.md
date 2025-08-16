# Handwritten Digit Recognition Neural Network

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

*A neural network built from scratch to recognize handwritten digits with 91.27% accuracy*

[![Demo](https://img.shields.io/badge/Live-Demo-red?style=for-the-badge&logo=github)](https://github.com/yourusername/simple-nueral-network-used-to-recognize-handwritten-digits)

</div>

## 🎯 Project Overview

This project implements a **feedforward neural network from scratch** to recognize handwritten digits from the MNIST dataset. Unlike using pre-built frameworks, this implementation demonstrates deep understanding of neural network fundamentals by building the forward propagation, backpropagation, and training algorithms manually using NumPy.

### Why This Project?

- **Educational Value**: Built entirely from scratch to understand neural network internals
- **Performance**: Achieves 91.27% accuracy on test data without using pre-built layers
- **Scalability**: Demonstrates efficient matrix operations for large-scale computations
- **Real-world Application**: Handwritten digit recognition is a fundamental computer vision problem

## 🚀 Features

- **From-Scratch Implementation**: No pre-built neural network layers used
- **Manual Backpropagation**: Custom gradient descent implementation
- **Binary Image Processing**: Converts grayscale images to binary for better feature extraction
- **Real-time Training Visualization**: Live accuracy monitoring during training
- **Efficient Matrix Operations**: Optimized NumPy operations for performance

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 91.27% |
| **Training Samples** | 60,000 |
| **Test Samples** | 10,000 |
| **Network Architecture** | 784 → 784 → 10 |
| **Activation Function** | Sigmoid |

## 🏗️ Architecture

```
Input Layer (784 neurons) → Hidden Layer (784 neurons) → Output Layer (10 neurons)
     ↓                           ↓                           ↓
28×28 pixels              Sigmoid activation         Digit classification (0-9)
```

### Key Components:

1. **Image Preprocessing**: Converts 28×28 grayscale images to binary (0/1) format
2. **Forward Propagation**: Matrix multiplication with sigmoid activation
3. **Backpropagation**: Manual gradient calculation and weight updates
4. **Loss Function**: Mean squared error for training optimization

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/simple-nueral-network-used-to-recognize-handwritten-digits.git
   cd simple-nueral-network-used-to-recognize-handwritten-digits
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow numpy matplotlib ipython
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook handwritten-digit-recognition-neural-network.ipynb
   ```

### Alternative: Direct Python Execution

```bash
python -c "
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x1, y1), (x2, y2) = mnist.load_data()

print('Dataset loaded successfully!')
print(f'Training samples: {len(x1)}')
print(f'Test samples: {len(x2)}')
"
```

## 📖 Usage

### Training the Network

The training process is implemented in the Jupyter notebook with the following key steps:

1. **Data Loading**: MNIST dataset is automatically downloaded
2. **Weight Initialization**: Random weights between -0.1 and 0.1
3. **Training Loop**: Forward and backward propagation for each sample
4. **Accuracy Testing**: Real-time accuracy calculation on test set

### Key Functions

```python
# Convert image to binary array
def img_to_array(x):
    res = []
    for j in x:
        for k in j:
            res.append(0 if k <= 128 else 1)
    return res

# Sigmoid activation function
def sigmoid(x):
    return [1/(1+np.exp(-j)) for i in x for j in i]

# Create one-hot encoded target vector
def create_vector(x):
    c = np.array([0.0]*10).reshape(10,1)
    c[x] = 1.0
    return c
```

## 🔬 Technical Implementation

### Forward Propagation
```python
xi = np.array(img_to_array(i)).reshape(784,1)
xh = np.array(sigmoid(np.matmul(theta1,xi))).reshape(784,1)
xo = np.array(sigmoid(np.matmul(theta2,xh))).reshape(10,1)
```

### Backpropagation
```python
error1 = np.subtract(xo, yo)
error2 = np.matmul(theta2.transpose(), error1)
delta1 = np.matmul(np.dot(np.multiply(np.multiply(xo,(1-xo)),error1),0.1),xh.transpose())
delta2 = np.matmul(np.dot(np.multiply(np.multiply(xh,(1-xh)),error2),0.1),xi.transpose())
```

## 📈 Results & Analysis

The network achieves **91.27% accuracy** on the MNIST test set, which is impressive for a from-scratch implementation. Key observations:

- **Convergence**: Network learns effectively through manual backpropagation
- **Generalization**: Good performance on unseen test data
- **Efficiency**: Matrix operations enable fast training despite large dataset

## 🎓 Learning Outcomes

This project demonstrates:

- **Deep Learning Fundamentals**: Understanding of neural network mathematics
- **Matrix Operations**: Efficient NumPy implementations
- **Gradient Descent**: Manual implementation of backpropagation
- **Computer Vision**: Image preprocessing and feature extraction
- **Performance Optimization**: Balancing accuracy with computational efficiency

## 🔧 Customization

### Modifying Network Architecture

```python
# Change hidden layer size
theta1 = np.random.uniform(-0.1, 0.1, (784, 400))  # 784 → 400
theta2 = np.random.uniform(-0.1, 0.1, (10, 400))   # 400 → 10
```

### Adjusting Learning Rate

```python
# Modify learning rate in backpropagation
learning_rate = 0.01  # Default: 0.1
delta1 = np.matmul(np.dot(np.multiply(np.multiply(xo,(1-xo)),error1),learning_rate),xh.transpose())
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: The classic handwritten digit dataset
- **TensorFlow**: For providing easy access to MNIST data
- **NumPy**: For efficient matrix operations
- **Matplotlib**: For visualization capabilities

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)
- **Email**: your.email@example.com

---

<div align="center">

⭐ **Star this repository if you found it helpful!** ⭐

*Built with ❤️ and ☕*

</div>
