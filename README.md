<!-- Improved compatibility of back to top link: See: https://github.com/dhmnr/skipr/pull/73 -->
<a id="readme-top"></a>

<!-- *** Thanks for checking out the Best-README-Template. If you have a suggestion *** that would make this better, please fork the repo and create a pull request *** or simply open an issue with the tag "enhancement". *** Don't forget to give the project a star! *** Thanks again! Now go create something AMAZING! :D -->

<!-- PROJECT SHIELDS -->
<!-- *** I'm using markdown "reference style" links for readability. *** Reference links are enclosed in brackets [ ] instead of parentheses ( ). *** See the bottom of this document for the declaration of the reference variables *** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use. *** https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">🔢 Handwritten Digit Recognition Neural Network</h3>

  <p align="center">
    A neural network built from scratch to recognize handwritten digits with 91.27% accuracy using the MNIST dataset.
    <br />
    <a href="https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits">View Demo</a>
    ·
    <a href="https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project implements a **feedforward neural network from scratch** to recognize handwritten digits from the MNIST dataset. Unlike using pre-built frameworks, this implementation demonstrates deep understanding of neural network fundamentals by building the forward propagation, backpropagation, and training algorithms manually using NumPy.

### Why This Project?

- **Educational Value**: Built entirely from scratch to understand neural network internals
- **Performance**: Achieves 91.27% accuracy on test data without using pre-built layers
- **Scalability**: Demonstrates efficient matrix operations for large-scale computations
- **Real-world Application**: Handwritten digit recognition is a fundamental computer vision problem

### Key Features

- **From-Scratch Implementation**: No pre-built neural network layers used
- **Manual Backpropagation**: Custom gradient descent implementation
- **Binary Image Processing**: Converts grayscale images to binary for better feature extraction
- **Real-time Training Visualization**: Live accuracy monitoring during training
- **Efficient Matrix Operations**: Optimized NumPy operations for performance

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 91.27% |
| **Training Samples** | 60,000 |
| **Test Samples** | 10,000 |
| **Network Architecture** | 784 → 784 → 10 |
| **Activation Function** | Sigmoid |

### Architecture

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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [Python 3.8+](https://www.python.org/downloads/)
* [TensorFlow 2.x](https://tensorflow.org/)
* [NumPy 1.21+](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Jupyter Notebook](https://jupyter.org/)
* [IPython](https://ipython.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Python 3.8 or higher
* pip package manager

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits.git
   ```
2. Navigate to the project directory
   ```sh
   cd simple-nueral-network-used-to-recognize-handwritten-digits
   ```
3. Install dependencies
   ```sh
   pip install tensorflow numpy matplotlib ipython
   ```
4. Run the notebook
   ```sh
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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

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

### Technical Implementation

#### Forward Propagation
```python
xi = np.array(img_to_array(i)).reshape(784,1)
xh = np.array(sigmoid(np.matmul(theta1,xi))).reshape(784,1)
xo = np.array(sigmoid(np.matmul(theta2,xh))).reshape(10,1)
```

#### Backpropagation
```python
error1 = np.subtract(xo, yo)
error2 = np.matmul(theta2.transpose(), error1)
delta1 = np.matmul(np.dot(np.multiply(np.multiply(xo,(1-xo)),error1),0.1),xh.transpose())
delta2 = np.matmul(np.dot(np.multiply(np.multiply(xh,(1-xh)),error2),0.1),xi.transpose())
```

### Results & Analysis

The network achieves **91.27% accuracy** on the MNIST test set, which is impressive for a from-scratch implementation. Key observations:

- **Convergence**: Network learns effectively through manual backpropagation
- **Generalization**: Good performance on unseen test data
- **Efficiency**: Matrix operations enable fast training despite large dataset

### Learning Outcomes

This project demonstrates:

- **Deep Learning Fundamentals**: Understanding of neural network mathematics
- **Matrix Operations**: Efficient NumPy implementations
- **Gradient Descent**: Manual implementation of backpropagation
- **Computer Vision**: Image preprocessing and feature extraction
- **Performance Optimization**: Balancing accuracy with computational efficiency

### Customization

#### Modifying Network Architecture

```python
# Change hidden layer size
theta1 = np.random.uniform(-0.1, 0.1, (784, 400))  # 784 → 400
theta2 = np.random.uniform(-0.1, 0.1, (10, 400))   # 400 → 10
```

#### Adjusting Learning Rate

```python
# Modify learning rate in backpropagation
learning_rate = 0.01  # Default: 0.1
delta1 = np.matmul(np.dot(np.multiply(np.multiply(xo,(1-xo)),error1),learning_rate),xh.transpose())
```

_For more examples, please refer to the [Documentation](https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Implement convolutional layers
- [ ] Add dropout regularization
- [ ] Support for different activation functions
- [ ] Real-time prediction interface
- [ ] Model export capabilities
- [ ] Performance benchmarking tools
- [ ] Multi-class classification support
- [ ] Transfer learning implementation

See the [open issues](https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Chandan Gowda K S - chandan.keelara@gmail.com

Project Link: [https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits](https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits)

Project Link: [https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits](https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) for the classic handwritten digit dataset
* [TensorFlow](https://tensorflow.org/) for providing easy access to MNIST data
* [NumPy](https://numpy.org/) for efficient matrix operations
* [Matplotlib](https://matplotlib.org/) for visualization capabilities
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emojis](https://gist.github.com/rxaviers/7360908)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search.html?q=search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits.svg?style=for-the-badge
[forks-shield]: https://img.shields.io/github/forks/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits.svg?style=for-the-badge
[stars-shield]: https://img.shields.io/github/stars/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits.svg?style=for-the-badge
[issues-shield]: https://img.shields.io/github/issues/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits.svg?style=for-the-badge
[license-shield]: https://img.shields.io/github/license/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits.svg?style=for-the-badge
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[contributors-url]: https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits/graphs/contributors
[forks-url]: https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits/network/members
[stars-url]: https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits/stargazers
[issues-url]: https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits/issues
[license-url]: https://github.com/virtual457/simple-nueral-network-used-to-recognize-handwritten-digits/blob/master/LICENSE.txt
[linkedin-url]: https://www.linkedin.com/in/chandan-gowda-k-s-765194186/
