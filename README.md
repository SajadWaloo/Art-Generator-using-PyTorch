
# Art Generator using PyTorch

This is a simple art generator project implemented using PyTorch. The generator model creates colorful images inspired by art, leveraging a Generative Adversarial Network (GAN) architecture.

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Description

The project implements a GAN-based generator and discriminator. The generator learns to create art-like images, while the discriminator aims to distinguish between real and generated images. The models are trained iteratively in a way that the generator improves over time, creating more convincing art-like images.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchvision

You can install the required packages using pip:


pip install torch torchvision


### Installation

1. Clone the repository:


git clone https://github.com/your-username/art-generator.git
cd art-generator

2. Download or prepare your art dataset and update the dataset path in the code.

## Usage

1. Navigate to the repository's root directory.

2. Run the training script:


python art_generator.py


3. The training process will start, and you'll see progress and generated images being saved.

4. Once training is complete, the generator model will be saved as `generator_model.pth`.

## Contributing

Contributions are welcome! If you find any issues or want to enhance the project, feel free to submit a pull request.
