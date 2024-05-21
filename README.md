# Motion Blur Simulation Tool

Welcome to the Motion Blur Simulation Tool! This repository contains the code and resources developed for simulating different exposure times, ambient lighting, and motion blur in images.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Assumptions](#assumptions)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
As part of a study on motion blur, this simulation tool was developed to allow for quick and efficient exploration of various test conditions. The tool simulates different exposure times, ambient lighting conditions, and motion blur based on pixel movement per second. The primary focus of the study and the software is on horizontal pixel movement per second (DX). An example comparing a real motion-blurred image to a simulated motion-blurred image is shown in Figure 2.

## Features
- **Exposure Time Simulation**: Simulate various exposure times to observe their effects on motion blur.
- **Ambient Lighting Simulation**: Adjust ambient lighting conditions to see how they impact the appearance of motion blur.
- **Motion Blur Simulation**: Simulate motion blur using pixel movement per second as the metric.
- **Efficient Exploration**: Quickly explore different test conditions to understand their impact on image quality.

## Installation
To install and set up the Motion Blur Simulation Tool, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/motion-blur-simulation-tool.git
    cd motion-blur-simulation-tool
    ```

2. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Command Line Interface
You can use the command line interface (CLI) to apply the simulations.

#### Simulating an Image
```sh
python simulate.py --input path/to/image.jpg --output path/to/output.jpg --exposure 0.01 --lighting 100 --dx 50
```
#### Parameters
--input: Path to the input image file.
--output: Path to save the output image file.
--exposure: Exposure time in seconds.
--lighting: Ambient lighting level (arbitrary units).
--dx: Horizontal pixel movement per second.
## Examples
Here are some example commands and their results:

Example 1: Basic Simulation
Original Image:

Command:

sh
Copy code
python simulate.py --input path/to/original.jpg --output path/to/simulated.jpg --exposure 0.01 --lighting 100 --dx 50
Simulated Image:

#### Assumptions
The simulation tool operates under the following assumptions:

Monochromatic Scene: The simulation assumes a monochromatic (single color channel) scene.
Global Shutter Capture Mode: The simulation assumes the use of a global shutter capture mode.
Simplified Lens Model: The simulation uses a simplified lens model for computations.
Contributing
We welcome contributions to the Motion Blur Simulation Tool! To contribute:

Fork the repository.
Create a new branch for your feature or bugfix.
Make your changes.
Submit a pull request with a detailed description of your changes.
Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

#### License
This project is licensed under the MIT License. See the LICENSE file for details.

#### Acknowledgments
We would like to thank all the contributors and the open-source community for their invaluable support and contributions to this project.

For any questions or issues, please open an issue on this repository or contact the maintainer.

Happy coding!