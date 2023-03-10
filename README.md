# COMP6721 *(TEAM-G)*
 ## Garbage Image Classification using Deep Learning


 [![Contributors][contributors-shield]][contributors-url]

 ## Project Intro/Objective

 Garbage classification is an important part of environmental
 protection. There are various laws regulating what
 can be considered garbage in different parts of the world. It
 can be difficult to accurately distinguish between different
 types of garbage. We are using deep learning to help us
 categorize garbage into different categories. We are using
 deep learning to prove that one of the biggest impediments
 to efficient waste management can finally be eliminated.

 ### Methods Used
 * Data Preprocessing - We resized all training and test images to a size of 227x227 to best fit with our models. We incorporated data augmentation techniques including vertical flip, horizontal flip, width shift, height shift, zooming and rotation 
 * Data Normalization - Using Mean and Standar Deviation to Normalize the image data and using tensors.
 * Deep Learning - Used deep cnn models to train the given dataset
 * Transfer Learning - Used pre-trained models with pre-trained weights to compare the metrics of our model. 
 * Ablation Studies - Tried different parameters to see what results are acheived.
 * Hyperparameter Tuning - Used various learning rates and epochs to choose the best possible combination to generate accurate output. 
 * Model Optimization Algorithms - Used Adam Optimation and Cross entropy loss function as optimization algorithms.

 ### Technologies
 * Python
 * sklearn
 * numpy
 * matplotlib
 * torchvision, torch
 * Collab (To run the code)
 
 ### Models
 * ResNet-50
 * AlexNet
 * VGG-16
 
 ## Needs of this project
 - Understanding AI and Deep Learning
 - Learn to build models and solve a problem using CNN's
 - Model Optimization and Fine Tuning
 - Researching in AI and Deep Learning
 
 ### Github File Structure
 * Scratch - Contains all the three models on three datasets trained from scratch.
 * ablation study - Contains the ablation studies and hyper parameter tuning perform using two scratch models (VGG16,AlexNet)
 * transfer learning = Contains all the three models on three datasets using pre-trained weights(performed transfer learning on ResNet-50 and AlexNet)
 * readme - Contains information about the project.

 ## Getting Started

 1. Clone this repo.
 2. Upload datasets [dataset1](https://drive.google.com/file/d/1zoqZ03wyNwkt1hcnZ5l1DOMqIZmawtH2/view?usp=sharing), [dataset2](https://drive.google.com/file/d/1e_B19HVtcSS-zTigVSuBRXJg5mLgQjpb/view?usp=sharing), [dataset3](https://drive.google.com/file/d/10s_k12qCr2Ce4CDhMOo_ZJ252wOasuev/view?usp=sharing) to the drive.

     *(If using datasets offline then change the path in the code)*

 3. Install the library Numpy, torchvision, matplotlib
    ```
    pip3 install numpy, torchvision, sklearn, torch, matplotlib
    ```
 4. Run the project one after segment.
 
 ## How to train/validate our models?
 1. Run the code files on colab that you want to run.
 2. Make sure if you are using another dataset, then change the path in data_dir variable.
 3. If want to test the trained model, we have created checkpoint models that could be used for testing based on our trained models from scratch.
 
 ## Run the pre-trained model on the provided sample test dataset
 
 1. Change the line of code from the file
    ```
    data_dir = "Garbage classification"
    ```
    to
    ```
    data_dir = "path to the sample test dataset"
    ```
 1. You need to change the 

 ## Contributing Members

 | Name                                              | Twitter Handle | Student Id |
 |---------------------------------------------------|---------------|-----------|
 | [Nihar Sheth](https://github.com/nihar1805)       | @sheth_nihar      | 40198433  |
 | [Karan Singla](https://github.com/karansingla007) | @karansinglaOO7      | 40225623  |
 | [Saurabh Sharma](https://github.com/saurabhs679)  | @saurabhs679      | 40226298  |
 | [Elahehpa](https://github.com/Elahehpa)           |        | 40204681  |

[Video Link](https://drive.google.com/file/d/1w3Y7VhcCRyrjIZaCpk5gof2FQxrmyOo3/view?usp=share_link)

 <!-- MARKDOWN LINKS & IMAGES -->
 <!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
 [contributors-shield]: https://img.shields.io/badge/4-Contributors-green
 [contributors-url]: https://github.com/karansingla007/SOEN6441_APP_project
 [stars-shield]: https://img.shields.io/badge/STARS-2-yellowgreen
 [stars-url]: https://github.com/karansingla007/SOEN6441_APP_project
 [linkedin-shield]: https://img.shields.io/badge/LINKEDIN-karansingla007-blue
 [linkedin-url]: https://www.linkedin.com/in/karansingla007/
 [product-screenshot]: images/screenshot.png
 [Angular.io]: https://forthebadge.com/images/badges/made-with-python.svg
 [Angular-url]: https://www.python.org
 [Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
 [Bootstrap-url]: https://getbootstrap.com
 [JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
 [JQuery-url]: https://jquery.com 
