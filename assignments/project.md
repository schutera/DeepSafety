# Final Project

## Overview
In this project you will implement, evaluate, monitor and evolve a validation campaign for an autonomous driving system. You will work with a neural network for classification of traffic signs using the [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html). We assume that the final goal is to use this model in an autonomous vehicle.

The aim of this project is to evaluate the performance of a traffic sign classification system with a focus on safety, which entails many concerns, including uncertainty, robustness and deployment.

The assignments are intended to provide you with insights and practice on individual aspects that are important for validating safety-critical machine learning systems. Take what you already did for the assignments, improve and extend it, and create a robust and reliable validation process. Following aspects may serve as an inspiration:

* An elaborate validation pipeline for your system, with the goal to precisely describe the risks, performance and functionality of your system.
* An elaborate safety concept, including roles and processes on safety strategy, safety management, safety engineering.
* At some point you will need to implement and decide on a safety architecture.

A word on **scope and difficulty**: some basic programming skills are expected as a prerequisite for this course but we do not assume that you are already familiar with all these technologies and tools. Nonetheless, we will not teach those technologies but expect that you can learn enough by yourself by reading documentation or tutorials as needed to perform small changes. Professional developers and data scientists will constantly have to learn new technologies on their own. Incrementally modifying and maintaining existing systems is far more common than developing new things from scratch. Learning new technologies, libraries and tools and troubleshooting problems with documentation, debugging, and sites like Stackoverflow are important skills.

## Technical Details
We provide a basic integration of [MLFlow](https://mlflow.org/) for experiment tracking in the given code base. However, you have full flexibility to use any other tools you prefer. You just need to ensure that we can run your code and reproduce your results.


## Submission
Submit all your code and your final report (max. 5 pages) via Moodle.

## Grading
Each project will be graded based on the following criteria:

* The report is clearly structured and describes your validation campaign. Do not only describe what you did, but especially describe why you did it and how it relates to your whole validation procedure. Reflect on issues and limitations of your approach and think about possible improvements.
* The assignments are well integrated into your whole validation pipeline. Explain how they contribute.
* A pipeline for systematically analysing the performance of models is implemented.
* Correct and understandable code which contributes to the overall goal.

If you are not sure what is expected of you, please ask.