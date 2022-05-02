# DeepSafety
This is a repository to study safety concepts for deep learning perception applications at Ravensburg-Weingarten University.

## The Codebase
### kickstart.py
Provides an initial pipeline, supporting pretrained models, data loading from directory and a train / validation split, preprocessing steps, tensorboard support, the model training pipeline, and save capabilities for training logs and the final model. The script is an adaptation from [here](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub)

### requirements.txt
Running python == 3.9, the crucial imports are
```
numpy==1.21.2
tensorflow==2.7.0
tensorflow-hub==0.12.0
tensorboard==2.6.0
```

### Data
The data to train and validate the model we will be using initially can be downloaded [here](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/download)
The data enables you to train an initial neural network for traffic sign recognition - we assume that the final goal is to use this model in an autonomous vehicle - meaning the system will not have any form of human intervention or supervision at any point. So when assessing performance, ask the question: "Would I sit behind a stop sign with this vehicle approaching?".
Once you trained the initial model, you are free to extend your dataset, the model, or for that anything. The golden rule is documentation and versioning of your changes and contributions from other sources.

## Applied Deep Learning meets Safety
During the lecture you will be tasked with providing a safety strategy for your deep learning application.
To make things spicy, you will be asked to validate your system on data that you will not know beforehand.
Your goal is to prepare yourself for these data ingests. 
Preparing includes acting and documenting on:
- An elaborate validation pipeline for your system, with the goal to precisely describe the risks, performance and functionality of your system
- An elaborate safety concept, including roles and processes on safety strategy, safety management, safety engineering 
- At some point you will need to implement and decide on a safety architecture

Obs! How I use the term system, and not neural network or model. 

## FAQ
So far so good.
1. How will my code be evaluated?
- Program Design (Solution well thought through)
- Program Execution (Program runs correctly)
- Specification Satisfaction (Program contributes to the overall validation pipeline)
- Coding Style (Well-formatted, modular & understandable code, use of language capabilities)
- Comments (Concise, meaningful, well-formatted comments)
- Extra Credit (for a very smart or 'beautiful' idea that reflects deep understanding of the topic)


## Additional Material
(If material is behind a paywall, remember! You can order a lot of material through your universitie's library - there is no need to buy everything yourself)

  [1] *Deep Learning - Ian Goodfellow and Yoshua Bengio and Aaron Courville*  
  Abstract: The Deep Learning textbook is a resource intended to help students and practitioners enter the field of machine learning in general and deep learning in particular. The online version of the book is now complete and will remain available online for free.  
  Link: http://www.deeplearningbook.org/
  
  [2] *Pattern Recognition and Machine Learning - Bishop*
  Abstract: Thisnewtextbookreﬂectstheserecentdevelopmentswhileprovidingacomprehensive introduction to the ﬁelds of pattern recognition and machine learning. It is aimed at advanced undergraduates or ﬁrst year PhD students, as well as researchers and practitioners, and assumes no previous knowledge of pattern recognition or machinelearningconcepts. 
  Link: http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf 
 
 [3] *Deep Learning and Machine Learning Courses on Coursera - Andrew Ng*
 Abstract: In these courses, you will learn the foundations of Deep Learning, understand how to build neural networks, and learn how to lead successful machine learning projects. You will learn about Convolutional networks, RNNs, LSTM, Adam, Dropout, BatchNorm, Xavier/He initialization, and more. You will work on case studies from healthcare, autonomous driving, sign language reading, music generation, and natural language processing. You will master not only the theory, but also see how it is applied in industry. You will practice all these ideas in Python and in TensorFlow, which we will teach.   
   Link: https://www.coursera.org/courses?query=andrew%20ng 
 
 [4] *CS231n: Convolutional Neural Networks for Visual Recognition - Andrej Karpathy*
 Abstract: Computer Vision has become ubiquitous in our society, with applications in search, image understanding, apps, mapping, medicine, drones, and self-driving cars. Core to many of these applications are visual recognition tasks such as image classification, localization and detection. Recent developments in neural network (aka “deep learning”) approaches have greatly advanced the performance of these state-of-the-art visual recognition systems. This course is a deep dive into details of the deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification. During the 10-week course, students will learn to implement, train and debug their own neural networks and gain a detailed understanding of cutting-edge research in computer vision. The final assignment will involve training a multi-million parameter convolutional neural network and applying it on the largest image classification dataset (ImageNet). We will focus on teaching how to set up the problem of image recognition, the learning algorithms (e.g. backpropagation), practical engineering tricks for training and fine-tuning the networks and guide the students through hands-on assignments and a final course project. Much of the background and materials of this course will be drawn from the ImageNet Challenge. 
   Link Lecture Series: https://www.youtube.com/watch?v=NfnWJUyUJYU&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC 
   Link Lecture Notes: http://cs231n.stanford.edu/
   
 [5] *The Missing Semester of Your CS Education*
 Abstract: Classes teach you all about advanced topics within CS, from operating systems to machine learning, but there’s one critical subject that’s rarely covered, and is instead left to students to figure out on their own: proficiency with their tools. We’ll teach you how to master the command-line, use a powerful text editor, use fancy features of version control systems, and much more!
Students spend hundreds of hours using these tools over the course of their education (and thousands over their career), so it makes sense to make the experience as fluid and frictionless as possible. Mastering these tools not only enables you to spend less time on figuring out how to bend your tools to your will, but it also lets you solve problems that would previously seem impossibly complex.
   Link Lecture Series: https://missing.csail.mit.edu/
 
 [6] *Coffee Table Solutions* - full disclosure: I am an Author of this one.
 Abstract: The neural network was well into training for 42 hours. It all looked good - the gradients were flowing, the weights were updating, and the loss was decreasing. But then came the predictions for validation - all zeroes, no pattern recognized. "What did I do wrong?" — I asked my computer, who didn't answer. I noticed the dull feeling of despair and hopelessness rise inside my chest. After some more debugging and wasting more of the precious working hours, I would usually rededicate myself to the books, tutorials, and courses I knew so well by then. Somewhere had to be a hint to what I was missing - there was not. To give me a boost and to stay well caffeinated, I would usually go and grab a coffee in the coffee kitchen. Standing there at the coffee table with other students, researchers, developers, and practitioners, I would soon find myself indulging in the soft, warm feeling of getting this issue off my chest. And this is where the magic happened. Either someone already had a similar issue and knew how to fix it. Someone had an idea of narrowing down what the issue's root actually is about. Someone realized there was a conceptional problem in the data set or the model. Most of the time, what was shared at the coffee table were vague hints, interpretations and ideas, heuristics, best practices, experiences from applying deep learning in research and development. Over time we realized that some pitfalls, issues, and knowledge gaps were reoccurring in deep learning novices and were regularly brought to the coffee table while these students developed into deep learning engineers. Shortly after, we were taking notes. Throughout many sessions at the coffee table and hours of debugging, training, and evaluating deep learning approaches, researching and applying state-of-the-art concepts, we polished and enriched these notes with the great ideas that have been around and extensive references for further reading, based on our own experience. The result is this book. We hope it will be of use to you, too.
  Link to Kindle Book: https://www.amazon.de/dp/B09QRGWWZP

### Contact
mark.schutera@gmail.com\
Feel free to [follow or connect](https://www.linkedin.com/in/schuteramark/)
