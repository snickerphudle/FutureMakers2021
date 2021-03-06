# MIT FutureMakers 2021

### Start Date: July 6, 2021

The MIT FutureMakers Deep Learning Program is a 6-week virtual program designed by [SureStart](https://mysurestart.com/) and [MIT Raise](https://raise.mit.edu/) to learn AI concepts, work with machine learning models and get hands-on experience applying those concepts to solve real world problems.

The program includes:

- Learning AI, machine learning and emotion AI concepts
- Turning skills into confidence through building AI solutions hands-on
- Consistent support and encouragement from a mentor 1-1
- Regular tech talks and career readiness seminars
- Entrepreneurship and leadership skills development

The first part of the program was the training phase, where I learned key machine learning concepts. The final two weeks were dedicated to the Create-a-thon, a [hackathon](https://static1.squarespace.com/static/5f45536caa356e6ab51588f4/t/6107c33cc9c0836c362dec4f/1627898684269/MIT_FutureMakers_2021_Themes.pdf) focused around designing solutions to relevant problems post-pandemic. For this event, my team and I designed [Automatic Inventory](https://github.com/snickerphudle/AutomaticInventory), an intelligent food systems manager that uses computer vision, object detection, and image segmentation to recognize pantry items and help reduce food waste. For this project, we addressed equitable resource allocation prompt.

### July 2021
|            Sun           |            Mon           |            Tue           |            Wed           |            Thu           |  Fri  |  Sat  | 
|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:-----:|:-----:|
|                          |                          |                          |             1            |             2            |   3   |   4   |
|             5            |[6](#day-1-july-6-2021)   |[7](#day-2-july-7-2021)   |[8](#day-3-july-8-2021)   |[9](#day-4-july-9-2021)   |   10  |   11  |
|[12](#day-7-july-12-2021) |[13](#day-8-july-13-2021) |[14](#day-9-july-14-2021) |[15](#day-10-july-15-2021)|[16](#day-11-july-16-2021)|   17  |   18  |
|[19](#day-14-july-19-2021)|[20](#day-15-july-20-2021)|[21](#day-16-july-21-2021)|[22](#day-17-july-22-2021)|[23](#day-18-july-22-2021)|   24  |   25  |
|[26](#day-21-july-26-2021)|[27](#day-22-july-27-2021)|[28](#day-23-july-28-2021)|[29](#day-24-july-29-2021)|             30           |   31  |       |

## Reflection Responses

### Day 1: July 6, 2021

Over the course of these next 6 weeks, I hope to become more familiar with the basics of machine learning, GitHub, and working in a team setting on a machine learning project. I have taken computer science courses before, but I have never worked on a project using GitHub and other platforms besides traditional languages such as Java, C++, or Python. At the end of the program, hopefully I will be more comfortable with this environment and can start some of my own projects.

### Day 2: July 7, 2021

In Dr. David Kong's talk about leadership today, I learned about what it means to be a leader and how to use my storytelling ability to enact change in my community. In any leadership role, the end goal is to unite others in a common goal to accomplish something. The best way to invoke change in others is through storytelling, mentoring, identifying a common goal, developing a plan to achieve that goal, and then actually executing steps to reach that goal. I learned that storytelling is a combination of conflict, actions, and results, and a moral is created at the end of the story. During the talk today, we watched several videos showing great examples of storytelling, and in each story, the speakers evoked powerful emotions in the audience to brings about change and act. No matter what, each story ended with a glimmer of hope, which prompted the audience to act. Overall, I learned a lot about leadership today, and understand that being a leader is not about taking glory for an end goal, but it's about empowering others to become better through a common goal.

### Day 3: July 8, 2021

In supervised learning, the data is labeled, the model learns from feedback, and the model is expected to predict an outcome, either regression or classification. The goal of supervised learning is to create a model that can classify data or predict outcomes accurately based on labeled input and output. Overtime, the model can measure its own accuracy and gradually learn. In unsupervised learning, the model uses algorithms to analyze unlabeled data sets and draw patterns between them without the need of human intervention. Some examples of unsupervised learning are clustering, association, and dimensionality reduction. Skikit Learn is used mainly for machine learning calculations and modelling. Other libraries must be used to visualize the data.

### Day 4: July 9, 2021

A problem that can be solved with machine learning is the recognition of skin problems, particularly acne. Everyone has a different skin type, whether it be oily, combination, or dry. Furthermore, some people have different skin problems such as lesions, whiteheads, cysts, etc. Using machine learning, we can identify the type of skin a patient has, which will help doctors identify a course of action to counteract those skin problems. If I were developing a model to solve this problem, I would most likely use a CNN to classify and identify the skin problems on someone's face. A dataset I could use to train the model comes from https://archive.ics.uci.edu/ml/machine-learning-databases/00229/

### Day 7: July 12, 2021

The specific definition of a tensor varies very much depending on which field it is being applied too as well as who you ask. For the purposes of machine learning, a tensor is a general term that encapsulates vectors and matrices in any dimension. In order for machines to learn, there must be sufficient data to do so. However, modern data is rarely limited to just 3 dimensions, therefore tensors provide a way to represent information and store data in a multimensional space. The computations I ran were quite fast.

### Day 8: July 13, 2021

Today I worked with Arezoo to go over the sarcasm detection notebook. Today I learned that the pandas library has an object called a dataframe, which can store data in a very convenient way. Not only can you easily print the dataframe and make it look nice using headline, but you can also access columns using ['feature']. Furthermore, there are many different tasks in machine learning, and each task uses different functions and approaches. For example, for computer vision, CNNs use convolutions and pooling. For NLP/sarcasm detection, tokenizers and embedding layers are used. A tokenizer takes every word in a string and assigns it to a dictionary with an index by frequency. tokenizer.texts_to_sequences creates a tensor which maps every word in a string to an aarry with a number based on its index in the dictionary.

### Day 9: July 14, 2021

Today I built a number recognition neural network CNN which can identify numbers from the MNIST dataset.

### Day 10: July 15, 2021

I have been struggling with trying to understand the various topics of machine learning and neural networks this past week, but I have reviewed by reading some articles and watching a couple videos, so I will outline my progress of understanding here. An artificial neural network is a network of artificial neurons which take in input, perform algorithms, and output values which can solve complex problems that can normally only be performed by "intelligent" humans. The process of a neural network is as follows: input -> prediction -> calculate error -> tweak weights and biases -> repeat. The way input data flows through the neural network is that each node's input is multiplied by a weight and added to a bias. Each of these connections are summed and ran through a non-linear activation function. Non-linear activations are a way of adding non-linearity to a model in order to detect more complex patterns in the data and is a means of defining how a node's output will be expressed for the next layer. The method that the model becomes trained is through gradient descent, in which the weights and biases are tweaked in the direction that the cost function is approaching a local minimum. Forward propagation is the process of feeding input data forward through the neural network. Backward propagation is the process of using training the model using gradient descent.

BIAS OF MACHINE LEARNING 

In the game Survival of the Best Fit, there was an algorithm that disproportionately hired workers for the company that were more yellow than blue. The way this worked was that the data of the manual selection of hired workers in the past served as the input for the machine learning model. Since I selected more yellow people during the manual selection, the machine learning model picked up and consequently chose more yellow people to work at the company. Even though the resumes did not say anything about color or protected attributes about a person, the results ended up being discriminatory towards a certain group. I believe that this is the case because yellow people tended to do the same things and have the same experiences, such as come from the same community or school. After leraning these features, the machine created a model that favored yellow people even though certain other people were definitely qualified.

Another example of a biased machine learning model is a model that determines one's salary based on certain features. This model could be biased be awarding certain people with more pay just because they possess a particular protected attribute that another person may not have. This model can be more fair or equitable by not including features of protected attributes, collecting data ethically and randomly, annotating the data with objective/hard metrics, doing a test blind experiment, etc. I chose this particular model because I know that women have historically been paid less than men. Instead of basing payment on subjective factors, a machine learning model that awards people for doing good work according to high metrics and efficiency should be rewarded, but only if the machine is not biased.

### Day 11: July 16, 2021

A convolutional neural network (CNN) and a Fully Connected Neural Network (ANN) are different. A CNN consists of convolution layers, pooling layers, flattening layers, and a fully connected neural network at the end. CNNs are developed mainly for computer vision and image analysis, especially classification. CNNs work by taking in input images, extracting features by performing convolution operations on the pixels, downsampling/generalizing the data using pooling, and finally flattening the feature maps into a vector which will be used by the CNN. One large advantage that CNNs have over MLP is that CNNs consider groups of pixels to determine features as opposed to individual pixels, which MLPs do. This allows the CNN to better recognize image features. Furthermore, CNNs learn by updating not only the weights in the fully connected layer, but also searching for the most optimal values for the filter/kernel.

### Day 14: July 19, 2021

Today I learned about cost functions, mini-batches, bias, variance, regularization, and gradient descent. In machine learning, cost functions are a calculation of how much error is produced from your model compared to the actual target value. 

Something I didn't know is that different types of problems require different cost functions. For example, in regression problems, some common cost functions include Mean Squared Error, Mean Absolute Error, and Mean Squared Logarithmic Error. For classification, there are certain functions such as Binary Cross Entropy, Multiclass Cross Entropy Loss, etc. The way that neural networks learn is by calculating the cost and following the gradient descent algorithm to update the weights. 

There are 3 different ways that the values are updated: batch gradient descent, mini-batch gradient descent, and stochastic gradient descent. Batch gradient descent aggregates, averages the error, and updates the weights after the entire data set passes through the neural network. This results in a smooth updating of model parameters, although it requires a lot of time to update the weights once. Stochastic gradient descent, or SGD, is when the weights are updated after a single data point passes through the network. This creates lots of noise in the model, and often leads to overfitting. Mini-batch gradient descent a compromise between the two, updating the weights after a certain number of iterations, or 1 batch of data. 

The learning rate is a hyper parameter that is multiplied with the gradient to update the weights of a neural network. Along with many other factors in building a neural network, the learning rate must be just the right value to strike a balance between computational time and not diverging from the minimum cost. Momentum is a fraction of the previous weight update that is added to the new weight update to help accelerate the convergence to a cost minimum. It helps to reduce noise and approach the minimum more quickly, especially when the gradient changes direction frequently.

A very common problem in Machine Learning is the issue of overfitting, in which the model creates too specific of a relationship between the features and target. This leads to lower accuracy, meaning that if the model is faced with new data, the model will be unable to provide an accurate answer. Bias is a model???s inability to capture the relationship between the data. A model with high bias means that it does not capture the relationship very well (underfitting), while a model with low bias means that it captures the relationship too well (overfitting). Variance is when there is a signifiant difference in fits between the training and testing set. Ideally, a model should have low bias and low variance, meaning it captures the relationship well and performs well on the new data sets.

To combat overfitting, there are several methods. One such method to do so is regularization. Drop regularization is when there is a probability for a node to be removed or inactive during a pass of data. This forces the model to search for new ways to represent the relationship and prevents the model from relying on the same nodes over and over, thus leading to overfitting. I have not looked into the specific details of L1 or L2 Regularization, although I plan to soon.

### Day 15: July 20, 2021
Some advantages of ReLU are that it is very simplistic, only requiring a max function, it helps circumvent the problem of vanishing gradient descent (acts linear to help the backpropagation process, but is nonlinear to help learn complex relationships), and is not sensitive to values close to zero. This allows models to learn more quickly and efficiently, while not being stuck on the problems associated with sigmoid and tanh. Common uses of ReLU include CNN and MLP, although it is not normally used in RNN.

### Day 16: July 21, 2021
Today I created a neural network to identify male or female voices. While creating the network, I learned about a lot of new functions that are useful for creating it. For example, the seaborn.countplot function creates a graph that shows the frequency of the labels. Furthermore, the pd.read_csv is used to read in data. This is very useful for visualizing raw input data and determining how to process it. Something substantially useful and new that I learned from this notebook is the experimentation phase of creating models. In the notebook, there were arrays for standardization and dropout, which contained a variety of values. The notebook then runs a loop through all the models while recording the accuracy. At the end, it is very easy to see the results and compare which hyperparameters were best.

### Day 17: July 22, 2021
Today I created a CNN model that can classify dogs and cats.

### Day 18: July 23, 2021
Today I created a NLP model that determines whether a Twitter comment is negative, positive, or neutral.

### Day 21: July 26, 2021
Today I made an autoencoder. I learned that there are multiple ways to create neural networks. So far, we have been creating models using the Sequential wrapper, but I learned that the native TensorFlow way of creating models is better for more complex networks. Using the sequential wrapper, all layers pass data in a linear fashion from one output node to the next input node. However, in more complex models, data sometimes splits to other layers or multiple layers, which is why it's better to code the model in the TensorFlow way to achieve these complex models.

### Day 22: July 27, 2021
Today I created a classification model that can analyze a audio clip and decide what type of emotion is being expressed in the audio file. Using the help of GitHub user MITESHPUTHRANNEU's repo, I was able to create a model that performed accurately produces the correct result about 70% of the time. In the modern day and age, affective computing can be applied to a variety of fields, such as improving security, mental health, and more. By analyzing how a person feels, the machine can adequately provide support by offering options to maintain that person's happiness or reducing ill-mannered feelings.

### Day 23: July 28, 2021
Today I created a NLP model that can classify whether a given movie review is positive or negative. The model I created was in conjunction with a TensorFlow tutorial on NLP. This particular model used a dropout of 0.5, early stopping of patience 5, batch size 625, and 10 epochs. 

Some large concerns about large NLP models include the bias against certain gender, racial, and LGBTQIA+ groups. Large NLP models learn from very large datasets created by humans, meaning that overtime it may pick up bias against certain groups, which is not how it should ideally be performing. Furthermore, this powerful technology could easily be misused to create fake news, hate messages, etc. Before large models like these are used commonly, questions regarding these topics must be answered and solutions against them must be developed.

### Day 24: July 29, 2021
Today I created an image classification model that can tell whether you are happy, sad, angry, fearful, or surprised.

### Weeks 5 & 6: July 31, 2021 - August 13, 2021
During the final weeks of the program, my team and I applied the concepts we learned to create an intelligent food systems manager, [Automatic Inventory](https://github.com/snickerphudle/AutomaticInventory).
 
