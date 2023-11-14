**Laboratory Exercise: Convolutional Neural Networks**

**1. Introduction**

In this laboratory exercise, we aim to explore the performance of various Convolutional Neural Network (CNN) configurations using the CalTech 101 Silhouettes Data Set. The dataset consists of 101 different silhouettes, each represented by 28x28 pixels, resulting in 784 input variables. The primary objective is to evaluate the impact of different CNN configurations on image classification performance.

**2. Experimental Configurations**

We will investigate the following configurations:

1. **Comparison of Convolutional Blocks:**
   - Configuration 1: CNN with one convolutional block (NB=1, FS=128).
   - Configuration 2: CNN with three convolutional blocks (NB=3, FS=32, 64, 128).

2. **Activation Functions in NHL:**
   - Configuration 3: CNN with Sigmoid activation function in the non-linear hidden layer (NHL).
   - Configuration 4: CNN with Rectified Linear activation function in NHL.

3. **Percentage Split of Data Sets:**
   - Configuration 5: CNN trained with an 80/10/10 split (training/validation/test).
   - Configuration 6: CNN trained with a 40/20/40 split.
   - Configuration 7: CNN trained with a 10/10/80 split.

**3. Parameter Selection**

- **Base Architecture:**
  - Input Layer → Convolutional Blocks (kernel size=3, filter size=FS) → NHL → Max-Pooling → Fully Connected Layer → Output Layer → Cost Function Layer.

- **Training Algorithm:**
  - Adam optimization algorithm is recommended for computational efficiency.

- **Other Parameters:**
  - Activation function in OL, cost function, maximum epochs, L2 regularization, learning rate, and momentum should be set a priori. Justification or prior search required.

**4. Results Analysis**

Include tables describing the results obtained for each configuration. Explain and justify the outcomes. Consider mean accuracy as the performance measure.

**5. Conclusions**

Summarize findings, insights, and implications drawn from the experiments. Discuss the impact of different configurations on performance.

**6. Code Files**

Include all relevant .m files with your code implementation.

**7. Tool Usage**

If ChatGPT or a similar tool is used during the document preparation, mention it each time and clarify that it's used for assistance, not for generating the core content.

**8. Recommendations**

Test the hardware and software environment beforehand, particularly noting the potential impact of using a GPU for faster execution times.

**Note:**
This exercise is designed for Matlab, but alternative environments are allowed. Refer to the Lab Class Guide for basic instructions on constructing and training a CNN with Matlab.