Download Link: https://assignmentchef.com/product/solved-comp135-project-1
<br>
For this assignment, you will explore the use of logistic regression for image classification. You will hand in a PDF containing results and analysis, along with the usual collaborators file; you <em>do not </em>hand in code for this assignment. You will also submit a text-file containing predictions of a classifier that you built; these results will be used to score your code on a leaderboard, based upon accuracy on a provided testing set (for which you do not know the correct outputs).

<h1>Part Zero: Collaborators file</h1>

Provide the usual file containing your name, the amount of time you worked on the assignment, and any resources or individuals you consulted in your work.

<h1>Part One: Logistic Regression for Digit Classification</h1>

You have been given data (in data_digits_8_vs_9_noisy) corresponding to images of handwritten digits (8 and 9 in particular).<sup>∗ </sup>As before, this data has been split into various training, and testing sets; each set is given in CSV form, and is divided into inputs (x) and outputs (y).

Each row of the input data consists of pixel data from a (28 × 28) image with gray-scale values between 0<em>.</em>0 (black) and 1<em>.</em>0 (white); this pixel data is thus represented as a single feature-vector of length 28<sup>2 </sup>= 784 such values. The output data is a binary label, with 0 representing an 8, and 1 representing a 9.

<ol>

 <li>You will fit logistic regression models to the training data, using sklearn’s implementation of the model, with the liblinear solver:</li>

</ol>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">https://scikit-learn.org/stable/modules/generated/sklearn.linear_model. </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">LogisticRegression.html</a>

Leaving all other parameters with default values, you will explore what happens when we limit the iterations allowed for the solver to converge on its solution.

For the values <em>i </em>= 1<em>,</em>2<em>,…,</em>40, build a logistic regression model with the max_iter set to <em>i</em>. Fit each such model to the training data, and keep track of the <em>accuracy </em>of the resulting model (via the model’s own score() function) along with the <em>logistic loss</em>, each measured on the training data.<sup>†</sup>

Produce two plots, each with the values of <em>i </em>as <em>x</em>-axis and with the accuracy/loss, respectively, as <em>y</em>. Place these plots into your PDF document, with captions labeling each appropriately. Below the plots, discuss the results you are seeing; what do they show, and why?

∗

This is based upon the popular <a href="http://yann.lecun.com/exdb/mnist/">MNIST data-set</a><a href="http://yann.lecun.com/exdb/mnist/">,</a> from the work of LeCun, Cortes, and Burges. To add to the challenge, the data has been preprocessed to add some random noise to each image.

†

When doing this, you will probably see warnings about non-convergence for lower values of parameter <em>i</em>. You can ignore these warnings, as they are expected. You can measure the loss using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html">https://scikit-learn.org/stable/ </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html">modules/generated/sklearn.metrics.log_loss.html</a>

<ol start="2">

 <li>After fitting a logistic model, you can access the weights it assigns to each feature in the data using its coef_ For each of the <em>i </em>models you generated, record the first such weight, which is the one the model applies to feature pixel000 in the input data. Produce a plot with the values of <em>i </em>as <em>x</em>-axis and with the feature weight as <em>y</em>. Place this plot into your PDF document, with a caption labeling it appropriately. Below the plot, discuss the results you are seeing; what do they show, and why?</li>

 <li>As in prior homework assignments, you will explore different values of regularization penalty for the logistic model.<sup>‡ </sup>Your code should explore a range of values for this parameter, using a regularly-spaced grid of values: C_grid = np.logspace(-9, 6, 31)</li>

</ol>

for C in C_grid:

# Build and evaluate model for each value C

For each such value of C create a model and fit it to the training data, and then compute the log loss of that model on the test data. Determine which value gives you the least loss on the test data. Record that value, along with the accuracy score of the model, in your PDF. Also include a table for the confusion matrix of that model on the test data.

<ol start="4">

 <li><em>.</em> Analyze some of the mistakes that your best model makes. Produce two plots, one consisting of 9 sample images that are <em>false positives </em>in the test set, and one consisting of 9 <em>false negatives</em>. You can display the images by converting the pixel data using the matplotlib function imshow(), using the Grey colormap, with vmin=0.0 and vmax=1.0. Place each plot into your PDF as a properly captioned figure. Below the figures, discuss the results you are seeing. What mistakes is the classifier making?</li>

 <li>Analyze all of the final weights produced by your classifier. Reshape the weight coefficients into a (28 × 28) matrix, corresponding to the pixels of the original images, and plot the result using imshow(), with colormap RdYlBu, vmin=-0.5, and vmax=0.5. Place this plot into your PDF as a properly captioned figure. Below it, discuss what it shows. Which pixels correspond to an 8 (have negative weights), and which correspond to a 9 (have positive weights)? Why do you think this is the case?</li>

</ol>

‡

As always should be the case, make sure to read the model documentation—in particular, note that the regularization parameter, C, is an <em>inverse </em>penalty (knowing this is important to interpreting and discussing your results).

<ul>

 <li></li>

</ul>

The use of that particular colormap will allow us to compare results across different submissions more easily. If you have a visual impairment that makes the output difficult to parse, please do feel free to replace it with another one that is more conducive to analysis (grayscale is always a possibility).

<h1>Part Two: Trousers v. Dresses</h1>

We have also provided some image data, in the same format as before, of trousers (output label 1) and dresses (output label 0).<sup>¶ </sup>We have again given you input and output data for a training set, along with <em>input data only </em>for a test set. Your task is to build a logistic regression classifier for this data. Your PDF for this part will describe your approach and the results you see.

When doing regression, you should explore different <em>feature transformations</em>, transforming the input features (in any way you see fit) that are given to the regression classifier. Your PDF should explain, as completely as you are able, what feature transformations you tried, and what processes you used to build your classifier (parameters you tried like regularization penalty, etc.), along with the reasoning behind your decisions. Your work should contain at least two figures comparing the results you get by regression using the original features of the data and some modified features. Your discussion should include the error rate on the testing data for various models, provided when you submit the model’s predictions on that data (see next section). Overall, the entire write-up for this part of the assignment should take 2–3 pages, including figures.

For this part of the assignment, <em>process </em>is more important than raw results. A well-designed set of tests, with coherent explanation and careful comparison of results will be worth more than something that achieves 0 error, but is not explained clearly.

You may use any transformations of the existing data you like. <em>Do not </em>use any additional sources of data; use only the input sets provided, and feature transformations on those sets. Be creative in thinking about how to transform data; some ideas you might consider (we encourage you to try other things as well):

<ul>

 <li>Consider things like histograms of parts of the data.</li>

 <li>Consider adding features that count overall numbers of white or black pixels.</li>

 <li>Consider adding features that capture spatial patterns in the original data.</li>

 <li>Consider exploring data augmentation, where you add to your data set via transformations of the data. For example, if you flip each image horizontally, you can double the training set size without needing fundamentally new data.</li>

</ul>

¶

These are taken from the <a href="https://github.com/zalandoresearch/fashion-mnist/">Fashion MNIST</a> data-set, originally released by Zalando research. Some noise has been added to the original data.

<h1>Part Three: Prediction submission</h1>

To test your regression classifiers for the clothing data, you can use them to generate the predicted probabilities for the data in the testing input set. You can then upload those predictions in the form of a text-file to the relevant Gradescope link, where the autograder will compare your results to the correct results (which it knows, while you don’t) and compute and display the overall quality of your predictions.

You can update your submission at any time up to the deadline, in attempts to improve predictive accuracy. A leaderboard will display the results, in order from best to worst. The autograder will also assign points based on overall leaderboard position.

The submission should be in the form of a text-file, yproba1_test.txt, containing one probability value (the probability of a positive binary label, 1) per example in the test input. Code like the following can be used to produce that file:

x_test = np.loadtxt(‘data_trouser_dress/troudress_test_x.csv’) yproba1_test = model.predict_proba(x_test)[:, 1] np.savetxt(‘yproba1_test.txt’, yproba1_test)