---
jupyter:
  jupytext:
    default_lexer: python
    formats: ipynb,md
    text_representation:
 j     extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Assignment 3 Analysis

## Question 1

Same as last week, the import was not difficult, especially because I knew exactly what I needed to import the data
But for the whitespace inside the TotalCharges column, I had to look for a method that I could use to find the values. 
I ended up settling on the to_numeric method. It's self explanatory but it helps to change the values in one run. 


## Question 2

For finding what columns required imputation, I used to website to get a grasp of the data and then used methods like info to 
see what columns contained things like whitespace. I ended up finding TotalCharges being the only column that actually requires
revision. 
Then I checked all rows and saw no values were null anymore. 

For the imputation, I used median on the numerical columns and most_frequent for the categorical columns.
I used median for the numerical columns because some columns are heavily skewed, so mean would have been a bad choice.
The categorical imputer was just to protect the data.

## Question 3

When using column transfomations and pipelines, it essentially just allows you to import all the methods you are using
to process the data into one call. This way, I can check the pipeline definition and the the CT definition and see that
I am using __ imputer for num column and __ encoder for categorical columns. It just streamlines the process to there isn't a 
bunch of bulky code that is hard to read. 
They also allow us to guard from data leakage. If you do everything by hand, it would be super easy to call fit_transform at the
wrong time.

The CT especially allows us to treat certain columns a certain way. In this case, we are using it to differentiate between
categorical and numeric columns in the preprocess step.

## Question 4

The difference between both scalers in this case was minimul to none. It didn't make enough of an impact on either side
to reasonably HAVE to choose one. I just chose Standard Scaler because I've used it before. 

## Question 5

Using logistic regression differs from something like loss='hinge' because the way it calculates the gd is different. 
Log loss uses a sigmoid function while hinge uses an svm. The probability outputs also help because it doesn't rely on a 
hard 0.5 cutoff. The ROC-AUC curve gives a more reliable picture in the end because the data is imbalanced.

## Question 6

The features that show the strongest relationship are tenure (-1.4) and contract type which had great results from task 2's churn rates.
This would imply that the business should 
change current customers contract type so that they stay longer. Length of stay has a high correlation to not leaving (of course)
so if they want to maintain stay with their customers, they should change the outgoing contracts to last longer.
