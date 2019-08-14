"""

Project: Forecasting graduate admissions with logistic regression

Name: Kevin Trinh
Date: 7/29/19

"""

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# read in the data set as a Panda dataframe
admission_dataframe = pd.read_csv("Admission_Predict_Ver1.1.csv", sep=",")

# shuffle UCLA Graduate Admission data set
admission_dataframe = admission_dataframe.reindex(
    np.random.permutation(admission_dataframe.index))

def preprocess_features(admission_dataframe):
    """Prepares input features from UCLA Graduate Admission data set.

    Args:
      admission_dataframe: A Pandas DataFrame expected to contain data
        from the UCLA Graduate Admission data set.
    Returns:
      A DataFrame that contains the features to be used for the model, including
      synthetic features.
    """
    selected_features = admission_dataframe[
      ["GRE_Score",
     "TOEFL_Score",
     "University_Rating",
     "SOP",
     "LOR",
     "CGPA",
     "Research"]]
    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(admission_dataframe):
    """Prepares target features (i.e., labels) from UCLA Graduate Admission data set.

    Args:
      admission_dataframe: A Pandas DataFrame expected to contain data
        from the UCLA Graduate Admission data set.
    Returns:
      A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Create a boolean categorical feature representing whether the
    # chance of admit is above a set threshold of 50%.
    threshhold = 0.50
    output_targets["Admission"] = (
      admission_dataframe["Chance of Admit"] > threshhold).astype(float)
    return output_targets




# Choose the first 300 (out of 500) examples for training.
training_examples = preprocess_features(admission_dataframe.head(300))
training_targets = preprocess_targets(admission_dataframe.head(300))

# Choose the next 100 (out of 500) examples for validation.
validation_examples = preprocess_features(admission_dataframe[300:400])
validation_targets = preprocess_targets(admission_dataframe[300:400])

# Chose the last 100 (out of 500) examples for testing.
test_examples = preprocess_features(admission_dataframe.tail(100))
test_targets = preprocess_targets(admission_dataframe.tail(100))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())

def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
      input_features: The names of the numerical input features to use.
    Returns:
      A set of feature columns.
    """
    return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                            
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    """Trains a linear classification model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing one or more columns from
        `admission_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `admission_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `admission_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `admission_dataframe` to use as target for validation.

    Returns:
      A `LinearClassifier` object trained on the training data.
    """

    # set number of periods to see evolution of our model
    periods = 15
    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)  
    linear_classifier = tf.estimator.LinearClassifier(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["Admission"], 
                                          batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["Admission"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["Admission"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
        # Take a break and compute predictions.    
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.5f" % (period, training_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.figure(1)
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return linear_classifier

# train our model and examine performance on validation set
execute_training = "y"
while (execute_training == "y"):
    
    # train our data
    if (execute_training == "y"):
        
        # prompt user for hyperparameters
        print("Enter value for learning rate (order of 0.000001 recommended):")
        learning_rate = input()
        learning_rate = float(learning_rate)
        print("Enter value for steps:")
        steps = input()
        steps = int(steps)
        print("Enter value for batch size:")
        batch_size = input()
        batch_size = int(batch_size)
        
        # train our model with logistic regression
        linear_classifier = train_linear_classifier_model(
            learning_rate,
            steps,
            batch_size,
            training_examples=training_examples,
            training_targets=training_targets,
            validation_examples=validation_examples,
            validation_targets=validation_targets)
    
        # examine model accuracy, ROC, and AUC
        predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                          validation_targets["Admission"], 
                                                          num_epochs=1, 
                                                          shuffle=False)
    
        evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
    
        print("AUC on the validation set: %0.5f" % evaluation_metrics['auc'])
        print("Accuracy on the validation set: %0.5f" % evaluation_metrics['accuracy'])
        
        # prompt user to retrain data
        print("Would you like to retrain your data with new hyperparameters? (y/n)")   
        execute_training = input()
        

# test model accuracy, ROC, and AUC
predict_test_input_fn = lambda: my_input_fn(test_examples, 
                                                test_targets["Admission"], 
                                                num_epochs=1, 
                                                shuffle=False)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_test_input_fn)

test_probabilities = linear_classifier.predict(input_fn=predict_test_input_fn)

print("AUC on the test set: %0.5f" % evaluation_metrics['auc'])
print("Accuracy on the test set: %0.5f" % evaluation_metrics['accuracy'])

# Compare our model against a random classifier
test_probabilities = np.array([item['probabilities'][1] for item in test_probabilities])

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    test_targets, test_probabilities)
plt.figure(2)
plt.plot(false_positive_rate, true_positive_rate, label="our model")
plt.plot([0, 1], [0, 1], label="random classifier")
plt.title("ROC")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc=2)