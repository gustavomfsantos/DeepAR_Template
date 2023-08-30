# DeepAR_Template
Template for DeepAR using Gluonts library

This project uses a Sales Dataset from Kaggle (https://www.kaggle.com/code/aremoto/retail-sales-forecast/input) with weekly frequency.
The first step for a pipeline that uses DeepAR is ajust the data format in a way that it fits the algorithm. This dataset contains Time series for each department in each Store, so it is a Store>Department hierarchy level. However we will only work on the minor level. For that, it is create a key matching Store-Department, in order do identify each distinct time series inside the dataset. The final Dataframe will have three columns: Date, Key and Weekly Sales Value, that is the model target and each row is a single observation in time for a specific key. Note that this aproach does not uses other features that not the target.

The GluonTS requires a minimum of two columns: the timestamp and the target. Our dataset will have a indentifier column beside these two. To build the train dataframe we will create a PandasDataset using the 'from_long_dataframe' function. The first argument is the dataframe. Then we pass the name of the column with the time series values as target. The item_id is the name of the column with the time series identifier. In our case we have one time series per department for each store, but this could be SKUs, stores, a mix of all of them, etc. The timestamp is the name of the column with the timestamp. Finally, we pass the freq argument, which is the frequency of the time series, which is Week starting on Friday (W-FRI).

Before we build and train a DeepAR model, it’s important to understand which levers we can pull to improve the model’s performance and their default values.
These are hyperparameters you can tune manually or with my favorite methods: Bayesian Optimization and Random Search.

# DeepAR’s Important Hyperparameters
num_layers:

This is the number of layers in the RNN. Just like a feedforward neural network, a recurrent neural network can have multiple layers to learn more complex patterns as it goes deeper. More layers means more capacity to learn, but also more time to train and a higher risk of overfitting. The default value is 2.

hidden_size:

This is the number of units (neurons) in each layer of the RNN. Again, you can think about it just like the hidden layers in a feedforward neural network. So if you have 2 layers of 10 units each, you have a total of 20 units. The default value is 40, which means 40 units in each layer, not in total.

context_length:

This is the number of time steps the model will use as inputs to predict the next time step. For example, if you have a daily time series and you set context_length=30, the model will use the last 30 days to predict the next day. The default value is None which means the model will use the same size as the prediction horizon (how many steps ahead you want to predict).

prediction_length:

This is the number of time steps you want to predict, the same as the prediction horizon. This is usually not something you tune, but you decide based on the problem you are trying to solve. If you need to manage the inventory of a store for the next 30 days, you would set prediction_length=30. There is no default value for this.

lr:

This is the learning rate of the optimizer. An extremely important hyperparameter in any deep learning model, the learning rate controls how much the model learns at each step. A low learning rate will make the model learn slowly, but it will also make it more stable. A high learning rate will make the model learn faster, but it can make it jump around the minimum and never converge. The default value is 0.001.

weight_decay:

This is the regularization parameter for the weights during training. A higher value means more regularization, which will help prevent overfitting, but if you set it too high, the model will be too limited to learn anything. The default value is 1e-8.

dropout_rate:

This is another regularization parameter. Dropout is a very successful regularization technique in deep learning that randomly sets a percentage of the units in a layer to zero during training. The percentage is controlled by the dropout_rate parameter, which by default is 0.1. Again, higher values mean more regularization, but if you set it too high, the model will have a hard time learning anything.

Source:https://forecastegy.com/posts/multiple-time-series-forecasting-with-deepar-in-python/
###

Now that we prepared the data and checked the hyperparameters, we can train a DeepAR model.

The first thing we do is create an instance of the DeepAREstimator class. We pass the frequency of the time series, the prediction horizon, and any hyperparameter we want to change from the default values.

Then we call the train method to train the model. Notice that, different from the usual scikit-learn API, the train returns a predictor object, which contains the trained model we need to make predictions. You can pass the argument num_workers to use multiple CPU cores to process the data in parallel.

Now that we have a trained model, we can use it to make predictions. The predict method from the predictor object takes a dataset as input and returns a generator with the predictions.

A important point to note is that the predictions are made on top of the train dataframe created. In many other libraries this would predict the examples in the training set, but in this case it will use the last context_length time steps from it to predict the prediction_length time steps ahead in a recursive fashion. If you never heard of recursive or autoregressive predictions in time series forecasting with machine learning, it just means that the model will predict one step ahead, then use the prediction as input to predict the next step, and so on. So the predictions become observations in the input sequence for the next prediction until we reach the prediction_length.

Because DeepAR returns 100 samples for each time step (this can be set to different value), we need to take the mean of the predictions to get the most likely value. The cool thing is that we can also create uncertainty intervals for the predictions. In this template I used the 20th and 80th percentiles of the samples. At the end we merge the predictions, the interval bound and the real values to measure accuracy achivied from the model. Also, all negative values and NAN values were set to zero.



