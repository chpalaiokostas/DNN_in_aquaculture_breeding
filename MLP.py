import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from scipy.stats.stats import pearsonr 
from sklearn.linear_model import LinearRegression

# data loading - preprocessing
# genotypes
file_with_genotypes = "geno.txt"
geno_data = pd.read_csv(file_with_genotypes, sep="\t", header=None, index_col=[0])
geno_data.head()

X = geno_data.to_numpy()

#phenotypes
traits = "pheno_data.txt"
pheno_data = pd.read_csv(traits, sep=" +", index_col=[0])
pheno_data.head()

y = pheno_data.loc[:,['Final_EBV']].to_numpy()

# create training, validation and test sets 
# In this case first 27000 records will be for training and validation
# 80% of the above will form the training set and 20% the validation set
# the rest 9000 will form the test set
X_train_full = X[:27000,:]
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, train_size=0.8)

# genotypes will be standardised
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# set up callbacks and the directory to save them 
# those can be used for Tensorboard and inspect the progress of the fitted model
logdir = os.path.join(os.curdir,"my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("rep1_run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(logdir,run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# set up Early stopping
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Example of fitting a multi-layer perceptron (MLP)
activation = "relu"
kernel_initializer = "he_normal"

model = keras.models.Sequential([
    keras.layers.Dense(300, activation=activation, kernel_initializer=kernel_initializer,input_shape=X_train.shape[1:]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation=activation, kernel_initializer=kernel_initializer),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation=activation, kernel_initializer=kernel_initializer),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation=activation, kernel_initializer=kernel_initializer),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation=activation, kernel_initializer=kernel_initializer),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=1e-03)
model.compile(loss='mean_squared_error',optimizer=optimizer)

tf.debugging.set_log_device_placement(True)
history = model.fit(X_train, y_train, epochs=100,
                   validation_data=(X_valid,y_valid), callbacks=[early_stopping_cb,tensorboard_cb])

mse_test = model.evaluate(X_test, y_test)
print(f"The mean squared error is: {mse_test}")
y_pred = model_cnn.predict(X_test)
accuracy = pearsonr(y_pred.reshape(-1),y_test.reshape(-1))[0]
print(f"The accuracy of the estimated breeding values is {accuracy}")

# Inspecting for bias
reg = LinearRegression(fit_intercept=True).fit(y_pred,y_test)
print(f"The potential bias of the estimated breeding values is: {reg.coef_}")