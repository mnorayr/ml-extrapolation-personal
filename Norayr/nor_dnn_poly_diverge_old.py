<<<<<<< HEAD
### test 3 
=======
>>>>>>> 4e2403de20cc3cda36af57b8044ca854378b04f6
import h2o
import pandas as pd
import matplotlib.pyplot as plt
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


<<<<<<< HEAD
### Start h2o
h2o.init(ip='localhost', port=65432, max_mem_size_GB=128, nthreads=70)

### Define columns and types
columns = ['x', 'y']
column_types = ['real', 'real']

### Generate dataset for y = x^n
n=3
=======
## Start h2o
h2o.init(ip='localhost', port=65432, max_mem_size_GB=128, nthreads=70)

## Define columns and types
columns = ['x', 'y']
column_types = ['real', 'real']

## Generate dataset for y = x^n
n=2
>>>>>>> 4e2403de20cc3cda36af57b8044ca854378b04f6
df = pd.DataFrame(columns=columns)
df['x'] = [0.001 * i for i in range(-10000, 10000)]
df['y'] = df.apply(lambda row: row['x'] ** n, axis=1)
df.to_csv("~/data/poly_train_{!s}.csv".format(n), index=False, header=True)

<<<<<<< HEAD
### Create test set with domain outside training
=======
## Create test set with domain outside training
>>>>>>> 4e2403de20cc3cda36af57b8044ca854378b04f6
test_df = pd.DataFrame(columns=columns)
test_df['x'] = [0.001 * i for i in range(-15000, 15000)]
test_df['y'] = test_df.apply(lambda row: row['x'] ** n, axis=1)
test_df.to_csv("~/data/poly_test_{!s}.csv".format(n), index=False, header=True)

<<<<<<< HEAD
### Create H2OFrame
=======
## Create H2OFrame
>>>>>>> 4e2403de20cc3cda36af57b8044ca854378b04f6
h2o.remove_all()
hf = h2o.H2OFrame(df, column_types=column_types)
train, val = hf.split_frame(ratios=[0.8])
test = h2o.H2OFrame(test_df, column_types=column_types)

<<<<<<< HEAD
### Create model
=======
## Create model
>>>>>>> 4e2403de20cc3cda36af57b8044ca854378b04f6
predictors = 'x'
response = 'y'
model = H2ODeepLearningEstimator(
    model_id='dnn_poly',
    epochs=5000,
    hidden=[1000, 1000],
    activation= 'rectifier',#'maxoutwithdropout',# 'rectifierwithdropout', #
    # hidden_dropout_ratios=[0.5, 0.5],
    l1=1e-6,
    l2=1e-6,
    max_w2=10.,
    stopping_rounds=10,
    # stopping_tolerance=1e-4,
    stopping_metric='rmse',

    # Control scoring epochs
    score_interval=0,
    score_duty_cycle=1,
    shuffle_training_data=False,
    replicate_training_data=True,
    train_samples_per_iteration=int(0.1 * len(df) / 1.258),

    # Controlling momentum
)
model.train(x=predictors, y=response, training_frame=train, validation_frame=val)

<<<<<<< HEAD
### Predicting
test_df['predict'] = model.predict(test).as_data_frame()
print model.model_performance(test)

### Plot results
=======
## Predicting
test_df['predict'] = model.predict(test).as_data_frame()
print model.model_performance(test)

## Plot results
>>>>>>> 4e2403de20cc3cda36af57b8044ca854378b04f6
driverless_predict = pd.DataFrame(columns=['x', 'y'])
driverless_predict['x'] = test_df['x']
driverless_predict['y'] = pd.read_csv('/home/norayrm/Downloads/test_preds.csv')
plt.plot(test_df['x'], test_df['y'])
plt.plot(test_df['x'], test_df['predict'])
plt.plot(driverless_predict['x'], driverless_predict['y'])
plt.show()
<<<<<<< HEAD

########################

=======
>>>>>>> 4e2403de20cc3cda36af57b8044ca854378b04f6
