import os
import h2o
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


def poly_df(domain, density, gap=0):
    """Generates y = x^2 over a symmetrical domain with a symmetrical gap.

    Args:
        domain (int): Absolute value of max/min x-value.
        density (int): Number of data points per unit x.
        gap (int): Absolute value of max/min x-values to exclude.

    Returns:
        DataFrame with 'x' and 'y' columns sampling y = x^2.

    """
    data_limit = domain * density
    gap_limit = gap * density
    df = pd.DataFrame(columns=['x', 'y'])
    df['x'] = [1.0 / density * i for i in range(-data_limit, -gap_limit)] + \
              [1.0 / density * i for i in range(gap_limit, data_limit)]
    df['y'] = df.apply(lambda row: row['x']**2, axis=1)

    return df


def create_model(params):
    """Creates model based on parameters.

    Args:
        params (dict): Model parameters.

    Returns:
        None

    """
    # Create and train model
    model = H2ODeepLearningEstimator(**params)
    model.train(x='x', y='y', training_frame=train, validation_frame=val)

    # Run model prediction
    pred_train_val = model.predict(train_val).as_data_frame()
    pred_test = model.predict(test).as_data_frame()

    # Plot real data
    plt.plot(df_train_val['x'], df_train_val['y'], color='orange')
    plt.plot(df_test['x'][:len(df_test)/2], df_test['y'][:len(df_test)/2], color='orange')
    plt.plot(df_test['x'][len(df_test)/2:], df_test['y'][len(df_test)/2:], color='orange')

    # Plot model predictions
    plt.plot(df_train_val['x'], pred_train_val, color='blue')
    plt.plot(df_test['x'][:len(df_test)/2], pred_test[:len(pred_test)/2], color='blue')
    plt.plot(df_test['x'][len(df_test)/2:], pred_test[len(pred_test)/2:], color='blue')

    # Get model metrics
    test_rmse = '{:.1f}'.format(model.model_performance(test).rmse())
    train_val_rmse = '{:.1f}'.format(model.model_performance(train_val).rmse())
    samples_per_iteration = int(round(model.train_samples_per_iteration*1.258/len(df_train_val)))

    # Save model and plot
    name = '{0}_{1}_{2}_{3}'.format(test_rmse, train_val_rmse, model.activation, samples_per_iteration)
    h2o.save_model(model, os.path.join(save_dir, name))
    plt.savefig('{}.svg'.format(os.path.join(save_dir, name)))

    # Close plot
    plt.close()


def main():

    # Load hyperparameters
    hyper_params = [
        dict(
            model_id='dnn_poly',
            epochs=5000,
            hidden=[1000, 1000],
            activation='rectifier',  # 'maxoutwithdropout',# 'rectifierwithdropout', #
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
            train_samples_per_iteration=int(0.1 * len(df_train_val) / 1.258),

            # Controlling momentum
        ),
    ]

    # Loop through hyperparameters
    for params in tqdm(hyper_params):
        create_model(params)


if __name__ == '__main__':

    # Start h2o
    h2o.init(ip='localhost', port=65432, max_mem_size_GB=128, nthreads=70)

    # Indicate save folder
    save_dir = 'dnn_poly_diverge_output'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define characteristics of training and test data
    train_domain = 5
    test_domain = 10
    density = 10000

    # Generate training and test sets
    df_train_val = poly_df(train_domain, density)
    df_test = poly_df(test_domain, density, gap=train_domain)

    # Create train, validation, test frames
    column_types = ['real', 'real']
    train_val = h2o.H2OFrame(df_train_val, column_types=column_types)
    train, val = train_val.split_frame(ratios=[0.8])
    test = h2o.H2OFrame(df_test, column_types=column_types)

    # Run main function
    main()
