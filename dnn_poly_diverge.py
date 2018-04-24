import os
import h2o
import pandas as pd
import matplotlib.pyplot as plt
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
    epochs_per_iteration = 0
    try:
        epochs_per_iteration = params.pop('epochs_per_iteration')
        params['train_samples_per_iteration'] = int(epochs_per_iteration * len(df_train_val) / 1.258)
    except KeyError:
        pass
    model = H2ODeepLearningEstimator(**params)
    model.train(x='x', y='y', training_frame=train, validation_frame=val)

    # Run model prediction
    pred_train_val = model.predict(train_val).as_data_frame()
    pred_test = model.predict(test).as_data_frame()

    # Plot real data
    plt.figure()
    plt.plot(df_train_val['x'], df_train_val['y'], color='orange')
    plt.plot(df_test['x'][:len(df_test)/2], df_test['y'][:len(df_test)/2], color='orange')
    plt.plot(df_test['x'][len(df_test)/2:], df_test['y'][len(df_test)/2:], color='orange')

    # Plot model predictions
    plt.plot(df_train_val['x'], pred_train_val, color='darkgreen')
    plt.plot(df_test['x'][:len(df_test)/2], pred_test[:len(pred_test)/2], color='darkblue')
    plt.plot(df_test['x'][len(df_test)/2:], pred_test[len(pred_test)/2:], color='darkblue')

    # Get model metrics
    test_rmse = '{:.1f}'.format(model.model_performance(test).rmse())
    train_val_rmse = '{:.1f}'.format(model.model_performance(train_val).rmse())

    # Define name for saving model, model parameters, plot
    name = '{0}_{1}_{2}_{3}_{4}'.format(test_rmse, train_val_rmse, model.activation, model.hidden, epochs_per_iteration)

    # Save model and model parameters
    h2o.save_model(model, os.path.join(save_dir, name))
    with open('{}.txt'.format(os.path.join(save_dir, name)), 'wb') as f:
        f.write('test_rmse = {}\n'.format(test_rmse))
        f.write('train_val_rmse = {}\n\n'.format(train_val_rmse))
        f.write('epochs = {}\n'.format(model.epochs))
        f.write('hidden = {}\n'.format(model.hidden))
        f.write('activation = {}\n'.format(model.activation))
        f.write('hidden_dropout_ratios = {}\n'.format(model.hidden_dropout_ratios))
        f.write('l1 = {}\n'.format(model.l1))
        f.write('l2 = {}\n'.format(model.l2))
        f.write('max_w2 = {}\n'.format(model.max_w2))
        f.write('stopping_rounds = {}\n'.format(model.stopping_rounds))
        f.write('stopping_tolerance = {}\n'.format(model.stopping_tolerance))
        f.write('stopping_metric = {}\n'.format(model.stopping_metric))
        f.write('score_interval = {}\n'.format(model.score_interval))
        f.write('score_duty_cycle = {}\n'.format(model.score_duty_cycle))
        f.write('epochs_per_iteration = {}\n'.format(epochs_per_iteration))
        f.write('adaptive_rate = {}\n'.format(model.adaptive_rate))
        f.write('rho = {}\n'.format(model.rho))
        f.write('epsilon = {}'.format(model.epsilon))
        f.write('rate = {}\n'.format(model.rate))
        f.write('rate_annealing = {}\n'.format(model.rate_annealing))
        f.write('momentum_start = {}\n'.format(model.momentum_start))
        f.write('momentum_ramp = {}\n'.format(model.momentum_ramp))
        f.write('momentum_stable = {}\n'.format(model.momentum_stable))
        f.write('initial_weight_distribution = {}\n'.format(model.initial_weight_distribution))

    # Save plot
    plt.savefig('{}.svg'.format(os.path.join(save_dir, name)))

    # Show plot
    plt.show()


def main():

    # Load hyperparameters
    params = dict(
        model_id='dnn_poly',
        epochs=5000,
        hidden=[300, 300],
        activation='tanh_with_dropout',
        hidden_dropout_ratios=[0.5, 0.5],
        l1=1e-4,
        l2=1e-4,
        max_w2=0.2,
        stopping_rounds=10,
        stopping_tolerance=1e-4,
        stopping_metric='rmse',

        # Control scoring epochs
        score_interval=0,
        score_duty_cycle=1,
        shuffle_training_data=False,
        replicate_training_data=True,
        epochs_per_iteration=1,

        # Control momentum
        rate=1e-5,
        rate_annealing=1e-10,
    )

    # Create model
    create_model(params)


if __name__ == '__main__':

    # Start h2o
    h2o.init(ip='localhost', port=54321, max_mem_size_GB=128, nthreads=70)

    # Indicate save folder
    save_dir = 'dnn_poly_diverge_output'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define characteristics of training and test data
    train_domain = 10
    test_domain = 20
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
