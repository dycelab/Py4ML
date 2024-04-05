# Import the Optuna library for hyperparameter optimization.
import optuna

# Define a function to train and evaluate the XGBoost model with given hyperparameters.
def train_evaluate(x_tr, y_tr, x_val, y_val, n_tree, max_depth, lr, sub_sp, tree_method, max_bin, colsample):
  # Initialize the XGBoost Regressor model with specific hyperparameters and train it on the training data.
  fit_md = XGBRegressor(n_estimators=n_tree, max_depth=max_depth, learning_rate=lr, 
                        min_samples_leaf=2, colsample_bytree=colsample, subsample=sub_sp, 
                        tree_method=tree_method, max_bin=max_bin).fit(x_tr, y_tr)
  
  # Predict the validation data.
  pred_val = fit_md.predict(x_val)
  
  # Calculate and return the explained variance score as the evaluation metric.
  f1_val = explained_variance_score(y_val, pred_val)
  return f1_val

# Define the objective function that Optuna will optimize.
def objective(trial):
    # Define the hyperparameter space for the XGBoost model.
    params = {
             'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.2),
              'n_tree': trial.suggest_int("n_tree", 100, 300, step=100),
              'max_depth': trial.suggest_int("max_depth", 4, 8, step=2),
              'subsample': trial.suggest_float("subsample", 0.4, 1.0, step=0.2),
              'tree_method': trial.suggest_categorical('tree_method', ['hist', 'approx', 'exact']),
              'max_bin': trial.suggest_int('max_bin', 100, 250, step=50),
              'colsample_bytree': trial.suggest_float("colsample_bytree", 0.2, 0.8, step=0.2)
              }
    # Extract parameters for the model training and evaluation.
    n_tree = params['n_tree']
    max_depth = params['max_depth']
    sub_sp = params['subsample']
    lr = params['learning_rate']
    tree_method = params['tree_method']
    max_bin = params['max_bin']
    colsample = params['colsample_bytree']
    
    # Train and evaluate the model using the defined function and return the evaluation metric.
    accuracy = train_evaluate(x_tr, y_tr, x_val, y_val, n_tree, max_depth, lr, sub_sp, tree_method, max_bin, colsample)
    return accuracy

# Start the timer for the parameter tuning process.
start_time = time.time()

# Create an Optuna study object to optimize the objective function.
para_opt = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())

# Start the optimization process with 100 trials and 8 parallel jobs.
para_opt.optimize(objective, n_trials=100, n_jobs=8)

# Collect and save the results of the trials in a dataframe.
output = para_opt.trials_dataframe()

# Print the total time taken for parameter tuning and save the results to a CSV file.
print('Parameter tuning time: {}'.format(time.time()- start_time))
output.to_csv('Paramter_tunning_XGBM_DC.csv', index=False)
