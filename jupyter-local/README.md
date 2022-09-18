# XGBoost - Local

## Cross Validation

```python
Usage
xgb.cv(
  params = list(),
  data,
  nrounds,
  nfold,
  label = NULL,
  missing = NA,
  prediction = FALSE,
  showsd = TRUE,
  metrics = list(),
  obj = NULL,
  feval = NULL,
  stratified = TRUE,
  folds = NULL,
  train_folds = NULL,
  verbose = TRUE,
  print_every_n = 1L,
  early_stopping_rounds = NULL,
  maximize = NULL,
  callbacks = list(),
  ...
)
```

### params 

- objective: reg:squarederror 또는 binary:logistic
- eta: 각 boosting step의 step size 
- max_depth: Tree의  maximum depth 
- nthread: Thread의 숫자

```python
hyperparameters = {
       "scale_pos_weight" : "29",    
        "max_depth": "3",
        "eta": "0.2",
        "objective": "binary:logistic",
        "num_round": "100",
}
```


## Reference

[XGBoost - Cross Validation](https://rdrr.io/cran/xgboost/man/xgb.cv.html)

