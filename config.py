"""
config.py - Configuration and Hyperparameters

Central configuration file for VN30 portfolio optimization framework.
All hyperparameters and constants are defined here for easy management.
"""


# Reproducibility
RANDOM_SEED = 42


# Risk-Free Rate
# Vietnam 10-year government bond yields (annual)
RISK_FREE_RATE = {
    2023: 0.030,
    2024: 0.028,
    2025: 0.032,
}


# Portfolio Construction
PORTFOLIO_SIZE = 10      # Number of stocks to select
MAX_WEIGHT = 0.25        # Maximum 25% per asset


# Neural Network Training
ANN_MAX_ITER = 5000
ANN_EARLY_STOPPING = True
ANN_N_ITER_NO_CHANGE = 10
ANN_VALIDATION_FRACTION = 0.20
ANN_CV_SPLITS = 2
ANN_N_JOBS = -1


# Grid Search Hyperparameters
ANN_HIDDEN_LAYERS = (
    (32,),
    (64,),
)

ANN_ACTIVATION = (
    'relu',
    'tanh',
)

ANN_ALPHA = (
    10.0,
    50.0,
    100.0,
)

ANN_LEARNING_RATE = (
    0.001,
    0.01,
)


# Feature Engineering Windows
# All windows are in months
MOMENTUM_3M = 3               # 3-month momentum
MOMENTUM_6M = 6               # 6-month momentum
VOLATILITY_WINDOW = 6         # 6-month volatility
MARKET_CORR_WINDOW = 12       # 12-month market correlation
DOWNSIDE_WINDOW = 6           # 6-month downside deviation
BETA_WINDOW = 12              # 12-month CAPM beta

# Lag to prevent look-ahead bias
FEATURE_LAG = 1
