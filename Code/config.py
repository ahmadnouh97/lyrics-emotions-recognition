import os


class Config:
    RAW_DATA_DIR = os.path.join('data', 'raw')
    RAW_DATA_PATH = os.path.join('data', 'raw', 'habibi.csv')
    PROCESSED_DATA_DIR = os.path.join('data', 'processed')
    PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'habibi.csv')
    EVAL_DATA_DIR = os.path.join('data', 'eval')
    EVAL_DATA_PATH = os.path.join(EVAL_DATA_DIR, 'eval.json')

    PARAMS_PATH = os.path.join('params.yaml')
    MODEL_PATH = os.path.join('model', 'lyrics_model.pkl')
    METRICS_DIR = os.path.join('metrics')

    PLOT_DIR = os.path.join('plots')
    PLOT_TRAINING_CURVE_ACC_FILE = os.path.join(PLOT_DIR, 'accuracy.png')
    PLOT_TRAINING_CURVE_LOSS_FILE = os.path.join(PLOT_DIR, 'loss.png')

    TENSORBOARD_DIR = os.path.join('logs')
