BASE_PATH = "data\\"
TRAIN_FOLDER = "\\Train"
TEST_FOLDER = "\\Test"
TARGET_FOLDER = "data"
MODEL_NAME = "\\trained_model.pkl"

INPUT_FEATURES_ARRAY = "X"
LABEL_ARRAY = "y"
BATCH_SIZE = 32

LEARNING_RATE = 0.01
N_EPOCHS = 50

VALIDATE_TRAINING = True

TEST_SINGLE = False
TEST_IMG_LABEL = 5
TEST_IMG = 'data\\greyscale5.jpg'

'''LABEL_TO_IDX = {
        'L': 0,
        'ManoPlana': 1,
        'OK': 2,
        'Palma': 3,
        'PulgarAbajo': 4,
        'PulgarArriba': 5,
        'Puny': 6,
        'RockAndRoll': 7
}'''

IDX_TO_LABEL = {
        0: 'C',
        1: 'Down',
        2: 'Fist',
        3: 'ThumbDown',
        4: 'L',
        5: 'OK',
        6: 'Palm',
        7: 'Thumb'
}

LABEL_TO_IDX = {
        'C': 0,
        'Down': 1,
        'Fist': 2,
        'ThumbDown': 3,
        'L': 4,
        'OK': 5,
        'Palm': 6,
        'Thumb': 7
}