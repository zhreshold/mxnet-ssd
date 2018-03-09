XML_FOLDER = 'Annotations'
GENERATED_DATA = 'GeneratedData'
TRAIN_FOLDER = 'Train'
VAL_FOLDER = 'Val'
TEST_FOLDER = 'Test'
LABEL = 'Label'

# Dimensions of the raw flag height and width
FLAG_HEIGHT = 144
FLAG_WIDTH = 224

# There are 202599 images in my CelebA dataset. Give this value appropriately.
CELEBA_TOTAL_FILES = 202599     # Directly hardcoded to save memory

MIN_FLAGS = 1
MAX_FLAGS = 2  # Currently supports upto 2 Maximum flags in one image. 

BORDER_WHITE_AREA = 40 # How much percent of card should be covered with white area.