import torch
import os

# i am not doing any changes to the model so i'll only use the test set
BASE_PATH = "dataset"
TEST_PATH = os.path.join(BASE_PATH, "test_set")

IMAGE_SIZE = 512
PRED_BATCH_SIZE = 4

DEVICE = "cpu"

OUTPUT_PATH = "output"
MIDAS_OUTPUT = os.path.join(OUTPUT_PATH,"midas_output")