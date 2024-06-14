# created by ilayd 06/14/24
''' @ARTICLE {Ranftl2022,
    author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
    title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    year    = "2022",
    volume  = "44",
    number  = "3"
}

@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ICCV},
	year      = {2021},
}
'''
from pyimagesearch.data_utils import get_dataloader
from pyimagesearch import config
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
import torch
import ssl

# disable ssl certificate verification (not recommended but it wasn't working without this for me)
ssl._create_default_https_context = ssl._create_unverified_context

# define the transformation for the test dataset: resize and convert to tensor
testTransform = Compose([Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), ToTensor()])
# load the test dataset using ImageFolder and apply the transformation
testDataset = ImageFolder(config.TEST_PATH, testTransform)
# get the dataloader for the test dataset
testLoader = get_dataloader(testDataset, config.PRED_BATCH_SIZE)

modelType = "DPT_Large"
# load the midas model from pytorch hub
midas = torch.hub.load("intel-isl/MiDaS", modelType, trust_repo=True)

# move the model to cpu
midas.to(config.DEVICE)
# set the model to evaluation mode
midas.eval()

# create an iterator for the test dataloader
sweeper = iter(testLoader)

print("Getting the test data")
# get the first batch of test data
batch = next(sweeper)
(images, _) =(batch[0], batch[1])
# move the images to the configured device
images = images.to(config.DEVICE)

# perform inference without computing gradients
with torch.no_grad():
    prediction = midas(images)

    # resize the prediction to the desired output size using bicubic interpolation
    prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=[384, 384], mode="bicubic",
                                                 align_corners=False).squeeze()

# convert the prediction to a numpy array for visualization
output = prediction.cpu().numpy()

# set up the plot grid dimensions
rows = config.PRED_BATCH_SIZE
cols = 2

# initialize the plot
axes = []
fig = plt.figure(figsize=(10,20))

# loop to create subplots
for totalRange in range(rows*cols):
    axes.append(fig.add_subplot(rows, cols, totalRange+1))

    # plot the original image in even positions
    if totalRange % 2 == 0:
        plt.imshow(images[totalRange//2].permute((1,2,0)).cpu().detach().numpy())
    # plot the predicted depth map in odd positions
    else:
        plt.imshow(output[totalRange//2])

# adjust layout for better spacing
fig.tight_layout()

# create output directory if it doesn't exist
if not os.path.exists(config.MIDAS_OUTPUT):
    os.makedirs(config.MIDAS_OUTPUT)

print("Saving the inference")
# save the plotted figure as an image file
outputFileName = os.path.join(config.MIDAS_OUTPUT, "output.png")
plt.savefig(outputFileName)