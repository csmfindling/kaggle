# TODO

## High priority

 - [ ] Create a submit script from a trained network
 - [ ] Create a valid split of the train dataset, monitor the loss and f2 score for this dataset while training. 
 - [ ] Compare the f2 score of the validation set to the "true" f2 measured when submitting to kaggle

## Medium priority

 - [ ] Data augmentation: implement random rotation (+ corresponding zoom so that the resulting images do not have black parts) 
 - [ ] Implement hdf5 using all channels from tiff images

## Low priority

 - [ ] Use a combination of cross entropy and differentiable f2 as a loss, find a smart way for weighting each term.
 - [ ] Implement a pretrained resnet
 - [ ] Open a bank account in a tax heaven for the prizes
