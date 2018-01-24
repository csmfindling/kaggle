# Summary of experiment :

### Model
Launch vgg 16 with pretrained weights for the convoluational layers

### Loss
Used the binary cross-entropy loss.

### Results
F2 score on training set without validation set monitoring : .85


############################ lr = 1e-3 ############################

### Logs
Using gpu device 0: GeForce GTX 1080 (CNMeM is disabled, cuDNN 5105)
The weights of 16 layers were updated
train score : 0.806233669943
valid score : 0.835250827824
new best score : 0.835250827824
patience : 0


train score : 0.819248715418
valid score : 0.805068670111
patience : 1


train score : 0.81811666195
valid score : 0.833545696827
patience : 2


train score : 0.713671881585
valid score : 0.544066102648
patience : 3


train score : 0.684981044949
valid score : 0.712519848411
patience : 4


train score : 0.71102423362
valid score : 0.712546438084

### Remarks
Maybe refine the learning rate will give better results

### Submit 
F2 score : .85


############################ lr = 1e-4 ############################

## logs
Using gpu device 0: GeForce GTX 1080 (CNMeM is disabled, cuDNN 5105)
The weights of 16 layers were updated
train score : 0.896381018802
valid score : 0.893658077473
patience : 0


train score : 0.898071486521
valid score : 0.896267856821
patience : 0


train score : 0.898752654327
valid score : 0.896172362483
patience : 1


train score : 0.898848168418
valid score : 0.89584512804
patience : 2


train score : 0.900343655762
valid score : 0.897248717523
patience : 0


train score : 0.900945944503
valid score : 0.895451108929
patience : 1


train score : 0.901693353612
valid score : 0.898277501381
patience : 0


train score : 0.902664431312
valid score : 0.900370034074
patience : 0


train score : 0.904163741283
valid score : 0.901348206389
patience : 0


train score : 0.903707590894
valid score : 0.89978018826
patience : 1


train score : 0.904543918406
valid score : 0.900669778163
patience : 2


train score : 0.905446235406
valid score : 0.900504267731
patience : 3


train score : 0.905233277905
valid score : 0.90064934943
patience : 4


train score : 0.906157807554
valid score : 0.898766552315