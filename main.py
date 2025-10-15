r"""   
We investigate the performance of the Muon optimizer on a two layer neural network 
        y = W2 @ act(W1@x + b1) + b2

some core questions of interest are:
- Is it really necessary to separate the weights and biases into two groups and use adamw for the non matrix parameters? i.e can we consider [W | b] instead?
- How can initializiation of the parameters be improved for orthogonalized gradients?
- How does performance change when considering the undertraining regime (like the LM pretraining task) vs overtraining regime
- Is there an optimal way to select the batch size? On LM pretraining it's suggested that Muon needs big batches but does this hold true in the simplest setting?
- How does full orthogonalization vs Newton Schulz look? We know the approximation of the orthogonalization is a little bit dicey.     
"""