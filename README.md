# Toy Problems

We investigate the performance of the Muon optimizer on a two layer neural network 
$$        
    y = W_1  \text{act}(W_2x + b_1) + b_2
$$
for an activation function $\text{act} \in \{\text{ReLU}, \text{tanh}, \text{sigmoid} \}$ 

some core questions of interest are:
- Is it really necessary to separate the weights and biases into two groups and use adamw for the non matrix parameters? 
    - can we consider [W | b] instead and just use Muon for everything? 
    - On the LM task it's suggested that Muon shouldn't be used for the embedding or final linear layer... there are only 2 layers here but how does it look for Muon on the final (second) linear layer vs adam
    - Is it really about Muon updating with matrix info or just that we have seperate learning rates which improves things?
- In this very simple loss landscape can we garner any insights into what kind of step Muon is actually doing?
- How can initializiation of the parameters be improved for orthogonalized gradients?
- How does performance change when considering the undertraining regime (like the LM pretraining task) vs overtraining regime
- Is there an optimal way to select the batch size? On LM pretraining it's suggested that Muon needs big batches but does this hold true in the simplest setting?
- How does full orthogonalization vs Newton Schulz look? We know the approximation of the orthogonalization is a little bit dicey.     