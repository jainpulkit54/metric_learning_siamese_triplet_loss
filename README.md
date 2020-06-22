# metric_learning_siamese_triplet_loss

A triplet network takes in three images as input i.e., an anchor image, a positive image (i.e., image having label same as the anchor) and a negative image (i.e., image having label different from the anchor). The objective here is to learn embeddings such that the positive images are closer to the anchor as compared to the negative images. The same can be pictorically represented using the image given below: <br>

![](images/anchor_negative_positive.png) <br>

Source: Schroff, Florian, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. CVPR 2015 <br>

Moreover, the triplet loss is mathematically expressed as:<br>

![](images/triplet_loss.png) <br>

The embeddings obtained for the MNIST dataset are as follows: <br>
For the MNIST Train Set:<br>
![](outputs_MNIST/embeddings_trainset.png)<br>
For the MNIST Test Set:<br>
![](outputs_MNIST/embeddings_testset.png)<br>

The embeddings obtained for the FMNIST dataset are as follows: <br>
For the FMNIST Train Set:<br>
![](outputs_FMNIST/embeddings_trainset.png)<br>
For the FMNIST Test Set:<br>
![](outputs_FMNIST/embeddings_testset.png)<br>

As it is evident from the above results, the embeddings obtained for different classes are separable clearly in MNIST but not so much in FMNIST, the reason being that the classes in FMNIST are not very different as in MNIST and also, training has been done by choosing random combinations of triplets (i.e., it is highly probable that the network has seen the easy samples quite a lot and did not see the hard samples otherwise it would have learned better). Random selection of triplets for training is not the right way to train, hence in literature a modification of this technique is used called as "Online Triplet Loss" with "Batch All" and "Batch Hard" strategy.
