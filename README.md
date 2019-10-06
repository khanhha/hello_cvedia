Hi Rodrigo,

I am very passionate about the vision of CVEDIA about synthetic data for deep learning. From my deep learning experience, data is the key to a high-quality model. Unfortunately, achieving a good data-set that could cover many edge cases is very hard and expensive in practice. Therefore, I firmly believe that synthetic data will play a crucial role in the future of deep learning, which I want to be a part of.   

However, every journey comes down to a good match. Therefore, to convince you that I can be a valuable addition to the vision of CVEDIA, I wrote an article to describe to you the challenges that I solved so far.

From my portfolio, I chose my latest project about images-based 3D human estimation because the skills I gained through this project will be valuable to projects in CVEDIA.  If you want to know more about other deep learning projects that I tackled such as crack detection, 3D textured head estimation, please feel free to let me know.


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

- [Overview](#overview)
- [Challenges](#challenges)
  - [Data synthesis](#data-synthesis)
    - [Merge 3D human data-sets on the Internet.](#merge-3d-human-data-sets-on-the-internet)
    - [New 3D human shapes synthesis based on the PCA model (Principal component analysis).](#new-3d-human-shapes-synthesis-based-on-the-pca-model-principal-component-analysis)
    - [New 3D human pose variants synthesis.](#new-3d-human-pose-variants-synthesis)
    - [Camera parameters](#camera-parameters)
  - [Challenges about training](#challenges-about-training)
    - [Model architecture: combine three models front, side, and fusion.](#model-architecture-combine-three-models-front-side-and-fusion)
    - [Two weight initialization techniques that boost accuracy](#two-weight-initialization-techniques-that-boost-accuracy)
    - [Design loss functions.](#design-loss-functions)
    - [Use lower learning rates for earlier layers help improve accuracy.](#use-lower-learning-rates-for-earlier-layers-help-improve-accuracy)
- [Conclusion](#conclusion)

<!-- /code_chunk_output -->

# Overview
Before explaining the challenges, I will briefly describe the general context of the project to give you a better picture.

The deep learning model described in this article is used for a virtual shopping application. It allows users to try on clothes on their 3D body estimated from their front/side pictures,  as shown in the first stage in the below diagram. In the second stage, a color segmentation map is estimated using other deep learning models in combination with image processing techniques.

These two segmentation maps will be then calibrated and passed to a deep learning model that predicts 50 PCA parameters that encode the 3D shape of the customer. These PCA parameters will be used to reconstruct the final 3D human mesh, as shown in the fourth stage.

![](/assets/images/3dhm/2019-564ea8d3.png)

# Challenges

In this part, I will describe the challenges that I tackled during this project. The experience that I gained through these challenges helps build my confidence in solving future machine learning problems.

To make it easier for you to follow, I classified them into two main types of challenges: the ones that relate to the data synthesis stage and the other ones about training techniques.

## Data synthesis

### Merge 3D human data-sets on the Internet.

In this challenge, I helped double the size of the training data from 3000 meshes to around 7000 meshes by merging different 3D human data-sets in the market. In contrast to images data-sets, combining different 3D human mesh data-sets are more complicated because of varying topology (different vertex number, triangle number, order).

To tackle this challenge, I implemented a registration algorithm that aligns/deforms the meshes to the same topology. The registration algorithm is implemented in a way that takes into account 72 3D human landmarks so that all the meshes are aligned correctly, as depicted by the red and blue segments in the below figure.
![](/assets/images/3dhm/2019-46ff3704.png)  

### New 3D human shapes synthesis based on the PCA model (Principal component analysis).

This challenge is about synthesizing more diverse human shapes, from skinny to fat, short to tall, etc. In the beginning, I just trained a deep learning model from 7000 meshes in the merged data-sets, which causes early over-fitting due to the lack of diverse data.

To resolve this problem, I first trained a PCA model that compresses human representation from the mesh format of 10.000 vertices to 50 principal components. The standard deviations along these principal components form a multivariate Gaussian distribution, which is then randomly sampled to create new PCA parameters representing new human shapes.

Thanks to this technique, I increase the size of training data from 7000 meshes to 100.000 meshes, which reduces the loss on the test-set by 20 percent.

### New 3D human pose variants synthesis.

This challenge is about synthesizing new A pose variants to make the model more robust to pose changes in the input images. In reality, the users can stand in different A poses, which causes large changes in 3D output meshes. For example, the output mesh will look fatter if the users tilt left, right, lean backward, or forward.  

To solve this problem, I apply rigging techniques to create more pose variants per subject randomly. For each mesh, a skeleton of 16 joints is estimated, which is then used to calculate rigging weights for mesh vertices. After that, pose variants are constructed by rotating arm, leg, and spine bones with random angles. The two below figure shows the colorized silhouettes of different poses of the same human subject. Thanks to this approach, the model trained with pose variants become more accurate and invariant to pose.

However, the output meshes still fluctuate a bit. To achieve a more stable result, pose parameters need to be integrated into the model to make it more distinctive.

![](/assets/images/3dhm/2019-ca754c68.png)
![](/assets/images/3dhm/2019-d7c91d41.png)

### Camera parameters

Camera transformation also plays an essential role in forming the pictures. A picture taken with a camera at 1.6 meters over the ground will look different from a photo taken at 1.2 meters. Therefore, to truly reflect these variants in the data-set, I implemented a script to randomly change the camera angles/positions to make the model more robust to perspective changes in the image. The general idea of the technique is shown below. The camera on the right picture is tilted a bit upward than in the left picture.

![](/assets/images/3dhm/2019-2445f9a9.png)


## Challenges about training

In this part, I will explain to you challenges related training models which consist of the following topics: model architecture design, loss function design, and learning-rate tuning,

### Model architecture: combine three models front, side, and fusion.

In the beginning, I used a single model for both front and side pictures, as depicted in the below figure. This model takes in a 2-channel image representing front and side silhouettes and predicts PCA parameters. However, the 3D mesh outputs seem to just match with the contour of the front silhouette, but not of the side silhouette. This result is undesirable because I expect the 3D mesh output to match both profiles.

It took me a while to find out this problem. My theory is that the model is biased toward the front silhouette, which means it just learns very little information from the side silhouette. To test this theory, I wrote code to visualize the feature maps of the model, and it turns out that most of the feature maps have similar shapes like the front silhouette.

![](/assets/images/3dhm/2019-3992b938.png)

To tackle this problem, I separately trained the front and side model and then trained another fusion model to combine their outputs. In other words, the front and side model serves as two feature extraction/encoders for the fusion model. The diagram of the new architecture is depicted below.

![](/assets/images/3dhm/2019-59c17035.png)

### Two weight initialization techniques that boost accuracy

At first, I did not pay much attention to weight initialization. Fortunately, after reading [an article](http://karpathy.github.io/2019/04/25/recipe/) by Karpathy about model training recipes, I tried two weight initialization techniques.
- initialize the bias of the last layers to the corresponding mean value of the PCA values.
- initialize the weights of the last layers to the corresponding principal component vectors of the PCA model.

It turns out that these two techniques help the training converge faster, and the loss at the epoch before over-fitting is minimized 5 more percent.

### Design loss functions.

In the beginning, I just monitored the training through the mean square error between the predicted PCA and ground-truth PCA values, which give good enough results. Later, I tried three more approaches for the loss function, which turns out to be the most influential factors that contribute to accuracy improvement.

- __integrate the mesh loss__:  from the predicted PCA values, I reconstructed the corresponding 3D mesh during the training and then calculated the mean-square-error between the reconstructed mesh and the corresponding ground-truth mesh. The final loss is the average of PCA loss and mesh loss. This technique help forces the model to predict more accurate vertices instead of just PCA values.

$$
epoch0:   loss = 0.5*mse{\_}pca{\_}loss + 0.5*mse{\_}mesh{\_}loss \\
epoch10:  loss = 0.5*mse{\_}pca{\_}loss + 0.5*mse{\_}mesh{\_}loss \\
epoch50:  loss = 0.5*mse{\_}pca{\_}loss + 0.5*mse{\_}mesh{\_}loss \\
epoch100: loss = 0.5*mse{\_}pca{\_}loss + 0.5*mse{\_}mesh{\_}loss \\
$$  

- __dynamic weights for loss terms__: as discussed previously, the final loss is calculated as $ 0.5*PCAloss + 0.5*meshloss$.  However, it is quite intuitive that it is easier for the model to detect underlying trends in PCA values rather than in the 3D mesh vertices. This difficulty could be due to the fact that mesh representation is much more dense and complex than the PCA representation. Therefore, to make the training easier, the loss is designed to force the model to learn PCA values first and toward the end, focus more on mesh loss.

$$
epoch0:   loss = 0.9*mse{\_}pca{\_}loss + 0.1*mse{\_}mesh{\_}loss \\
epoch10:  loss = 0.7*mse{\_}pca{\_}loss + 0.3*mse{\_}mesh{\_}loss \\
epoch50:  loss = 0.5*mse{\_}pca{\_}loss + 0.5*mse{\_}mesh{\_}loss \\
epoch100: loss = 0.2*mse{\_}pca{\_}loss + 0.8*mse{\_}mesh{\_}loss \\
$$  

- __integrate silhouette re-projection loss__: the ultimate target of the training is finding a mesh that best explains the shape of the input silhouettes. Therefore, the best monitoring strategies is comparing the re-projection silhouettes of the predicted mesh with the input silhouettes. This is the most challenging loss term that I implemented in this project because it requires doing a rending in real-time with a correct projection matrix to get the re-projected silhouettes. The idea is outlined below with the loss equation

$$
  loss = 0.5 *pca{\_}mesh{\_}loss + 0.5*sil{\_}reproj{\_}iou{\_}loss
$$

![](/assets/images/3dhm/2019-db83accd.png)


### Use lower learning rates for earlier layers help improve accuracy.

As I mentioned in the section "Model Architecture," the front/side models are trained first, and then they are used as the encoder branches for the fusion model.

At first, training the fusion model also involves updating the weights of front/side branches, which are already trained before. However, the fusion model, again, is biased toward the front silhouette. Based on this observation, I tried to freeze the front/side branch weights, which help makes the training more stable.

However, this prevents front/side branches from fine-tuning their weights during the training of the fusion model. With this idea in mind, I took advantage of a Pytorch feature to assign a tiny lower learning rate to the front/side branches. This trick could still help them  learn new patterns but not so "out of the box." It is very wonderful to me that this technique helps increase the evaluation metrics by four more percent.

# Conclusion

In this note, I tried to wrap up the challenges I handled through my latest deep learning project. These challenges gave me lots of valuable lessons, which will be tremendously helpful for my future machine learning projects. I hope that it convinces you that I will be a valuable addition to the vision that CVEDIA is realizing.

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
