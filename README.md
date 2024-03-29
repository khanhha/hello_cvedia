Hi Rodrigo,

The vision of CVEDIA about synthetic data really inspires me. From my deep learning experience, data is the key to a high-quality model. Unfortunately, achieving a good data-set that could cover many edge cases is very hard and expensive in practice. Therefore, I firmly believe that synthetic data will play a crucial role in the future of deep learning. That is the reason why I want to be a part of CVEDIA's journey.

I know that every journey comes down to a good match. Therefore, to convince you that I will be a valuable addition to CVEDIA, I wrote an article to show you the deep learning challenges I tacked so far.

From my portfolio, I chose my latest project about images-based 3D human estimation because I believe that the skills I gained through this project will be very useful for projects in CVEDIA.   If you want to know more about other deep learning projects that I tackled such as crack detection, 3D textured head estimation, please feel free to let me know.

Due to the length of the article, I temporarily put it on my Github to make it easier to follow.

Best regards,
Khanh Ha


[](https://kamil-kielczewski.github.io/fractals/mandelbulb.html)

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

<p align="center"><img src="/assets/images/3dhm/2019-564ea8d3.png" align=middle height=300pt/></p>

# Challenges

In this part, I will describe the challenges that I tackled during this project. The experience that I gained through these challenges helps build my confidence in solving future machine learning problems.

To make it easier for you to follow, I classified them into two main types of challenges: the ones that relate to the data synthesis stage and the other ones about training techniques.

## Challenges about data synthesis

### Merge 3D human data-sets on the Internet.

In this challenge, I helped double the size of the training data from 3000 meshes to around 7000 meshes by merging different 3D human data-sets in the market. In contrast to images data-sets, combining different 3D human mesh data-sets are more complicated because of varying topology (different vertex number, triangle number, order).

To tackle this challenge, I implemented a registration algorithm that aligns/deforms the meshes to the same topology. The registration algorithm is implemented in a way that takes into account 72 3D human landmarks so that all the meshes are aligned correctly, as depicted by the red and blue segments in the below figure.

<p align="center"><img src="/assets/images/3dhm/2019-46ff3704.png" align=middle height=500pt/></p>

### New 3D human shapes synthesis based on the PCA model (Principal component analysis).

This challenge is about synthesizing more diverse human shapes, from skinny to fat, short to tall, etc. In the beginning, I just trained a deep learning model from 7000 meshes in the merged data-sets, which causes early over-fitting due to the lack of diverse data.

To resolve this problem, I first trained a PCA model that compresses human representation from the mesh format of 10.000 vertices to 50 principal components. The standard deviations along these principal components form a multivariate Gaussian distribution, which is then randomly sampled to create new PCA parameters representing new human shapes.

Thanks to this technique, I increase the size of training data from 7000 meshes to 100.000 meshes, which helps reduce the loss on the test-set by 20 percent.

### New 3D human pose variants synthesis.

This challenge is about synthesizing new A pose variants to make the model more robust to pose changes in the input images. In reality, the users can stand in different A poses, which causes large changes in 3D output meshes. For example, given the same user, the output mesh will look fatter if he tilts left or right, leans backward, or forward.  

To solve this problem, I apply rigging techniques to create more pose variants per subject randomly. For each mesh, a skeleton of 16 joints is estimated, which is then used to calculate rigging weights for mesh vertices. After that, pose variants are constructed by rotating arm, leg, and spine bones with random angles. The two below figure shows the colorized silhouettes of different poses of the same human subject. Thanks to this approach, the model trained with these pose variants become more accurate and invariant to pose.

However, the output meshes of different poses of the same user still fluctuate a bit. In the future, to achieve a more stable result, pose parameters need to be integrated into the model to make give the model more input information to ease the training.

<p align="center"><img src="/assets/images/3dhm/2019-ca754c68.png" align=middle height=300pt/></p>

<p align="center"><img src="/assets/images/3dhm/2019-d7c91d41.png" align=middle height=300pt/></p>


### Camera parameters

Camera transformation also plays an essential role in forming the pictures. A picture taken with a camera at 1.6 meters over the ground will look different from a photo taken at 1.2 meters. Therefore, to truly reflect these factors in the data-set, I implemented a script to randomly adjust the camera angles/positions to make the model more robust to perspective changes in the image. The general idea of the technique is shown below. The camera on the right picture is tilted a bit upward than in the left picture.

<p align="center"><img src="/assets/images/3dhm/2019-2445f9a9.png" align=middle height=150pt/></p>

## Challenges about training

In this part, I will explain to you challenges about training models which consist of the following topics: model architecture design, loss function design, and learning-rate tuning,

### Model architecture: combine three models front, side, and fusion.

In the beginning, I used a single model for both front and side pictures, as depicted in the below figure. This model takes in a 2-channel image representing front and side silhouettes and predicts PCA parameters. However, the 3D mesh outputs seem to match with just the contour of the front silhouette, not of the side silhouette. This result is undesirable because I expect the 3D mesh output to match both profiles.

It took me a while to find out this problem. My theory is that the model is biased toward the front silhouette, which means it just learns very little information from the side silhouette. To test this theory, I wrote code to visualize the feature maps of the model, and it turns out that most of the feature maps have similar shapes like the front silhouette.

<p align="center"><img src="/assets/images/3dhm/2019-3992b938.png" align=middle height=300pt/></p>


To tackle this problem, I separately trained the front and side model and then trained another fusion model to combine their outputs. In other words, the front and side model serves as two feature extraction/encoders for the fusion model. The diagram of the new architecture is depicted below.

<p align="center"><img src="/assets/images/3dhm/2019-59c17035.png" align=middle height=300pt/></p>

### Two weight initialization techniques that boost accuracy

At first, I did not pay much attention to weight initialization. Fortunately, after reading [an article](http://karpathy.github.io/2019/04/25/recipe/) by Karpathy about model training recipes, I tried two weight initialization techniques.
- initialize the bias of the last layers to the corresponding mean value of the PCA values.
- initialize the weights of the last layers to the corresponding principal component vectors of the PCA model.

It turns out that these two techniques help the training converge faster, and the loss at the epoch before over-fitting is minimized 5 more percent.

### Design loss functions.

In the beginning, I just monitored the training through the mean square error between the predicted PCA and ground-truth PCA values, which give good enough results. Later, I tried three more approaches for the loss function, which turns out to be the most influential factors that contribute to accuracy improvement.

- __integrate the mesh loss__:  from the predicted PCA values, I reconstructed the corresponding 3D mesh during the training and then calculated the mean-square-error between the reconstructed mesh and the corresponding ground-truth mesh. The final loss is the average of PCA loss and mesh loss. This technique help forces the model to predict more accurate vertices instead of just PCA values.

<p align="center"><img src="/tex/6d6e5a4a5f78a2bc661635ac44a76f19.svg?invert_in_darkmode&sanitize=true" align=middle width=409.87947495pt height=14.611878599999999pt/></p>
<p align="center"><img src="/tex/ca8e84741b22b89eaed5332cdab924a1.svg?invert_in_darkmode&sanitize=true" align=middle width=418.0986843pt height=14.611878599999999pt/></p>
<p align="center"><img src="/tex/2cec76c84c12e1665daf9a4780f1ed4c.svg?invert_in_darkmode&sanitize=true" align=middle width=418.0986843pt height=14.611878599999999pt/></p>
<p align="center"><img src="/tex/71b29723b5e712169228e4e0b067372b.svg?invert_in_darkmode&sanitize=true" align=middle width=426.31789365pt height=14.611878599999999pt/></p>  

- __dynamic weights for loss terms__: as discussed previously, the final loss is calculated as <img src="/tex/693868ee43d1d8ebd459da6806d48b5b.svg?invert_in_darkmode&sanitize=true" align=middle width=227.7191598pt height=22.831056599999986pt/>.  However, it is quite intuitive that it is easier for the model to detect underlying trends in PCA values rather than in the 3D mesh vertices. This difficulty could be due to the fact that mesh representation is much more dense and complex than the PCA representation. Therefore, to make the training easier, the loss is designed to force the model to learn PCA values first and toward the end, focus more on mesh loss.

<p align="center"><img src="/tex/cc4fe8470958ad5b2311cdcbf6ca6d19.svg?invert_in_darkmode&sanitize=true" align=middle width=409.87947495pt height=14.611878599999999pt/></p>
<p align="center"><img src="/tex/c49fd72a8d98920709df0369f181eb4e.svg?invert_in_darkmode&sanitize=true" align=middle width=418.0986843pt height=14.611878599999999pt/></p>
<p align="center"><img src="/tex/2cec76c84c12e1665daf9a4780f1ed4c.svg?invert_in_darkmode&sanitize=true" align=middle width=418.0986843pt height=14.611878599999999pt/></p>
<p align="center"><img src="/tex/e43eea1099d6c14c739aaa5902fded42.svg?invert_in_darkmode&sanitize=true" align=middle width=426.31789365pt height=14.611878599999999pt/></p>


- __integrate silhouette re-projection loss__: the ultimate target of the training is finding a mesh that best explains the shape of the input silhouettes. Therefore, the best monitoring strategies is comparing the re-projection silhouettes of the predicted mesh with the input silhouettes. This is the most challenging loss term that I implemented in this project because it requires doing a rending in real-time with a correct projection matrix to get the re-projected silhouettes. The idea is outlined below with the loss equation

<p align="center"><img src="/tex/9b0a2e73fa23bca3fab4ed8a3d02913b.svg?invert_in_darkmode&sanitize=true" align=middle width=382.8042372pt height=14.611878599999999pt/></p>

<p align="center"><img src="/assets/images/3dhm/2019-db83accd.png" align=middle height=300pt/></p>


### Use lower learning rates for earlier layers help improve accuracy.

As I mentioned in the section "Model Architecture," the front/side models are trained first, and then they are used as the encoder branches for the fusion model.

At first, training the fusion model also involves updating the weights of front/side branches, which are already trained before. However, the fusion model, again, is biased toward the front silhouette. Based on this observation, I tried to freeze the front/side branch weights during training the fusion model. This helps make the training more stable.

However, this prevents front/side branches from fine-tuning their weights during the training of the fusion model. With this idea in mind, I took advantage of a Pytorch feature to assign a tiny lower learning rate to the front/side branches. This trick still give front/side branches a chance to adjust their learned weights, but not in a too "out of the box" way. It is very wonderful to me that this technique helps increase the evaluation metrics by four more percent.

# Conclusion

In this note, I tried to wrap up the challenges I handled through my latest deep learning project. These challenges gave me lots of valuable lessons, which will be tremendously helpful for my future machine learning projects. I hope that it convinces you that I will be a valuable addition to the vision that CVEDIA is realizing.
