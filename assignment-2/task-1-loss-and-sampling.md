# Assignment 2, Task 1: Loss function and sampling strategy

Deep metric learning for automated waste sorting (theoretical solution; nothing is implemented or trained).

> Work in progress. Sections 2 to 6 follow.

**Task.** Choose a deep metric learning loss for learning an embedding of the waste categories. Describe (a) the principle and objective of the loss, (b) how training samples must be selected and combined for it, and (c) a sampling strategy for the given dataset. Discuss how the sampling strategy influences training and the resulting embedding space, and justify the decisions with the dataset's characteristics.

**Answer in one paragraph.** We use the triplet loss with a margin on L2-normalised embeddings of object crops. Triplets are formed online inside class-balanced batches of 6 classes x K crops. Batches are built by choosing the source image first, then at most one or two crops of a class from it, and never two crops from the same duplicate group. Crops shorter than 24 pixels are excluded, and no image contributes more than 10 crops per epoch. After a warm-up, negatives are chosen semi-hard rather than hardest. Each of these choices answers a measured property of the data: a 61% majority class, crowded images, tiny objects, duplicated photos and confusable materials.

---

## 1. Setting and dataset characteristics

### 1.1 What the goal is

Waste of the same category can look very different (a crushed clear bottle and a green one are both plastic), and waste of different categories can look alike (plastic and glass bottles, paper and cardboard). A deep metric learning model learns an embedding function `f` that maps an image to a vector so that objects of the same category end up close together and objects of different categories end up far apart. We take the six classes of the Garbage Detection Dataset: biodegradable, cardboard, glass, metal, paper and plastic. (The class names are inferred from the file names and the class ids 0 to 5 in alphabetical order; no official class file was supplied.)

### 1.2 What is embedded: object crops, not images

The dataset is an object detection dataset: 10,464 images of 416 x 416 pixels with 74,073 labelled boxes (after removing 17 zero-area rows). We embed **object crops cut from the labelled boxes**, because a whole image has no single category:

- Nearly half of the images (5,082) hold one object, but the median image holds 2 and the largest holds 358.
- 10.2% of the images (1,066) contain more than one class. Glass appears together with metal in 531 images and with plastic in 226 images; no single label describes such an image.
- A crop has exactly one class, which is what the loss needs.

### 1.3 Dataset properties that drive the design

All numbers were computed on the cleaned labels (17 zero-area rows removed) and, where a split is mentioned, on the team's fixed group-aware split (train 7,325 / validation 1,570 / test 1,569 images).

| Property | Measured fact | Consequence for the loss and sampling |
| --- | --- | --- |
| Box-level imbalance | Class 0 (biodegradable) has 45,395 of 74,073 crops (61.3%). The others have 4,387 to 7,809 (5.9% to 10.5%). In the training split the largest class has 10.3 times as many crops as the smallest. | Plain random batches would be mostly biodegradable. Batches must be class-balanced. |
| Image-level imbalance | Much milder: the number of training images that contain a class ranges from 1,024 (plastic) to 1,885 (glass). | Every class has enough distinct source images to build balanced batches without repeating a few photos. |
| Crowded images | The top 1% of images (104) hold 21.2% of all crops, the top 5% hold 50.9%. These crowded images are mostly biodegradable (86% to 92% of their crops). | One photo must not supply many crops of a class to a batch or an epoch. Cap crops per image. |
| Tiny objects | 36.45% of the boxes cover less than 1% of the image. The median crop is 45 pixels on its short side, and 11.4%, 24.1% and 35.3% of the crops are shorter than 16, 24 and 32 pixels. Biodegradable crops are the smallest (median 34 pixels). | Very small crops carry almost no visual signal and would act as label noise. Set a minimum crop size. |
| Duplicated photos | The images look like a Roboflow export with flipped, rotated and re-cropped copies. Our own search found about 300 duplicate groups covering about 650 images (about 6%). | Two crops from the same source photo must never be treated as unrelated, and must not meet in one batch as a "negative". |
| Confusable materials | Glass, metal and plastic share shapes (bottles, cans); paper and cardboard overlap; glass-named images contain about 14% metal crops. | These pairs produce the hard negatives. They need care in mining. |

Sections 2 to 6 use these facts to choose the loss (section 2), explain how samples are combined (section 3), design the sampling strategy (section 4), discuss its effect on training and on the embedding space (section 5), and state limits and alternatives (section 6).
