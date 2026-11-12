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

---

## 2. The loss: triplet loss with a margin

### 2.1 Principle and objective

Let `f` be a neural network (a pretrained convolutional backbone with a small fully connected head) that maps a crop `x` to an embedding vector `f(x)` of 128 numbers. We L2-normalise it, so every embedding has length 1 and lies on the unit sphere. The distance between two crops is the Euclidean distance of their embeddings, `d(x, y) = || f(x) - f(y) ||`.

The triplet loss works on three crops at once, with the same network `f` (shared weights) applied to each:

- the **anchor** `a`, any crop;
- the **positive** `p`, another crop of the same class as `a`;
- the **negative** `n`, a crop of a different class.

For one triplet the loss is

$$\mathcal{L}(a, p, n) = \max\bigl(0,\; d(a, p) - d(a, n) + m\bigr),$$

where `m > 0` is the **margin**. The training objective is to make, for every triplet, the negative farther from the anchor than the positive by at least the margin: `d(a, n) >= d(a, p) + m`. A triplet that already satisfies this contributes loss 0; a triplet that does not is pulled into order, because the gradient moves `a` towards `p` and away from `n`. The loss for a batch is the average over the triplets in it (in practice often over the non-zero ones only). After training, crops of one class form a compact cluster and clusters of different classes are separated by roughly the margin.

### 2.2 Three kinds of triplet

For a fixed margin, a triplet falls into one of three cases, depending on where the negative lies relative to the positive:

| Case | Condition | Loss | What it teaches |
| --- | --- | --- | --- |
| Easy | `d(a, n) >= d(a, p) + m` | 0 | Nothing: already correct. |
| Semi-hard | `d(a, p) < d(a, n) < d(a, p) + m` | between 0 and `m` | Negative is farther than the positive but inside the margin. Correct order, not enough room. |
| Hard | `d(a, n) < d(a, p)` | greater than `m` | Negative is closer than the positive: the embedding is wrong here. |

Worked example with `d(a, p) = 1` and `m = 0.5`: a negative at distance 2 is easy (loss 0); at 1.2 it is semi-hard (loss `1 - 1.2 + 0.5 = 0.3`); at 0.8 it is hard (loss `1 - 0.8 + 0.5 = 0.7`). With `d(a, p) = 1` and `d(a, n) = 1.5` the loss is `max(0, m - 0.5)`: it is 0 for any `m <= 0.5`, 0.5 for `m = 1` and 1.5 for `m = 2`.

This is why the sampling strategy decides whether anything is learned: if most sampled triplets are easy, the loss is 0 and the gradient vanishes; if they are all hard, training can collapse (section 5).

### 2.3 The margin

Because the embeddings lie on the unit sphere, distances lie between 0 and 2, and a margin must be achievable. Six classes whose centres are as far apart as possible (the corners of a regular simplex on the sphere) are at distance `sqrt(2 * 6 / 5) = 1.55`. The margin therefore has to be well below this, and also leave room for the spread inside each class. We start with `m = 0.2` and tune it on the validation split; published tuned runs of the triplet loss also found small margins (about 0.1) best. The margin controls the trade-off: too small and almost every triplet is already satisfied (no gradient, clusters stay close); too large and nearly all triplets stay active, so the loss never reaches zero and training pushes harder than the geometry allows.

### 2.4 Why the triplet loss for this dataset

1. **It uses relative distances.** The lecture notes that metric learning is robust to class imbalance because it learns relative distances instead of absolute class distributions. The loss never asks "how many examples has this class", only "is the positive closer than the negative".
2. **Sampling controls the imbalance.** With 61% of all crops in one class, the imbalance cannot be removed by the loss alone; the triplet loss lets us fix it where it arises, in how anchors, positives and negatives are drawn (section 4).
3. **It is transparent.** The margin and the three triplet cases can be explained and checked by hand, which matches the lecture (deck 07, pages 14 and 15: triplet loss, with the note that the sampling strategy is important).
4. **The alternatives were considered.** The contrastive loss uses pairs and the same margin idea; Proxy-NCA and Proxy-Anchor replace sampled negatives with one learned vector per class and would also suit six classes; ArcFace is a classification loss with an angular margin. Section 6 explains why we keep the triplet loss and what we would change if we did not.
