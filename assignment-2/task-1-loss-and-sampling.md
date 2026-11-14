# Assignment 2, Task 1: Loss function and sampling strategy

Deep metric learning for automated waste sorting (theoretical solution; nothing is implemented or trained).

Contents: 1 Setting and dataset characteristics; 2 The loss; 3 How samples are selected and combined; 4 Sampling strategy for this dataset; 5 Influence of the sampling on training and on the embedding space; 6 Limits, alternatives and references.

**Task.** Choose a deep metric learning loss for learning an embedding of the waste categories. Describe (a) the principle and objective of the loss, (b) how training samples must be selected and combined for it, and (c) a sampling strategy for the given dataset. Discuss how the sampling strategy influences training and the resulting embedding space, and justify the decisions with the dataset's characteristics.

**Answer in one paragraph.** We use the triplet loss with a margin on L2-normalised embeddings of object crops. Triplets are formed online inside class-balanced batches of 6 classes x K crops. Batches are built by choosing the source image first and then one crop of the class from it, so that the K crops of a class come from K different photos and K different duplicate groups. Crops shorter than 24 pixels are excluded, and because images rather than crops are drawn, a crowded photo is not drawn more often than any other. After a warm-up, negatives are chosen semi-hard rather than hardest. Each of these choices answers a measured property of the data: a 61% majority class, crowded images, tiny objects, duplicated photos and confusable materials.

---

## 1. Setting and dataset characteristics

### 1.1 What the goal is

Waste of the same category can look very different (a crushed clear bottle and a green one are both plastic), and waste of different categories can look alike (plastic and glass bottles, paper and cardboard). A deep metric learning model learns an embedding function `f` that maps an image to a vector so that objects of the same category end up close together and objects of different categories end up far apart. We take the six classes of the Garbage Detection Dataset: biodegradable, cardboard, glass, metal, paper and plastic. (The class names are inferred from the file names and the class ids 0 to 5 in alphabetical order; no official class file was supplied.)

### 1.2 What is embedded: object crops, not images

The dataset is an object detection dataset: 10,464 images of 416 x 416 pixels with 74,073 labelled boxes (after removing 17 zero-area rows). We embed **object crops cut from the labelled boxes**, because a whole image has no single category:

- Nearly half of the images (5,082) hold one object, but the median image holds 2 and the largest holds 358.
- 10.2% of the images (1,066) contain more than one class. Among the images with exactly two classes, the most common pairs are glass with metal (531 images) and glass with plastic (226 images); no single label describes such an image.
- A crop has exactly one class, which is what the loss needs.

### 1.3 Dataset properties that drive the design

All numbers were computed on the cleaned labels (17 zero-area rows removed) and, where a split is mentioned, on the team's fixed split (train 7,325 / validation 1,570 / test 1,569 images). That split keeps together the duplicate groups confirmed in the team's own review; it does not yet include all groups from our wider duplicate search (section 4.4).

| Property | Measured fact | Consequence for the loss and sampling |
| --- | --- | --- |
| Box-level imbalance | Class 0 (biodegradable) has 45,395 of 74,073 crops (61.3%). The others have 4,387 to 7,809 (5.9% to 10.5%). In the training split the largest class has 10.3 times as many crops as the smallest. | Plain random batches would be mostly biodegradable. Batches must be class-balanced. |
| Image-level imbalance | Much milder: the number of training images that contain a class ranges from 1,024 (plastic) to 1,885 (glass). | Every class has enough distinct source images to build balanced batches without repeating a few photos. |
| Crowded images | The top 1% of images (104) hold 21.2% of all crops, the top 5% hold 50.9%. These crowded images are mostly biodegradable (86% to 92% of their crops). | One photo must not supply many crops of a class to a batch or an epoch. Draw images, not crops. |
| Tiny objects | 36.45% of the boxes cover less than 1% of the image. The median crop is 45 pixels on its short side, and 11.4%, 24.1% and 35.3% of the crops are shorter than 16, 24 and 32 pixels. Biodegradable crops are the smallest (median 34 pixels). | Very small crops carry almost no visual signal and would act as label noise. Set a minimum crop size. |
| Duplicated photos | The images look like a Roboflow export with flipped, rotated and re-cropped copies. Our own search found about 300 duplicate groups covering about 660 images (about 6%). | Two copies of one photo contain the same objects, so they must not be treated as unrelated samples: as a positive a copy is trivial, and as a negative it would be a labelling error. |
| Confusable materials | Glass, metal and plastic share shapes (bottles, cans) and often appear together (about 14% of the crops in glass-named images are metal); paper and cardboard look alike and the boundary between their labels is fuzzy. | These pairs produce the hard negatives. They need care in mining. |

Sections 2 to 6 use these facts to choose the loss (section 2), explain how samples are combined (section 3), design the sampling strategy (section 4), discuss its effect on training and on the embedding space (section 5), and state limits and alternatives (section 6).

---

## 2. The loss: triplet loss with a margin

### 2.1 Principle and objective

Following the lecture, a deep metric learning system has three parts: the input samples, the network model and the loss function (deck 07, pages 7 and 8). The input samples are object crops (section 1.2) and are chosen by the sampling strategy (sections 3 and 4). The network `f` (the lecture's `G_W`) is a convolutional backbone pretrained on a large image dataset, with a small fully connected head; with only about 1,000 to 1,900 training images per class, a pretrained network needs less data and converges faster than one trained from scratch (deck 07, page 12). It maps a crop `x` to an embedding vector `f(x)` of 128 numbers. We L2-normalise it, so every embedding has length 1 and lies on the unit sphere. The distance between two crops is the Euclidean distance of their embeddings, `d(x, y) = || f(x) - f(y) ||`.

The triplet loss works on three crops at once, with the same network `f` (shared weights) applied to each:

- the **anchor** `a`, any crop;
- the **positive** `p`, another crop of the same class as `a`;
- the **negative** `n`, a crop of a different class.

For one triplet the loss is

$$\mathcal{L}(a, p, n) = \max\bigl(0,\; d(a, p) - d(a, n) + m\bigr),$$

where `m > 0` is the **margin** (the lecture's `alpha`; deck 07, pages 14 and 15). The training objective is to make, for every triplet, the negative farther from the anchor than the positive by at least the margin: `d(a, n) >= d(a, p) + m`. A triplet that already satisfies this contributes loss 0; a triplet that does not is pulled into order, because the gradient moves `a` towards `p` and away from `n`. The loss for a batch is the average over the triplets in it (in practice often over the non-zero ones only; Hermans et al., 2017). After training, crops of one class form a compact cluster and clusters of different classes are separated by roughly the margin.

### 2.2 Three kinds of triplet

For a fixed margin, a triplet falls into one of three cases, depending on where the negative lies relative to the positive:

| Case | Condition | Loss | What it teaches |
| --- | --- | --- | --- |
| Easy | `d(a, n) >= d(a, p) + m` | 0 | Nothing: already correct. |
| Semi-hard | `d(a, p) < d(a, n) < d(a, p) + m` | between 0 and `m` | Negative is farther than the positive but inside the margin. Correct order, not enough room. |
| Hard | `d(a, n) <= d(a, p)` | at least `m` | Negative is as close as or closer than the positive: the embedding is wrong here. |

Worked example with `d(a, p) = 1` and `m = 0.5`: a negative at distance 2 is easy (loss 0); at 1.2 it is semi-hard (loss `1 - 1.2 + 0.5 = 0.3`); at 0.8 it is hard (loss `1 - 0.8 + 0.5 = 0.7`). With `d(a, p) = 1` and `d(a, n) = 1.5` the loss is `max(0, m - 0.5)`: it is 0 for any `m <= 0.5`, 0.1 for `m = 0.6` and 0.3 for `m = 0.8`. The same triplet is easy or not depending only on the margin.

This is why the sampling strategy decides whether anything is learned: if most sampled triplets are easy, the loss is 0 and the gradient vanishes; if they are all hard, training can collapse (section 5).

### 2.3 The margin

Because the embeddings lie on the unit sphere, distances lie between 0 and 2, and a margin must be achievable. Six classes whose centres are as far apart as possible (the corners of a regular simplex on the sphere) are at distance `sqrt(2 * 6 / 5) = 1.55`: for unit vectors `d^2 = 2 - 2 cos(theta)`, and the six corners of a regular simplex have `cos(theta) = -1/5` between each pair, so `d^2 = 2 + 2/5 = 12/5`. The margin therefore has to be well below this, and also leave room for the spread inside each class. We start with `m = 0.2` and tune it on the validation split; Musgrave et al. (2020) remark that the best triplet margin tends to be around 0.1. (FaceNet also uses 0.2, but on squared distances, so the two values are not directly comparable.) The margin controls the trade-off: too small and almost every triplet is already satisfied (no gradient, clusters stay close); too large and nearly all triplets stay active, so the loss never reaches zero and training pushes harder than the geometry allows.

### 2.4 Why the triplet loss for this dataset

1. **It uses relative distances.** The lecture (deck 07, page 4) notes that metric learning is robust to class imbalance because it learns relative distances instead of absolute class distributions. The loss never asks "how many examples has this class", only "is the positive closer than the negative".
2. **Sampling controls the imbalance.** With 61% of all crops in one class, the imbalance cannot be removed by the loss alone; the triplet loss lets us fix it where it arises, in how anchors, positives and negatives are drawn (section 4).
3. **It is transparent.** The margin and the three triplet cases can be explained and checked by hand, which matches the lecture (deck 07, pages 14 and 15: triplet loss, with the note that the sampling strategy is important).
4. **It fits the structure of these classes better than one proxy per class.** The lecture's main alternative, Proxy-NCA, pulls every crop of a class towards one learned class vector, and the lecture notes that it converges faster and costs less than the triplet loss (deck 07, page 18). Two properties of this dataset weaken that advantage. First, the classes are visually multi-modal: biodegradable holds both fruit and vegetables, plastic holds bottles, bags and containers. A single proxy pushes such a class towards one point, while the triplet loss only asks that each crop be closer to some crops of its class than to other classes, so sub-clusters may remain (section 5.3). Second, the cost argument matters most when a batch cannot contain every class; with only six classes, every P x K batch contains all of them, and one batch of 48 crops already yields 13,440 triplets from a single distance matrix (section 3.3). The lecture's own comparison shows neither loss winning on every benchmark (deck 07, page 19). The triplet loss also compares crops with crops directly, which is the relation that retrieval in Task 2 uses. Section 6.2 discusses the alternatives and when we would switch.

---

## 3. How training samples are selected and combined

### 3.1 Too many triplets to list

The training split holds 51,861 crops. Every ordered choice of an anchor, a positive of the same class and a negative of another class is a valid triplet, which gives about 2.4 x 10^13 triplets. We cannot use all of them, and most would be useless: the loss of an easy triplet is 0. Worse, if triplets were drawn uniformly, 83.5% of them would have a biodegradable anchor, because that class owns 61% of the crops and its crops appear in many more combinations. The selection rule is therefore part of the method, not a detail.

### 3.2 Online instead of offline triplet formation

There are two ways to form triplets:

- **Offline:** embed the whole training set with the current network, list the triplets, then train on that list. The list is out of date after the first few updates, because the embedding has already changed, and embedding 51,861 crops again for each list is expensive.
- **Online:** draw a batch of crops, embed it with the current network, and form triplets inside the batch from those fresh embeddings. Which triplets are easy, semi-hard or hard is always judged with the network as it is now.

We use online formation, since it is cheap and always current.

### 3.3 The batch: P classes x K crops

Triplets can only be formed inside a batch if the batch contains both positives and negatives for each anchor. A random batch of 48 crops from this dataset would contain about 29 biodegradable crops but only about 3 paper crops, and in roughly one batch in five at most one paper crop, which leaves that anchor without a positive. We therefore build each batch as **P x K**: P classes with K crops of each, the batch design popularised by Hermans et al. (2017). Since there are only six classes, P = 6, so every batch contains all classes, and every anchor has K - 1 positives and (P - 1) x K negatives.

| Setting | Batch size | Triplets formed from one batch |
| --- | --- | --- |
| P = 6, K = 8 | 48 | 48 x 7 x 40 = 13,440 |
| P = 6, K = 16 | 96 | 96 x 15 x 80 = 115,200 |

One forward pass of 48 or 96 crops produces thousands of triplets, because the pairwise distances inside the batch are computed once and reused for every triplet. We choose K between 8 and 16: larger K gives more triplets per step, but each class then needs more distinct source photos in the batch (section 4).

### 3.4 Which triplets inside the batch are used

With the 48 x 48 distance matrix of the batch we can mark, for every anchor `a`, its positives (same class, a different crop) and its negatives (other classes), and then choose from these. The lecture separates random sampling from informative sampling (deck 07, pages 8 to 10); the options are:

| Strategy | Rule for the negative | Property |
| --- | --- | --- |
| All valid triplets (closest to the lecture's random sampling) | Every positive with every negative | Many easy triplets; the average loss is diluted by zeros |
| Hardest negative (lecture: hard negative mining) | The negative closest to the anchor | Strongest signal and prevents the model from learning only trivial patterns, but with noisy labels or tiny crops the closest negative is often one of them, and early in training it can collapse the embedding (Schroff et al., 2015) |
| Semi-hard | `d(a, p) < d(a, n) < d(a, p) + m` (Schroff et al., 2015) | Negative is farther than the positive but still inside the margin: informative and more stable. Such negatives become rare late in training (Wu et al., 2017), hence the fallback below |
| Distance-weighted (Wu et al., 2017) | Sample negatives with weights inversely proportional to how common their distance is on the sphere | Negatives spread over all distances instead of being dominated by the easy ones that random sampling gives; the weights are clipped so that the closest, noisiest negatives are not over-sampled; optional |

**A note on the definition of semi-hard.** The lecture describes semi-hard negatives as "closer to the anchor than a positive example but not the hardest ones" (deck 07, page 9). Read literally, that would include negatives that are closer than the positive, which our table calls hard. We use the precise definition from FaceNet (Schroff et al., 2015): the negative is farther than the positive but still inside the margin, so the order is correct and only the gap is missing. FaceNet states this with squared distances; we apply the same inequality to the unsquared distances of the lecture's triplet formula.

Hermans et al. (2017) found that the hardest negative of each anchor in a P x K batch ("batch hard") works well for person re-identification. In this dataset the closest negative is often a mislabelled, tiny or truncated crop (section 4.5), so we use semi-hard negatives by default and treat batch-hard mining as an alternative to compare on the validation split.

We use all valid triplets for a short warm-up (for example the first epoch), so that the newly initialised embedding head settles on top of the pretrained backbone before harder triplets are selected, and then **semi-hard negatives**: for every anchor-positive pair we choose a negative that satisfies the inequality above (picking one at random among the candidates, which adds variety). The batch loss is the average over the triplets with non-zero loss, so that the many zero-loss triplets do not shrink the gradient. If an anchor-positive pair has no semi-hard negative, we fall back to the least hard negative among the harder ones (the farthest negative that is still closer than the positive), and we skip the pair when all its negatives are already beyond the margin. This fallback is our own design choice.

### 3.5 Rules for a valid triplet in this dataset

A triplet is only valid for training when all of the following hold. The positive part of rule 1 and rule 3 hold by how the batch is built (section 4); the negative part of rule 1 is enforced by a mask on the batch distance matrix (section 3.6, step 4); rule 2 holds by the definition of a negative.

1. The anchor and the positive come from different source photos and from different duplicate groups. Otherwise the positive would be a copy of the anchor (or another crop of its photo with the same background), and the loss would be reduced by matching photos rather than materials.
   The negative must never be a copy of the anchor's own object. In practice, crops from the *other* images of the anchor's duplicate group are not used as its negatives, because those images show the same objects again. Crops of *other objects in the anchor's own image* are allowed as negatives: a glass and a metal object on one table share a background, so only the material separates them, which is a useful hard negative. The difference is that in a copy of the photo we cannot tell which crop shows the anchor's own object again (the copy may be flipped, rotated or cropped), whereas in the anchor's own image every other box is, by construction, a different object.
2. The class of `n` differs from the class of `a`, using the cleaned labels.
3. Each crop is at least 24 pixels on its short side. Smaller crops are excluded from the batches (section 4).

### 3.6 One training step

1. Draw a P x K batch with the rules of section 4.
2. Crop, pad, resize and augment each member; pass the batch through the network and L2-normalise the 48 or 96 embeddings.
3. Compute all pairwise distances in the batch.
4. Mask out invalid pairs (same-class pairs from one photo or one duplicate group, and negatives from the other images of the anchor's duplicate group), then for each anchor-positive pair choose a negative (all triplets during warm-up, semi-hard afterwards).
5. Average `max(0, d(a, p) - d(a, n) + m)` over the chosen triplets with non-zero loss.
6. Update the network weights with the gradient of that average; repeat with a new batch.

---

## 4. Sampling strategy for this dataset

The strategy has three layers: which crops may be used at all (4.1), how a batch is drawn from them (4.2 to 4.4), and which triplets inside the batch are used (3.4 and 4.5). Every rule answers one of the measured properties in section 1.3. Unless stated otherwise, counts below are for the training split (7,325 images, 51,861 crops); the validation and test splits are never sampled or capped for training.

### 4.1 Which crops are eligible: minimum size and context

- **Minimum size: 24 pixels on the short side.** Crops below about 12 pixels are colour blobs with no recognisable shape, and between 12 and 24 pixels the signal is weak; such a crop behaves like a wrong label. A threshold of 24 pixels keeps 39,574 of the 51,861 training crops (76.3%). Biodegradable loses the most (31%), because its objects are the smallest; the other classes lose 6% to 18%. We treat 24 as a setting to tune on validation, with 16 and 32 pixels as the alternatives.
- **Context padding and resizing.** The boxes are tight. We enlarge each box by 10% to 20% so that shape and surface cues at the edge are kept, pad to a square, and resize to one fixed size (for example 128 x 128). The padding is limited because boxes are often nested or touching (over all images, a third of the boxes lie more than half inside another box), so a wide margin would bring pieces of neighbouring objects into the crop.

### 4.2 Draw the image first, then the crop

If a class's crops were drawn uniformly, they would often come from the same few photos. A simulation over all training crops shows that 16 plastic crops drawn at random come from 16 different photos only about half the time, and 16 cardboard crops in about two thirds of the draws; with K = 8 the figures are about 84% and 91%. Restricted to crops of at least 24 pixels the figures are somewhat better (about 58% and 72% for K = 16) but the problem remains. This is a form of the lecture's diversity sampling (deck 07, page 9): the crops of one class in a batch should be as varied as possible. Crops from one photo share lighting, background and camera, so they are too easy as positives, they teach the network to match backgrounds, and in crowded piles they are even overlapping fragments of one object.

For each class in a batch we therefore proceed in this order:

1. choose K images that contain an eligible crop of that class, uniformly at random and from K different duplicate groups;
2. from each chosen image take one eligible crop of that class.

This is always possible with K = 8 or 16: every class has eligible crops (at least 24 pixels) in at least 999 training images (plastic has the fewest, glass the most with 1,883). Choosing the image uniformly instead of the crop also removes the weight of crowded photos: counted on the same eligible crops, the number of images per class differs by a factor of about 1.9 (999 to 1,883), while the number of crops per class differs by a factor of 7.8 (2,797 paper to 21,777 biodegradable crops). Within a class, every eligible image is equally likely to be chosen.

The images for the different classes of a batch are chosen independently, so a negative may come from the same photo as an anchor of another class. This is allowed (section 3.5).

### 4.3 Class-balanced batches, epoch length and an optional cap

- **Every batch contains all six classes with K crops each** (P x K, section 3.3), so no batch is dominated by biodegradable.
- **An epoch is a fixed number of iterations**, not a pass over all crops, because the batches are balanced and not proportional to the data. We set it to about 500 iterations of 48 crops (about 24,000 crops, roughly the size of the eligible crop pool after an optional cap of 10 crops per image).
- **How often an image is seen.** With image-first drawing (4.2), each class receives 8 image draws per batch, about 3,950 per epoch. Each eligible image is therefore drawn about 2.1 times per epoch (glass, the class with the most images) to 4.0 times (plastic, the fewest), a ratio of about 1.9. A crowded biodegradable photo is drawn about 2.5 times per epoch, like any other biodegradable photo, and contributes one crop each time. This, not the number of crops per image, is what the network actually sees.
- **An optional cap of 10 crops per image per epoch.** Under image-first drawing the cap almost never binds, because no image is drawn more than a few times per epoch. It only matters if crops instead of images were drawn: then the size filter and the cap would shrink the crop pool to about 23,700 crops and the largest-to-smallest class ratio from 10.3 to 3.7 (biodegradable from 31,772 to about 9,100 crops; the other classes about 2,500 to 4,050). We keep it as an option to test on validation, not as part of the core strategy.

For comparison, if crops were drawn uniformly within each class with equal class shares, each biodegradable crop would be seen 0.27 times per epoch and each paper crop 2.8 times, a ten-fold difference. With image-first drawing, rare-class images are still repeated somewhat more often than biodegradable ones, so augmentation (flips, scale changes, colour jitter) is applied to the training crops to keep the network from memorising them.

### 4.4 Duplicate photos

The images look like a Roboflow export containing flipped, rotated and re-cropped copies of the same photo; our own search found about 300 duplicate groups (about 6% of the images). For the training samples this means:

- two crops from one duplicate group are never chosen for the same class in a batch (4.2), so a copy cannot act as a free positive, and crops from the other images of an anchor's duplicate group are masked out as its negatives (3.5), so a copy of the same object can never be pushed away;
- the split into training, validation and test must be made by duplicate group, so that no copy of a validation or test photo is trained on. The team's current split does this only for the 267 groups confirmed in its own review of 364 candidate pairs. Our wider search found 307 groups (664 images), and 44 of them still cross the split; 23 of these contain a test image whose copy is in training or validation. A visual check judged about 19 of the 23 to be real copies (about 1.2% of the 1,569 test images). Before the embedding is trained or evaluated, the split therefore has to be rebuilt from the union of both duplicate lists, or those test images removed. Duplicates that neither search found can remain and would make validation and test scores slightly optimistic.

### 4.5 Which negatives: warm-up, then semi-hard, with care for the confusable pairs

Section 3.4 gives the mining rule: all valid triplets for a short warm-up, then semi-hard negatives. The dataset adds three reasons to avoid the hardest negatives:

- **Fuzzy class boundaries.** Paper and cardboard look alike (cups, cartons, packaging), and in our visual inspection of sample crops their labels were not always consistent. A hardest-negative rule would single out exactly these borderline cases and ask the network to separate things that the labels themselves do not separate cleanly. The lecture names this as the reason for semi-hard sampling: it "avoids extremely hard (potentially mislabeled) negatives" (deck 07, page 9).
- **Tiny and truncated crops.** Even above 24 pixels, a crop may show only part of an object; in the training split, 50% of the paper boxes and 33% of the cardboard boxes touch the image border (within 2 pixels).
- **Nested boxes.** Over all images, 52% of the plastic boxes lie more than half inside another box, so a "negative" may contain a piece of the anchor's own object.

The pairs we expect to produce most hard negatives, and therefore monitor, are glass and metal, glass and plastic, and paper and cardboard. If they remain mixed late in training, the batches can be changed for the last epochs so that these classes are drawn together with a larger K (a curriculum from easy to hard, close to the lecture's adaptive sampling; deck 07, page 10); this is an option, not a requirement of the method. The lecture's uncertainty sampling (deck 07, page 10), which picks the samples the model is least sure about, is not used: with labels for all crops, the semi-hard rule already selects the informative triplets.

### 4.6 Summary of the strategy

| Component | Choice | Dataset reason | To be tuned on validation |
| --- | --- | --- | --- |
| Unit | Object crop from the cleaned boxes | Images hold several objects and classes | no |
| Minimum size | 24 px on the short side | 24% of crops are below 24 px; tiny crops carry no signal | 16, 32 px |
| Context | 10% to 20% padding, square, fixed size | Tight boxes, nested neighbours | padding, size |
| Batch | P = 6 classes x K = 8 (or 16) crops | 61% majority class; every class needs positives | K |
| Sampling unit | Image first, one crop per image per class in a batch | Crowded images; 16 random plastic crops come from 16 different photos only about half the time | no |
| Epoch length | About 500 iterations of 48 crops; each image drawn about 2 to 4 times per epoch | Balanced batches are not proportional to the data | length |
| Crop cap (optional) | 10 crops per image per epoch, only relevant if crops instead of images are drawn | Top 1% of images hold 21% of crops | 10, or none |
| Duplicates | At most one crop per duplicate group per class in a batch; split by the union of both duplicate lists | About 300 groups; 44 cross the current split | no |
| Negatives | Warm-up on all triplets, then semi-hard | Fuzzy class boundaries, tiny and truncated crops | warm-up length, margin, batch-hard as alternative |
| Augmentation | Flip, scale, colour jitter | Rare-class crops are repeated | strength |

---

## 5. How the sampling strategy influences training and the embedding space

The loss only ever sees the triplets that the sampling produces. The same network, loss and data can therefore give a useful or a useless embedding, depending only on which crops are drawn and which triplets are formed. This section follows that chain: sampling decides the gradient (training), and the gradient decides the geometry (embedding space). The effects described are what the theory and published experience lead us to expect; they are not results of an experiment, since nothing is trained in this assignment.

### 5.1 Influence on the training process

| Sampling choice | What happens during training |
| --- | --- |
| Random triplets | Almost every triplet is easy, so the loss is nearly 0 and the gradient is close to zero: training stalls. Averaging over all triplets, including the zeros, shrinks the gradient even further. |
| Only the hardest negatives | The largest gradients come from the most extreme cases, which are the mislabelled, tiny or truncated crops in this dataset. Early in training this can also push the embedding towards collapse (all crops at nearly one point), as reported for hardest negatives by Schroff et al. (2015) and Wu et al. (2017); Hermans et al. (2017) show that batch-hard mining can work well, so the risk depends on the data. |
| Semi-hard negatives after a warm-up | Each triplet has a positive loss but a correct order, so the gradient is informative and bounded. Training is more stable, and steps stay informative until the semi-hard band empties late in training (Wu et al., 2017), which the fallback rule in section 3.4 handles. |
| Class frequencies as they come (random batches) | 61% of the crops, and 83.5% of all triplets, have a biodegradable anchor. The updates are dominated by that class, and the rare classes are learned last and worst. |
| Class-balanced P x K batches | Each class supplies K anchors, one sixth of the anchors in every step, so every class has an equal say in the update. |
| Many crops from one image, or duplicates in one batch | Positives are too easy (same background, same lighting) and the loss can be reduced by matching photos rather than materials. If two copies carry different labels, a copy used as a negative gives a large loss for a pair that is in fact the same object, which is a wrong learning signal. |
| One crop per image and per duplicate group for each class in a batch | Positives come from different photos, so the network must find what the crops of one material have in common. |
| Drawing images instead of crops | A few crowded photos cannot dominate an epoch: each photo is drawn about as often as any other of its class. |
| Repeating rare-class crops more often | Without augmentation the network can memorise them; with flips, scale changes and colour jitter, repetition is harmless. |

Two quantities are worth watching during training (as also suggested by Hermans et al., 2017): the fraction of triplets with non-zero loss in a batch (close to 100% means the margin is too large or the embedding is still random; close to 0% means the sampling gives only easy triplets), and the mean positive and negative distances (collapse shows as both going to 0).

### 5.2 Influence on the embedding space

| Sampling choice | Effect on the geometry of the embedding space |
| --- | --- |
| Random batches, imbalanced | The biodegradable region is large and well shaped; rare classes are squeezed and their boundaries are set by the majority class. A rare-class crop near a boundary is placed with biodegradable, so a nearest-neighbour rule favours the majority. |
| Class-balanced batches | Clusters of comparable size and spacing; no class is favoured at the boundaries. |
| Easy-only triplets | The embedding stays close to what the pretrained network gave at the start; clusters are loose and touch, because nothing pushes them apart. |
| Hardest-only triplets | A collapsed embedding, or clusters fitted to label noise: a mislabelled crop pulls its whole class region towards the wrong place. |
| Semi-hard triplets | Clusters separated by roughly the margin, with a cleaner boundary; the spread inside a class is not forced to zero. |
| Same-photo positives and duplicates | Small islands of crops from one photo appear, so that the nearest neighbours of a crop are other crops of the same picture rather than other objects of the same material; retrieval then looks good on paper and fails on new photos. |
| Distinct photos for the positives | The space is organised by material and shape, not by photo, background or camera. |
| Excluding crops below 24 pixels | No catch-all region where unrecognisable crops pile up. The price is that very small objects (24% of all crops are under 24 pixels) are never trained on, so their embeddings will be less reliable; Task 2 should report results by crop size. |
| Focus on confusable pairs (glass, metal, plastic) | These are the classes that provide the semi-hard negatives, so they receive most of the gradient and their boundaries are sharpened first. If the focus is too strong, the network may separate them by background cues (a transparent bottle on a white table) rather than by material. |

### 5.3 What the final space is expected to look like

All K - 1 positives of an anchor are used, rather than only the farthest one (as batch-hard mining would), so no single distant positive dominates and the loss does not force one compact ball per class. A class may form several sub-clusters, for example fruit and vegetable crops within biodegradable, or bottles and bags within plastic. This is acceptable because nearest-neighbour retrieval and classification (the subject of Task 2) only need the nearest crops to be of the right class, not one centre per class. The classes that share shapes, such as glass, metal and plastic, are expected to remain the closest together; the clusters of cardboard and paper, which look alike and have a fuzzy label boundary, are expected to overlap most.

### 5.4 Before and after

| | Random batches and triplets | Our strategy |
| --- | --- | --- |
| Share of anchors from the biodegradable class | 83.5% of all triplets (61% of the crops) | 16.7% (one class in six) |
| Exposure per epoch | Biodegradable crops make up 61% of what is seen | Each class one sixth; each eligible image drawn about 2 to 4 times per epoch (ratio about 1.9 between classes) |
| Positives of one class in a batch | Often from the same photo (K = 16 plastic crops come from different photos only half of the time) | From different photos and different duplicate groups, by construction |
| Informative triplets | Few; mostly easy | Semi-hard, with a warm-up |
| Tiny, unrecognisable crops | Included | Excluded below 24 pixels |

---

## 6. Limits, alternatives and references

### 6.1 Limits of the proposal

- **Nothing is trained.** The effects in section 5 are expectations from the theory and from published studies. The numerical settings (24 pixels, K = 8 or 16, margin 0.2, the epoch and warm-up length, the optional cap of 10 crops per image) are design choices, and each would be tuned on the validation split, never on the test split.
- **Labels.** The class names are inferred from the file names, and the boundary between paper and cardboard labels is fuzzy. Semi-hard mining should reduce the influence of borderline or wrong labels but cannot remove it.
- **Small and truncated objects.** Crops below 24 pixels are not trained on, and boxes at the image border or inside other boxes show only part of an object. Embeddings of such crops will be less reliable, so results should be reported by crop size.
- **Duplicates.** The team's current split leaves 44 of our 307 duplicate groups across training, validation and test (section 4.4), which must be corrected before any evaluation. Our search itself is not guaranteed to be complete: in our own test with planted copies, its automatic stage found about 93% of them. Missed duplicates would make validation and test scores slightly optimistic.
- **Background shortcuts.** Many images come from similar stock-photo settings; the network may use a background (for example a white table behind a transparent bottle) instead of the material. Distinct photos for the positives and augmentation reduce, but do not remove, this risk.
- **Sensitivity of the loss.** The triplet loss depends more on its sampling than the alternatives below. In a large benchmark with tuned hyperparameters, classification-style and proxy-based losses beat the triplet loss on all three datasets, by about 1.4 to 6.4 points of Precision@1 in the 512-dimensional setting of their Tables 4 to 6 (Musgrave et al., 2020). The triplet loss there was trained without mining, so that comparison does not measure semi-hard mining, and the authors conclude that the methods perform similarly overall. Our choice rests on the fit with this dataset and task (section 2.4), not on a benchmark lead; the proxy-based alternative is the first we would try if the triplet loss underperformed on validation.
- **Closed set.** All six classes are seen in training. The design does not tell how well the space generalises to a material that was not in the training data.

### 6.2 Alternatives and why we did not choose them

| Alternative | How it works | Why not the main choice here |
| --- | --- | --- |
| Contrastive loss (Hadsell et al., 2006; lecture deck 07, page 16) | Works on pairs: positive pairs are pulled together, negative pairs are pushed beyond a margin (the margin applies to negative pairs only) | Compares each pair with a fixed distance and not one pair with another, so it ignores relative distances; it needs the same sampling care as the triplet loss |
| Proxy-NCA and Proxy-Anchor (Movshovitz-Attias et al., 2017; Kim et al., 2020; lecture deck 07, pages 17 and 18) | One learned vector (proxy) per class replaces the sampled positives and negatives, so no triplet mining is needed | A strong alternative: according to the lecture it converges faster and costs less (deck 07, page 18), and Proxy-Anchor is reported to be robust to noisy labels (Kim et al., 2020). We did not choose it because one proxy per class pulls visually multi-modal classes (fruit and vegetables, bottles and bags) towards a single point, and because its cost advantage is small when every batch already contains all six classes (section 2.4). It is the first alternative we would try, with several proxies per class, if mining proved unstable |
| ArcFace (Deng et al., 2019; lecture deck 07, pages 20 and 21) | A classification loss with an additive angular margin between each embedding and its class centre | Developed for large-scale face recognition with tens of thousands of identities, where enumerating triplets is impractical (Deng et al., 2019); with six classes that motivation largely disappears. It is a strong option and performs well in retrieval benchmarks (Musgrave et al., 2020); we keep the triplet loss for the reasons in section 2.4 |
| Multi-similarity loss and supervised contrastive loss (Wang et al., 2019; Khosla et al., 2020) | Weight the pairs of a batch by difficulty: the multi-similarity loss first mines informative pairs and then weights them; the supervised contrastive loss uses all positives and negatives of a batch, and its softmax form weights hard ones more | Would work on the same P x K batches; they are not covered in the lecture |

The sampling strategy of section 4 does not depend on the triplet loss. The size filter, the image-first drawing, the duplicate rule and the balanced P x K batches would be used unchanged with any of the losses above; only the mining rule of section 3.4 is specific to the triplet loss.

### 6.3 How this answers the task

| Requirement of the task | Where |
| --- | --- |
| Choice of a loss, its principle and objective | Section 2 |
| How samples are selected and combined for this loss | Section 3 |
| A sampling strategy designed for this dataset, with justification | Sections 1.3 and 4 |
| Influence of the sampling on training and on the embedding space | Section 5 |
| Limits and alternatives | Section 6 |

### 6.4 References

1. Hadsell, R., Chopra, S., LeCun, Y. Dimensionality reduction by learning an invariant mapping. CVPR 2006. http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
2. Schroff, F., Kalenichenko, D., Philbin, J. FaceNet: a unified embedding for face recognition and clustering. CVPR 2015. https://arxiv.org/abs/1503.03832
3. Hermans, A., Beyer, L., Leibe, B. In defense of the triplet loss for person re-identification. 2017. https://arxiv.org/abs/1703.07737
4. Wu, C.-Y., Manmatha, R., Smola, A. J., Krähenbühl, P. Sampling matters in deep embedding learning. ICCV 2017. https://arxiv.org/abs/1706.07567
5. Movshovitz-Attias, Y., Toshev, A., Leung, T. K., Ioffe, S., Singh, S. No fuss distance metric learning using proxies. ICCV 2017. https://arxiv.org/abs/1703.07464
6. Wang, X., Han, X., Huang, W., Dong, D., Scott, M. R. Multi-similarity loss with general pair weighting for deep metric learning. CVPR 2019. https://arxiv.org/abs/1904.06627
7. Deng, J., Guo, J., Xue, N., Zafeiriou, S. ArcFace: additive angular margin loss for deep face recognition. CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html
8. Kim, S., Kim, D., Cho, M., Kwak, S. Proxy anchor loss for deep metric learning. CVPR 2020. https://arxiv.org/abs/2003.13911
9. Khosla, P., et al. Supervised contrastive learning. NeurIPS 2020. https://arxiv.org/abs/2004.11362
10. Musgrave, K., Belongie, S., Lim, S.-N. A metric learning reality check. ECCV 2020. https://arxiv.org/abs/2003.08505
11. Lecture "Deep Metric Learning" (deck 07): page 4 (robustness to class imbalance), 7 and 8 (parts of deep metric learning, random and informative sampling), 9 and 10 (sampling strategies), 12 (pretrained network), 14 and 15 (triplet loss), 16 (contrastive loss), 17 to 19 (Proxy-NCA and its comparison with the triplet loss), 20 and 21 (ArcFace).
