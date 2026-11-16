# Assignment 2, Task 2: Analysis and use of the learned embedding space

Deep metric learning for automated waste sorting (theoretical solution; nothing is implemented or trained).

Contents: 0 Starting point; 1 Retrieval of similar waste objects; 2 Assigning an unknown sample to a known category; 3 Visualising and analysing the structure; 4 Quantitative evaluation; 5 Limits; 6 How this answers the task, and references.

**Task.** Assume that the deep metric learning model from Task 1 has been successfully trained. Develop an approach for analysing and using the resulting embedding space. Address (a) how the embedding can be used to retrieve similar waste objects for a query image, (b) how an unknown sample could be assigned to one of the known waste categories, (c) how the structure of the space can be visualised and analysed, and (d) how its quality can be evaluated quantitatively. For each aspect, describe a suitable method and explain how its results should be interpreted.

**Answer in one paragraph.** Every eligible training crop is embedded once and stored with its class, source image and duplicate group; this gallery of 39,574 vectors is the reference for everything else. (a) A query image is turned into crops by our YOLO26s detector, each crop is embedded, and the nearest gallery crops are returned, at most one per source photo. (b) An unknown crop is assigned by a class-balanced k-nearest-neighbour rule (the k nearest crops of *each* class, so that the 55% biodegradable share of the gallery cannot outvote the rest), with a distance threshold and an ambiguity margin that let the system answer "unknown" or "uncertain" instead of guessing, as in the lecture's face-recognition example. (c) The structure is shown with PCA and t-SNE on a class-balanced sample of test crops, and analysed in the full 128-dimensional space with a class-to-class distance matrix and a nearest-neighbour confusion matrix. (d) Quality is measured with Recall@K, MAP@R, class-balanced k-NN accuracy and macro-F1, always against a stated chance level, per class, per crop size, with confidence intervals over test images and with the leakage rules of Task 1. Every setting is chosen on the validation split; the test split is used once.

---

## 0. Starting point

### 0.1 What Task 1 delivers

The trained model is the one specified in Task 1 (sections 2 to 4): a pretrained convolutional backbone with a small head that maps an object crop to a vector of 128 numbers, L2-normalised so that every embedding lies on the unit sphere, trained with the triplet loss (starting margin `m = 0.2`, tuned on validation) on class-balanced 6 x K batches with semi-hard negatives. Two consequences matter for everything below:

- **Distances lie between 0 and 2.** For unit vectors, `d^2 = 2 - 2 cos(theta)`, so ranking by Euclidean distance and ranking by cosine similarity give exactly the same order. We therefore use one distance, `d(x, y) = || f(x) - f(y) ||`, throughout.
- **The scale of the space is known.** Six perfectly separated class centres would be `1.55` apart (Task 1, section 2.3), and the loss asks for a gap of at least the margin `m` (0.2 unless validation chose another value) between a positive and a negative. These two values are the ruler against which distances in sections 2 to 4 are read.

The model sees **crops**, not whole images, and was trained only on crops at least 24 pixels on the short side, padded by the fixed context margin chosen in Task 1 (between 10% and 20%), squared and resized to the fixed input size. Every crop used below is prepared in exactly this way, without augmentation.

### 0.2 Our data, split and detector

All numbers come from our cleaned labels (17 zero-area rows removed) and our fixed split (`assignment-1/data/split/manifest.csv`: seed 42, 70/15/15, whole duplicate groups kept together). The image counts, box counts and class presence recomputed for this document match `assignment-1/data/split/summary.json` exactly for all three subsets.

| | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Images | 7,325 | 1,570 | 1,569 |
| All crops (valid boxes) | 51,861 | 11,114 | 11,098 |
| Eligible crops (>= 24 px short side) | 39,574 | 8,276 | 8,353 |
| Images with at least one eligible crop | 7,325 | 1,570 | 1,568 |
| Images containing more than one class | 758 | 156 | 152 |

Eligible crops per class:

| Class | Train (gallery) | Share of gallery | Validation | Test | Test images with an eligible crop |
| --- | ---: | ---: | ---: | ---: | ---: |
| biodegradable | 21,777 | 55.0% | 4,363 | 4,441 | 342 |
| cardboard | 3,080 | 7.8% | 665 | 678 | 225 |
| glass | 4,919 | 12.4% | 1,085 | 1,092 | 398 |
| metal | 3,375 | 8.5% | 766 | 747 | 254 |
| paper | 2,797 | 7.1% | 633 | 625 | 264 |
| plastic | 3,626 | 9.2% | 764 | 770 | 215 |

The detector of Assignment 1 is COCO-pretrained YOLO26s, fine-tuned on the same split, run at 640 pixels with `max_det = 400` (`assignment-1/train_detector.py`). The selected checkpoint (best of 60 epochs on validation) reaches, on the 1,569 test images and 11,098 test boxes, precision 74.9%, recall 54.6%, mAP50 61.9% and mAP50-95 44.5% (`assignment-1/reports/task-2-results.md`). Its recall is lowest for biodegradable (43.5%) and plastic (48.5%) and highest for glass (64.6%).

### 0.3 Three rules carried over from Task 1

1. **Gallery and queries come from different splits.** The gallery is the 39,574 eligible training crops. Queries for tuning come from validation, queries for the final numbers from test. The embedding was trained on the gallery crops, so a training crop must never be used as a query.
2. **No copy of a query may count as its match.** Our split keeps the 267 reviewed duplicate groups together, but Task 1 (section 4.4) found 23 test images whose copy is still in training or validation. Before evaluation the split is rebuilt from the union of both duplicate lists, or those test images are removed; otherwise a test crop can retrieve its own copy.
3. **Within one split, a crop's own image and duplicate group are excluded.** Whenever crops are compared with crops of the same split (the measurements in section 3.3, NMI and silhouette in section 4.2, and the same-image check in section 4.6), crops from the query's own image and its duplicate group are removed from the candidates. This is not a detail: 89% of the eligible test crops (7,435 of 8,353) have another crop of the same class in the same photo, so without this rule "same class" would mostly mean "same photo".

---

## 1. Retrieving similar waste objects for a query image

### 1.1 Method

**Offline, once.** Embed every eligible training crop and store the vector together with its class, its source image, its duplicate group and its box size. The gallery is 39,574 x 128 numbers, about 20 MB in 32-bit floats, so it fits in memory; one query needs about 5 million multiply-adds (39,574 x 128). Exact search over the whole gallery is therefore cheap, and no approximate index is needed. Exact search also means retrieval errors can only come from the embedding, not from the index.

**Online, per query image.**

1. **From image to crops.** A query is a photo, and a photo of waste usually holds several objects (median 2, up to 358, often of different classes; Task 1, section 1.2). An image as a whole has no single category, so the query photo is passed through our YOLO26s detector, with the same 640-pixel input and `max_det = 400` as in Assignment 1. Each predicted box becomes one query crop, prepared as in section 0.1. A user may also select one box by hand. Either way, retrieval answers "objects similar to *this* object", not "photos similar to this photo".
2. **Embed** each query crop with the trained network `f`.
3. **Rank** all gallery crops by distance `d(q, g)` to the query and keep the nearest ones.
4. **Make the list useful for this dataset.**
   - *One result per source photo.* Crowded photos contribute dozens of crops of one class (one training image holds 125 eligible crops; the top 1% of training images hold 13.3% of all eligible crops). Without this rule a single crowded pile of fruit could fill the whole top 10. Only the nearest crop of each gallery photo is kept, and the next photo is taken instead.
   - *Optional class filter.* "Show me similar plastic objects" restricts the ranking to one class; this is the same search over a subset of the gallery.
   - *Size flag.* A query crop below 24 pixels is outside what the model was trained on (Task 1, section 4.1). It is still searched, but marked as unreliable. In the test split 24.7% of all boxes (2,745 of 11,098) are below 24 pixels, so this case is common.
5. **Return** the top results with thumbnail, class, distance and source image.

### 1.2 How to interpret the results

- **The class of the neighbours.** If the nearest crops share the query's class, the space has grouped the material correctly. If a plastic bottle retrieves glass bottles, the embedding has learned shape rather than material, which Task 1 predicts for the glass, metal and plastic group; paper and cardboard neighbours of each other are also expected (Task 1, section 5.3).
- **The distance, not only the rank.** The nearest neighbour always exists, even for a query the model cannot represent. Its distance tells whether it is a real match. A top-1 distance well below the typical distance between two same-class validation crops is a confident match; a top-1 distance comparable to the distance between class centres (around 1.55 for perfectly separated classes) means that nothing in the gallery is really close. Section 2.3 turns this into a threshold.
- **The gap between neighbours.** A clear jump in distance after the first few results means a tight group of near-identical objects; a flat curve means the query lies in a region where many crops are about equally similar.
- **What the neighbours have in common besides the class.** Viewing the retrieved thumbnails side by side is the most direct check on shortcuts. If the neighbours share the background, the lighting or the watermark rather than the material, the network has learned photo features instead of material features (Task 1, section 6.1). If the neighbours come from the photo that the query image was copied from, the split has a duplicate leak (rule 2 in section 0.3).
- **Confidence of the detector stage.** A wrong or loose box gives a wrong query. A retrieval failure for a box that the detector itself labelled with low confidence should be blamed on the detection, not on the embedding; this is why the quantitative evaluation in section 4 uses ground-truth boxes.

---

## 2. Assigning an unknown sample to one of the known categories

Here an unknown sample is a crop whose class we do not know, such as an object proposed by the detector in a new image. The lecture distinguishes "default" classification, which learns a decision boundary, from metric classification, which builds a new space and decides by a threshold (deck 07, page 5). The embedding gives the second kind. The basic tool is the k-nearest-neighbour rule from the recap lecture (deck 03, pages 33 to 37): a new point takes the class of its nearest labelled points.

### 2.1 Why a plain k-NN majority vote is biased on our data

In a plain majority vote over the k nearest gallery crops, the classes enter in proportion to their size in the gallery, and biodegradable is 55.0% of it. The effect is easy to quantify. If the embedding carried no information at all, so that the neighbours were random gallery crops, then among k = 10 neighbours there would be at least one biodegradable crop with probability `1 - 0.45^10 = 99.97%`, and on average 5.5 of the 10 would be biodegradable, while a paper crop would be one of the 10 neighbours only half the time (`1 - 0.929^10 = 0.52`). In a good embedding the real neighbours are far from random, but near class boundaries, where the decision matters, the vote is still tilted towards the majority. This is the same imbalance that Task 1 removed from training with balanced batches; it must also be removed from the decision.

### 2.2 Method: class-balanced k-NN

For a query crop `q` and every class `c`:

1. find the `k` gallery crops of class `c` nearest to `q`, taking at most one crop per gallery photo (as in section 1.1);
2. compute the class score `s_c(q)` = the mean distance from `q` to these `k` crops.

The predicted class is the class with the smallest score. Every class contributes exactly `k` candidates, so the size of a class in the gallery no longer matters, only how close its nearest members are. Taking a few nearest crops per class rather than one class centre matches the expected shape of the space: Task 1 (section 5.3) expects sub-clusters within a class, for example fruit and vegetables within biodegradable, or bottles and bags within plastic, and the mean of a class with two sub-clusters can lie between them, far from both. `k` is chosen on validation, from about 3 to 20; a small `k` follows the local shape of the class, a large `k` is more robust to single mislabelled gallery crops.

Two simpler rules serve as comparisons:

- **Nearest class centre** (the mean embedding of each class, renormalised to length 1): six distances per query, the cheapest rule, but it suffers from the sub-cluster problem above. If it is clearly worse than class-balanced k-NN on validation, that is direct evidence of sub-clusters.
- **Linear classifier on the frozen embeddings**, the "linear evaluation protocol" from the self-supervised learning lecture (deck 12, page 13). It shows how much class information is linearly available in the embedding; it is used as a reference in section 4, not as the sorting rule.

### 2.3 Saying "uncertain" instead of guessing

The face-recognition example in the lecture (deck 07, pages 24 to 31) compares a new face with a gallery of known people and outputs a name only if the distance is below a threshold (0.6); otherwise it outputs `unknown_person`. The humpback-whale challenge (deck 07, page 22) poses the same question: one of the known individuals, or a new one. The deepfake lecture uses the same idea for samples outside the known distribution: the k-NN distance to the training features, or the Mahalanobis distance from the training distribution, flags samples that lie outside it (deck 13, page 32). In waste sorting there are at least three kinds of crop for which a forced decision is wrong:

- detector false positives (background, shadows, parts of a table) that are no waste object at all;
- objects of a material that is not one of the six classes, or that is a mix of them (a plastic-coated paper cup, a metal lid on a glass jar);
- tiny or truncated crops: 24.7% of the test boxes are below 24 pixels, and half of the eligible paper crops in the test split (313 of 625) touch the image border, so they show only part of an object.

We therefore add two checks to the rule of section 2.2:

1. **Distance threshold `tau`.** If even the best class score is too large, `min_c s_c(q) > tau`, the crop is not close to any known class and the answer is "unknown". `tau` is set on validation: for every eligible validation crop, compute the score `s_c` of its own true class and take, for example, the 95th percentile of these scores. By construction about 5% of genuine validation crops would then be rejected; the percentile is the operating point, chosen by how costly a wrong sort is compared with a manual check. The value 0.6 from the face example belongs to another model and another space and is not transferred.
2. **Ambiguity margin `delta`.** If the best and the second-best class are almost equally close, `s_(2)(q) - s_(1)(q) < delta`, the answer is "uncertain between these two classes". On our data this check is meant for glass/plastic, glass/metal and paper/cardboard. In the test split, images named after glass contain 109 eligible metal crops and 95 eligible plastic crops, so these materials appear side by side. `delta` is chosen on validation in the same way as `tau`.

The output for each crop is therefore one of: a class with its score, "uncertain between class A and class B", or "unknown".

### 2.4 How to interpret the results

- **Score and margin as confidence.** A small best score and a large margin to the second class is a confident decision. A small best score with a small margin means the crop lies where two classes meet. That is expected for the confusable pairs, but if it happens often for an unrelated pair (say, biodegradable and metal), the clusters are not separated.
- **What the rejected crops are.** The crops rejected as "unknown" should be looked at. If they are mostly tiny, truncated or non-waste detector boxes, the threshold works as intended. If they are ordinary, well-visible objects of one class, that class is poorly represented in the space (too few or too uniform training crops), not unknown.
- **Agreement with the detector.** For a detected box we then have two labels, the class predicted by YOLO26s and the class from the embedding. Where both agree, the decision is supported by two different models; where they disagree, the crop is a candidate for manual checking, and on labelled dataset images a candidate for a label error. On a sorting line this is a cheap second opinion.
- **The rate of "uncertain" per class pair** is itself an analysis result: it shows which material boundaries the embedding does not resolve, which links directly to the confusion matrix in section 4.

---

## 3. Visualising and analysing the structure of the space

The embedding has 128 dimensions, so it cannot be looked at directly. The recap lecture motivates clustering and analysis in a learned latent space instead of on raw images, because of the curse of dimensionality (deck 04, slides 129 and 130; PDF pages 124 and 125). We use two kinds of tools: **projections** to two dimensions, which are good for seeing but distort distances, and **measurements in the full 128-dimensional space**, which are exact but harder to see. Every conclusion drawn from a projection is checked with a measurement.

### 3.1 Which crops are shown

Plotting all 8,353 eligible test crops would show mainly biodegradable (53.2% of them) and, inside it, a few crowded photos. We therefore draw a **class-balanced sample**: up to 600 crops per class, chosen image-first as in Task 1 (section 4.2), with at most three crops per image and class. Every class has between 625 (paper) and 4,441 (biodegradable) eligible test crops in 215 to 398 test images. If a class yields fewer than 600 crops under the three-per-image rule, all of its eligible crops are taken, and the class sizes are stated on the plot. Each point keeps its metadata (class, source image, duplicate group, crop size, whether the box touches the image border, whether the image holds several classes, and the class named in the file name), so that one map can be coloured in several ways.

### 3.2 Projections: PCA and t-SNE

- **PCA** (deck 03, page 44) is a linear projection onto the directions of largest variance. The first two components give a first, undistorted overview. The cumulative explained variance tells how many of the 128 dimensions the model actually uses. Six well-separated class centres need five dimensions (the corners of a simplex), so if about 90% of the variance lies in five or six components, the space is organised mainly by class. If the variance is spread over many components, the crops also vary strongly within classes (sub-clusters, size, background).
- **t-SNE** (van der Maaten and Hinton, 2008; named for visualising representations in deck 12, page 13) places points in 2D so that each point keeps its nearest neighbours. It shows clusters and sub-clusters much more clearly than PCA. We first reduce to 50 dimensions with PCA, as is common, then run t-SNE with a perplexity of about 30. Because the result depends on the random start and on the perplexity, we run several seeds and perplexities (for example 10, 30 and 50) and only trust structures that appear in all of them. These settings are chosen on the validation crops.

**The same map, coloured differently:**

| Colouring | What it reveals |
| --- | --- |
| by class | whether the six classes form separate regions, which classes touch, and where sub-clusters exist |
| by crop size (24 to 32, 32 to 64, at least 64 pixels) | whether small crops collect in a "catch-all" region instead of with their class |
| by border contact | whether truncated objects are placed apart from complete ones |
| by source image or duplicate group | whether crops cluster by photo instead of by material (small islands of one photo) |
| by single-class versus mixed-class image | whether objects from crowded mixed scenes (152 test images) are placed differently |

Next to the colours, **thumbnails** of the crops in a cluster are shown in a grid. This is how a sub-cluster is identified as "fruit" versus "vegetables", or "bottles" versus "bags".

**How to read the maps, and how not to.**

- In t-SNE, neighbourhoods are meaningful; the **size of a cluster and the distance between clusters are not** (Wattenberg et al., 2016). A large biodegradable region or a big gap between metal and paper says nothing about the real spread or separation. Those questions are answered in 3.3, in the full space.
- Expected picture if training worked (Task 1, section 5.3): six regions; glass, metal and plastic close to each other; paper and cardboard overlapping most; sub-clusters inside biodegradable and plastic. A region dominated by one source photo, or by tiny crops of mixed classes, is a failure sign, not a class.
- A crop placed deep inside another class's region is either an embedding error or a label error. Its thumbnail tells which. Task 1 noted that paper and cardboard labels are not always consistent.

### 3.3 Measurements in the full 128-dimensional space

1. **Class-to-class distance matrix (6 x 6).** For every pair of classes, the mean distance between their test crops; on the diagonal, the mean distance within each class. A class is well defined when its diagonal value is clearly smaller than its row's off-diagonal values. The values are read against the ruler of section 0.1: inter-class distances near 1.55 mean the classes are as far apart as six classes can be, and an inter-class gap smaller than the margin `m` means the loss did not reach its objective for that pair.
2. **Nearest-neighbour confusion matrix (6 x 6).** For every test crop, the class of its nearest gallery crop (section 1). Row = true class, column = neighbour class, each row normalised to 1. Off-diagonal mass shows which classes are mistaken for which. Unlike the t-SNE map, it is measured in the real space. It also shows asymmetric confusion: for example, cardboard crops whose neighbours are paper does not imply the reverse.
3. **Silhouette per class** (Rousseeuw, 1987). For each crop, `(b - a) / max(a, b)`, where `a` is its mean distance to the other crops of its class and `b` its mean distance to the nearest other class. Values near 1 mean well inside its class, near 0 on a boundary, below 0 closer to another class. The per-class mean summarises compactness and separation in one number, and the crops with negative values are the ones to inspect.
4. **Sub-clusters within a class.** k-means with two to five clusters inside a single class, inspected with thumbnails. This shows what the class consists of in the space (for example, fruit versus vegetables). It does not have to be "fixed": Task 1 (section 5.3) allows sub-clusters, since retrieval and k-NN only need the nearest crops to be correct.

In all four measurements, distances between crops of the same split exclude crops from the same image and the same duplicate group (rule 3 in section 0.3). Otherwise the 89% of test crops that have a same-class partner in their own photo would make every class look more compact than it is.

---

## 4. Evaluating the quality of the space quantitatively

### 4.1 Protocol

- **Boxes.** Queries are the ground-truth boxes of the test split, cropped as in section 0.1, not the detector's boxes. The embedding is evaluated on its own; detector errors are measured in Assignment 1. (With a test recall of 54.6%, and only 43.5% for biodegradable, almost half of the test objects would otherwise be missing from the evaluation, and unevenly across classes.) A second, end-to-end figure on detector boxes can be reported separately, matching detections to ground truth by IoU, as for detection mAP (deck 04, slide 69; PDF page 68).
- **Queries and gallery.** 8,353 eligible test crops as queries against the 39,574 training crops as gallery, after the duplicate correction (rule 2 in section 0.3).
- **One test run.** `k`, `tau`, `delta`, the t-SNE settings and any comparison between models are decided on validation queries (8,276 eligible crops) against the same gallery. The test numbers are computed once at the end. This is the same rule as in Assignment 1.

### 4.2 Metrics

| Metric | Definition | What it measures here |
| --- | --- | --- |
| **Recall@K** (K = 1, 5, 10) | fraction of queries with at least one gallery crop of the right class among the K nearest | the lecture's metric for comparing DML losses (Recall@1 / Top-1, deck 07, page 19); directly the success rate of retrieval in section 1 |
| **Precision@10** | fraction of the 10 nearest gallery crops (one per photo) that have the right class | how clean the whole result list is, not only its first entry |
| **MAP@R** (Musgrave et al., 2020) | for a query of class `c`, with `R` = number of gallery crops of `c`: look at the first `R` results, add up the precision at every position that holds a correct crop, and divide by `R` (1 means all `R` same-class crops come first) | the ranking of all same-class crops at once; more informative than Recall@1, which saturates for good models |
| **Class-balanced k-NN accuracy, per-class precision and recall, macro-F1** | the rule of section 2.2 applied to every query (precision `TP/(TP+FP)`, recall `TP/(TP+FN)`, deck 04, slide 68; PDF page 67) | the quality of category assignment, the use case of section 2 |
| **Coverage and accuracy with rejection** | share of queries not rejected by `tau` and `delta`, and accuracy on those | the trade-off of section 2.3: more rejections give fewer but more reliable decisions |
| **NMI of k-means (k = 6)** and **mean silhouette** | agreement between unsupervised clusters and the classes; compactness and separation (section 3.3) | the clustering quality named in deck 12, page 13 |

**Which ranking is scored.** Recall@K and MAP@R are computed on the full ranking of all gallery crops, as defined in the literature, so that they stay comparable with published values. Precision@10 and the k-NN rule use the list with one crop per gallery photo, because that is what a user sees (section 1.1) and what the assignment rule uses (section 2.2).

**Macro, not only micro.** A micro average over all test crops is 53.2% biodegradable (4,441 of 8,353), so it mostly measures one class. We report the macro average (each class weighted 1/6) as the main figure and the micro average next to it.

### 4.3 What the numbers must be compared with

A score is only meaningful against what it would be without a useful embedding. For our split, a random ranking of the gallery gives these Recall@K values (probability that at least one of K random gallery crops has the query's class, `1 - (1 - p_c)^K` with `p_c` the gallery share of class `c` from section 0.2; drawing with replacement makes this a close approximation, since K is tiny compared with the gallery):

| Chance level | K = 1 | K = 5 | K = 10 |
| --- | ---: | ---: | ---: |
| biodegradable | 0.550 | 0.982 | 1.000 |
| cardboard | 0.078 | 0.333 | 0.555 |
| glass | 0.124 | 0.485 | 0.735 |
| metal | 0.085 | 0.360 | 0.590 |
| paper | 0.071 | 0.307 | 0.520 |
| plastic | 0.092 | 0.382 | 0.617 |
| micro (test class mix) | 0.336 | 0.703 | 0.821 |
| macro | 0.167 | 0.475 | 0.669 |

Three conclusions follow for interpretation. A Recall@1 of 0.6 is barely above chance for biodegradable but excellent for paper, so per-class values must be read against their own chance level. Recall@10 is at least 0.82 micro even for a random ranking, so it says little; Recall@1, Precision@10 and MAP@R carry the information. For classification, always answering "biodegradable" gives 53.2% micro accuracy but a macro recall of only 1/6. That is why macro-F1 is the main classification figure.

Two learned baselines complete the comparison:

- **The pretrained backbone before metric learning.** The same protocol is run on the backbone features before Task 1's training. The difference between the two is what deep metric learning added. A small difference would mean that pretraining alone already separates the materials and the triplet training contributed little.
- **The linear probe** (section 2.2). If a linear classifier is much better than k-NN, the class information is present in the embedding, but the distances do not yet reflect it well.

### 4.4 Breakdowns

Every metric is reported per class and also split by the properties that Task 1 identified as risks:

| Breakdown | Groups in the test split | Why |
| --- | --- | --- |
| crop size (short side) | 24 to 32 px: 1,114 crops; 32 to 64 px: 2,981; at least 64 px: 4,258; below 24 px (not trained, reported separately): 2,745 | Task 1 excluded small crops from training and asked for results by size (section 5.2) |
| border contact | 2,041 of the 8,353 eligible crops touch the border (within 2 pixels), including 313 of 625 paper and 214 of 678 cardboard crops | truncated objects are expected to be harder |
| single-class versus mixed-class image | 152 of 1,569 test images hold several classes | crops from mixed scenes share background with other materials |
| duplicate correction | with and without the 23 test images that have copies in training or validation (Task 1, section 4.4) | the difference shows how much leakage inflated the scores |

### 4.5 Uncertainty of the numbers

Crops from one photo are not independent (same light, same background, often the same pile). Confidence intervals are therefore computed with a bootstrap over **test images**, not crops: resample the 1,568 test images that contain eligible crops with replacement about 1,000 times, recompute each metric, and report the 2.5th and 97.5th percentiles. The intervals are expected to be widest for the classes with the fewest test images, plastic (215) and cardboard (225), and narrowest for glass (398). To compare two models or settings, the same resampled images are scored with both, and the interval of the *difference* is reported; they are only called different if that interval excludes 0.

### 4.6 How to interpret combinations of results

| Observation | Interpretation |
| --- | --- |
| Recall@1 well above the per-class chance level for every class, macro-F1 high | the space separates the materials; retrieval and k-NN assignment can be used |
| high Recall@1 but low NMI | the classes consist of several sub-clusters, so k-means with k = 6 does not find them as six groups; acceptable for retrieval (Task 1, section 5.3) and visible in the thumbnails of section 3.3 |
| low Recall@1 for paper and cardboard only, with most errors between the two | the fuzzy label boundary noted in Task 1, not necessarily an embedding failure; check the thumbnails of the confused crops |
| errors concentrated among glass, metal and plastic | the model separates by shape (bottle, can) more than by material; candidate for more training on these pairs (Task 1, section 4.5) |
| scores fall sharply from at least 64 px to 24 to 32 px | the resolution limit of the crops; small objects need a larger input or a different approach |
| scores drop clearly once the 23 leaked test images are removed | the earlier numbers were inflated by duplicates; only the corrected figure is valid |
| within-split retrieval much better without the same-image exclusion than with it | the model matches photos, not materials (rule 3 in section 0.3) |
| little gain over the pretrained backbone | metric learning added little; revisit the sampling or the loss (Task 1, section 6.2) |
| high accuracy after rejection but low coverage | the threshold is too strict, or many crops really are ambiguous; check which classes are rejected most |

---

## 5. Limits

- **Nothing is trained.** The methods and their interpretation are designed for the model of Task 1; no result is claimed. The numbers in this document describe our data and split (and the chance levels that follow from them), not the performance of a model.
- **Labels as ground truth.** All metrics measure agreement with the labels. The class names are confirmed by the dataset's `data.yaml` (Task 1, section 1.1), but the boundary between paper and cardboard labels is fuzzy. A part of the measured "error" may be label error, which is why the inspection steps in sections 2.4 and 3 are part of the method.
- **Closed set.** All six classes are seen in training. The "unknown" answer of section 2.3 is tuned on validation crops of the known classes only, so how well it rejects truly new materials can only be checked with examples of such materials, which the dataset does not contain.
- **Remaining duplicates.** Duplicates that neither duplicate search found can still link test and training images and make all scores slightly optimistic (Task 1, section 6.1).
- **Detector dependence in use.** The quantitative evaluation uses ground-truth boxes. In operation, retrieval and assignment can only be as good as the detector's boxes: the final YOLO26s finds 54.6% of the test objects, so in an end-to-end system about half of the objects would never reach the embedding, and biodegradable and plastic objects would be missed most often.

---

## 6. How this answers the task, and references

### 6.1 Requirement mapping

| Question of the task | Method | Interpretation |
| --- | --- | --- |
| Retrieve similar waste objects for a query image | Section 1.1: detector crops, nearest gallery crops, one per photo | Section 1.2 |
| Assign an unknown sample to a known category | Sections 2.2 and 2.3: class-balanced k-NN with threshold and ambiguity margin | Section 2.4 |
| Visualise and analyse the structure | Sections 3.1 to 3.3: PCA, t-SNE, distance and confusion matrices, silhouette, sub-clusters | Sections 3.2 and 3.3 |
| Evaluate the quality quantitatively | Sections 4.1 to 4.5: Recall@K, Precision@10, MAP@R, macro-F1, NMI, with chance levels, breakdowns and confidence intervals | Sections 4.3 and 4.6 |

### 6.2 References

1. Musgrave, K., Belongie, S., Lim, S.-N. A metric learning reality check. ECCV 2020. https://arxiv.org/abs/2003.08505 (MAP@R; weaknesses of Recall@K and NMI)
2. van der Maaten, L., Hinton, G. Visualizing data using t-SNE. Journal of Machine Learning Research 9, 2008. https://www.jmlr.org/papers/v9/vandermaaten08a.html
3. Wattenberg, M., Viégas, F., Johnson, I. How to use t-SNE effectively. Distill, 2016. https://distill.pub/2016/misread-tsne/
4. Rousseeuw, P. J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics 20, 1987. https://doi.org/10.1016/0377-0427(87)90125-7
5. Schroff, F., Kalenichenko, D., Philbin, J. FaceNet: a unified embedding for face recognition and clustering. CVPR 2015. https://arxiv.org/abs/1503.03832 (verification by a distance threshold, retrieval and clustering in one embedding)
6. Lecture "Deep Metric Learning" (deck 07): page 5 (metric classification decides by a threshold), 19 (Recall@1 comparison of losses), 22 (humpback whale: known individual or new; Google Landmarks: retrieval and recognition tracks), 24 to 31 (face recognition with a gallery, threshold and `unknown_person`).
7. Lecture "Recap" (deck 03): pages 33 to 37 (k-nearest neighbours), 44 (PCA). Lecture "Recap 2" (deck 04): slides 68 and 69 (precision, recall, IoU-based matching for AP; PDF pages 67 and 68), 129 and 130 (clustering in latent space; PDF pages 124 and 125).
8. Lecture "Self-supervised Learning" (deck 12): page 13 (linear evaluation, clustering and t-SNE for representation quality).
9. Lecture "Detection of Deepfakes and AI-Generated Visual Content" (deck 13): page 32 (k-NN distance, One-Class SVM and Mahalanobis distance for samples outside the known distribution).
10. Our data and code: `assignment-1/data/split/manifest.csv` and `summary.json` (split), `assignment-1/reports/task-1-findings.md` (dataset audit and detector), `assignment-1/train_detector.py` (YOLO26s settings), `assignment-2/task-1-loss-and-sampling.md` (model, loss and sampling).
