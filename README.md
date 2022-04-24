# Supervised ML & Sentiment Analysis

1. The image shows a gray box labeled "Prediction Function" that takes Features (X) and "Parameters (theta)" as inputs and predicts "Output (Y^)". The Output (Y^) goes as input into another gray box labeled "Cost (Output Y^ vs Label Y)" that also takes "Labels (Y)" as input. The output from this box goes into "Parameters (theta)" thus completing the loop: "Prediction Function" -> "Output (Y^)" -> "Cost (Output Y^ vs Label Y)" -> "Parameters (theta)" -> "Prediction Function"

2. The image shows a box that takes "I am happy because I am learning NLP" as input and outputs "Postive: 1". The internals of this box contain three boxes labeled "X" -> Train LR -> Classify.

# Vocabulary and Feature Extraction

1. The image shows a sparse vector of length |V| where the initial values are 1 and the remaining 0's. The vector is a representation of the text "I am happy because I am learning NLP". Below this vector is shown another vector [theta0, theta1, theta2, ..., thetan] (where n = |V|). This second vector is labeled (1) Large training time and (2) Large prediction time.

# Feature Extraction with Frequencies

1. The image shows two boxes labeled "Positive tweets" and "Negative tweets". The box labeled "Postive tweets" contains the tweets: "I am happy because I am learning NLP" and "I am happy". The box labeled "Negative tweets" contains the tweets: "Iam sad, I am not learning NLP" and "I am sad"

2. The image shows a table labeled "freqs: dictionary mapping from (word, class) to frequency": 

| Vocabulary | PosFreq(1) | NegFreq(0) |
--- | --- | ---
I | 3 | 3
am | 3 | 3
happy | 2 | 0
because | 1 | 0
learning | 1 | 1
NLP | 1 | 1
sad | 0 | 2
not | 0 | 1

3. The image shows the table labeled "I am sad, I am not learning NLP":

| Vocabulary | PosFreq(1) |
 --- | --- 
 I | **3**
 am | **3**
 happy | 2
 because | 1
 learning | **1**
 NLP | **1**
 sad | **0**
 not | **0**
 
 On the right of the table is the equation:
 
 X_m = [1, **SumOverW(freqs(w, 1))**, SumOverW(freqs(w, 0))]
 
 An arrow from the second term of the right hand side of equation above points to 8.
 
 4. The image shows the table labeled "I am sad, I am not learning NLP":
 5. 
 Vocabulary | NegFreq(0)
 --- | ---
 I | 3
 am | 3
 happy | 0
 because | 0
 learning | 1
 NLP | 1
 sad | 2
 not | 1
 
 On the right of the table is the equation:
 X_m = [1, SumOverW(freqs(w, 1)), **SumOverW(freqs(w, 0))**]
 
 An arrow from the third term of the right hand side of equation above points to 11
