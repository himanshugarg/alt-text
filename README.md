# Week 2

## Bayes' Rule

Conditional probabilities help us reduce the sample search space. For example given a specific event already happened, i.e. we know the word is happy:

A Venn diagram diagram illustrating the embedded equation P(Positive|"happy") = P(Positive ∩ "happy") / P("happy"). The Venn diagram contains a rectangle representing the Corpus. Within it are two partially overlapping circles - blue the smaller one representing "happy" and green i.e. the larger one representing "Positive". The partially overlapping area represents the intersection "happy" ∩ Positive.

Then you would only search in the blue circle above. The numerator will be the red part and the denominator will be the blue part. This leads us to conclude the following: 

P(Positive|"happy") = P(Positive ∩ "happy")/P("happy")

P("happy"|Positive) = P("happy" ∩ Positive)/P(Positive)

Substituting the numerator in the right hand side of the first equation, you get the following: 

P(Positive|"happy") = P("happy"|Positive) x P(Positive)/P("happy")

Note that we multiplied by P(positive) to make sure we don't change anything.  That concludes Bayes Rule which is defined as 

P(X∣Y)=P(Y∣X)P(X)/P(Y)

# Week 1
## Reading - Supervised ML & Sentiment Analysis

1. The image shows a gray box labeled "Prediction Function" that takes Features (X) and "Parameters (theta)" as inputs and predicts "Output (Y^)". The Output (Y^) goes as input into another gray box labeled "Cost (Output Y^ vs Label Y)" that also takes "Labels (Y)" as input. The output from this box goes into "Parameters (theta)" thus completing the loop: "Prediction Function" -> "Output (Y^)" -> "Cost (Output Y^ vs Label Y)" -> "Parameters (theta)" -> "Prediction Function"

2. The image shows a box that takes "I am happy because I am learning NLP" as input and outputs "Postive: 1". The internals of this box contain three boxes labeled "X" -> Train LR -> Classify.

## Reading - Vocabulary and Feature Extraction

1. The image shows a sparse vector of length |V| where the initial values are 1 and the remaining 0's. The vector is a representation of the text "I am happy because I am learning NLP". Below this vector is shown another vector [theta0, theta1, theta2, ..., thetan] (where n = |V|). This second vector is labeled (1) Large training time and (2) Large prediction time.

## Reading - Feature Extraction with Frequencies

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

## Reading - Preprocessing

The image shows a box containing "tun~ing~ GREAT AI model" and the preprocessed tweet `[tun, great, ai, model]`. Two additional mappings are provided for illustration. The first mapping shows "tun" mapped to "tun", "tuned", "tuning" and the second shows "GREAT", "Great", "great" all mapped to "great"

## Reading - Putting it all together

1. The image shows a 2 stage flow in which the input "I am Happy Because i am learning NLP @deeplearning" becomes `[happy, learn, nlp]` after the first stage - Preprocessing stage. In the second stage - Feature Extraction - `[happy, learn, nlp]` becomes `[1, 4, 2]` in which the terms are Bias, Sum positive frequencies and Sum negative frequencies respectively

2. The image shows a matrix X where the i'th row corresponds to the i'th tweet - 
   - 1, X subscript 1 superscript (1), X subscript 2 superscript 1
   - 1, X subscript 1 superscript (2), X subscript 2 superscript 2
   - ...
   - 1, X subscript 1 superscript (m), X subscript 2 superscript m
   
 3. The image shows the following python code:
    ```python
    freqs = build_freqs(tweets, labels) # Build frequencies dictionary
    X = np.zeros((m, 3)) #Initialize matrix X
    for i in range(m): # For every tweet
        p_tweet = process_tweet(tweets[i]) # Process tweet
        X[i,:] = extract_features(p_tweet, freqs) # Extract Features
    ```
    
## Reading - Logistic Regression: Overview 

1. The image shows the formula for the sigmoid function whose input is the i'th x and theta. Its value is 1 over (1 + e raised to (-1 * theta transposed * i'th x). The plot of this function has y values ranging between 0 and 1 and is shaped like an S stretched from the end points such that there is one y corresponding to each x.

2. The image shows i'th x and theta column vectors corresponding to the tweet "@YMourri and @AndrewYNG are tuning a GREAT AI model" or `[tun, ai, great, model]`. The i'th x column vector is `[1, 3476, 245]` and the theta column vector is `[0.00003, 0.0015, -0.00120]`. The corresponding point on the graph is shown at x = 4.92, y =~ 1 implying a positive classification.

## Reading - Logistic Regression: Training

1. The image shows two flows depicting the steps of Logistic Regression. The first flow shows mathematical equations/formulas. The second flow is a mirror image of the first and shows the equivalent text for the math. The flow is - a box labeled theta (Initialize Parameters) -> h = h(X, theta) (classify/predict) -> gradient = Xtranspose(h-y)/m (get gradient) -> theta = theta - alpha * gradient (update) -> J(theta) (get loss). An arrow from the final box of "get loss" goes back to the box labeled "classify/predict". The arrow itself is labeled "until good enough" indicating the repetition cycles of the flow.

2. The image shows a Iteration vs Cost graph. The Cost is close to 1000 at 0'th iteration and declines non-linearly to 0 at iteration 1000.

## Reading - Logistic Regression: Testing

1. The image shows an equation with the left handside containing a column vector `[0.3, 0.8, 0.5 ...]` being checked if >= 0.5. The right hand side contains the resulting column vector with values `[0, 1, 1, ...]` because the corresponding values compare accordingly.

2. The image shows formula for calculating accuracy. It is the average over the m test examples of 1's (correct prediction) and 0's(incorrect prediction)
