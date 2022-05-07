# Week 2

## Reading - Probability and Bayes' Rule

You learned about probabilities and Bayes' rule. 

Image containing a corpus of tweets - represented as a 4x5 grid with 13 green boxes representing Positive and 7 orange representing Negative tweets.

A -> Positive tweet

P(A) = N(pos)/N = 13/20 = 0.65

P(Negative) = 1 - P(Positive) = 0.35

To calculate a probability of a certain event happening, you take the count of that specific event and you divide by the sum of all events. Furthermore, the sum of all probabilities has to equal 1. 

Image containing a Corpus of tweets represented as a 4x5 grid with 10 green boxes representing Positive, 3 presenting Positive AND happy, 1 representing negative and happy. Also is shown, a Venn diagram illustrating the embedded equation P(A ∩ B) = P(A, B) = 3/20 = 0.15. The Venn diagram contains a rectangle representing the Corpus. Within it are two partially overlapping circles - blue the smaller one representing "happy" and green i.e. the larger one representing "Positive". The partially overlapping area represents the intersection "happy" ∩ Positive.

To compute the probability of 2 events happening, like "happy" and "positive" in the picture above, you would be looking at the intersection, or overlap of events. In this case red and blue boxes overlap in 3 boxes. So the answer is 3/20.

## Reading - Bayes' Rule

Conditional probabilities help us reduce the sample search space. For example given a specific event already happened, i.e. we know the word is happy:

A Venn diagram diagram illustrating the embedded equation P(Positive|"happy") = P(Positive ∩ "happy") / P("happy"). The Venn diagram contains a rectangle representing the Corpus. Within it are two partially overlapping circles - blue the smaller one representing "happy" and green i.e. the larger one representing "Positive". The partially overlapping area represents the intersection "happy" ∩ Positive.

Then you would only search in the blue circle above. The numerator will be the red part and the denominator will be the blue part. This leads us to conclude the following: 

P(Positive|"happy") = P(Positive ∩ "happy")/P("happy")

P("happy"|Positive) = P("happy" ∩ Positive)/P(Positive)

Substituting the numerator in the right hand side of the first equation, you get the following: 

P(Positive|"happy") = P("happy"|Positive) x P(Positive)/P("happy")

Note that we multiplied by P(positive) to make sure we don't change anything.  That concludes Bayes Rule which is defined as 

P(X∣Y)=P(Y∣X)P(X)/P(Y)

## Reading - Naive Bayes Introduction

To build a classifier, we will first start by creating conditional probabilities given the following table:

word | Pos | Neg
-- | -- | --
I | 3 | 3
am | 3 | 3
happy | 2 | 1
because | 1 | 0
learning | 1 | 1
NLP | 1 | 1
sad | 1 | 2
not | 1 | 2
N | 13 | 13

Assuming these Positive tweets:
- I am happy because I am learning NLP
- I am happy, not sad

and these Negative tweets:
- I am sad, I am not learning NLP
- I am sad, not happy

This allows us compute the following table of probabilities:

word | Pos | Neg
-- | -- | --
I | 0.24 | 0.25
am | 0.24 | 0.25
happy | 0.15 | 0.08
because | 0.08 | 0
learning | 0.08 | 0.08
NLP | 0.08 | 0.08
sad | 0.08 | 0.17
not | 0.08 | 0.17

Once you have the probabilities, you can compute the likelihood score as follows

Tweet: I am happy today; I am learning

word | Pos | Neg
-- | -- | --
I | 0.20 | 0.20
am | 0.20 | 0.20 
happy | 0.14 | 0.10
because | 0.10 | 0.05
learning | 0.10 | 0.10
NLP | 0.10 | 0.10
sad | 0.10 | 0.15
not | 0.10 | 0.15

Product of P(w|pos)/P(w|neg) for all the words in the tweet = (0.20/0.20) * (0.20/0.20) * (0.14/0.10) * (0.20/0.20) * (0.20/0.20) * (0.10/0.10) = 0.14 / 0.10 = 1.4 > 1

A score greater than 1 indicates that the class is positive, otherwise it is negative.

## Reading - Log Likelihood, Part 1

Log Likelihood, Part 1

To compute the log likelihood, we need to get the ratios and use them to compute a score that will allow us to decide whether a tweet is positive or negative. The higher the ratio, the more positive the word is:

<-- Negative (close to 0) -- Neutral (close to 1) -- Positive (towards infinity) -->

word | Pos | Neg | ratio
-- | -- | -- | --
I | 0.19 | 0.20
am | 0.19 | 0.20
happy | 0.14 | 0.10
because | 0.10 | 0.05
learning | 0.10 | 0.10
NLP | 0.10 | 0.10
sad | 0.10 | 0.15
not | 0.10 | 0.15

ratio(w_i) = P(w_i|Pos) / P(w_i|Neg) ~ (freq(w_i, 1)+1)/(freq(w_i, 0)+1)

To do inference, you can compute the following: 

(P(pos)/P(neg)) times the product of P(w_i|pos)/P(w_i|neg) for i ranging from 1 to m is greater than 1

As m gets larger, we can get numerical flow issues, so we introduce the log, which gives you the following equation: 

log of the previous formula = log (P(pos)/P(neg)) + summation of log(P(w_i|pos)/P(w_i|neg)) for i ranging from 1 to m

The first component is called the log prior and the second component is the log likelihood. We further introduce λ\lambda λ as follows: 

doc: I am happy because I am learning.

lambda(w) = log (P(w/pos)/P(w/neg)) 

word | Pos | Neg | lambda
-- | -- | -- | --
1 | 0.05 | 0.05 | 0
am | 0.04 | 0.04 | 0
happy | 0.09 | 0.01 | lambda(happy) = log(0.09/0.01) ~ 2.2
because | 0.01 | 0.01
learning | 0.03 | 0.01
NLP | 0.02 | 0.02
sad | 0.01 | 0.09
not | 0.02 | 0.03

Having the λ\lambdaλ dictionary will help a lot when doing inference. 

## Reading - Log Likelihood Part 2

Once you computed the λ\lambdaλ dictionary, it becomes straightforward to do inference: 

doc: I am happy because I am learning

word | Pos | Neg | lambda
-- | -- | -- | --
I | 0.05 | 0.05 | 0
am | 0.04 | 0.04 | 0
happy | 0.09 | 0.01 | 2.2
because | 0.01 | 0.01 | 0
learning | 0.03 | 0.01 | 1.1
NLP | 0.02 | 0.02 | 0
sad | 0.01 | 0.09 | -2.2
not | 0.02 | 0.03 | -0.4

sum of log(P(w_i|pos)/P(w_i|neg)) over all i ranging from 1 to m = sum of lambda(w_i) over all i ranging from 1 to m

log likelihood = 0 + 0 + 2.2 + 0 + 0 + 0 + 1.1 = 3.3

As you can see above, since 3.3>0, we will classify the document to be positive. If we got a negative number we would have classified it to the negative class. 

## Reading - Training naive Bayes

To train your naïve Bayes classifier, you have to perform the following steps:
1) Get or annotate a dataset with positive and negative tweets
2) Preprocess the tweets: process_tweet(tweet) ➞ [w1, w2, w3, ...]:

    - Lowercase
    - Remove punctuation, urls, names
    - Remove stop words
    - Stemming
    - Tokenize sentences

3) Compute freq(w, class):
    
    Positive tweets:
    
        - [happi, because, learn, NLP]
        - [happi, not, sad]
    
    Negative tweets
    
        - [sad, not, learn, NLP]
        - [sad, not, happi]
   
   a word count of the above gives the following table:
   word | Pos | Neg
   -- | -- | --
   happi | 2 | 1
   because | 1 | 0
   learn | 1 | 1
   NLP | 1 | 1
   sad | 1 | 2
   not | 1 | 2
   N | 7 | 7

4) Get P(w|pos),P(w|neg)

    You can use the table above to compute the probabilities.

5) Get lambda(w)

    λ(w)=log⁡P(w∣pos)P(w∣neg) 

6) Compute logprior=log⁡(P(pos)/P(neg))

    logprior = log (D(pos)/D(neg)), where D(pos) and D(neg) correspond to the number of positive and negative documents respectively

## Reading - Testing naive Bayes

word | lambda
-- | --
I | -0.01
the | -0.01
happi | 0.63
because | 0.01
pass | 0.5
NLP | 0
sad | -0.75
not | -0.75

- log-likelihood dictionary lambda(w) = log (P(w|pos)/P(w|neg))
- logprior = log (D(pos)/D(neg)) = 0
- Tweet: I, pass, the, NLP, interview
    
    score = -0.01 + 0.5 - 0.01 + 0 + logprior = 0.48
    
    pred = score > 0

The example above shows how you can make a prediction given your lambda dictionary. In this example the logprior is 0 because we have the same amount of positive and negative documents (i.e. log⁡1 = 0 )

## Reading - Naive Bayes Assumptions

Naïve Bayes makes the independence assumption and is affected by the word frequencies in the corpus. For example, if you had the following:

1. Image of Sahara desert labeled "It is sunny and hot in the Sahara desert"
2. Image labeled "It's always cold and snowy in _____"

In the first image, you can see the word sunny and hot tend to depend on each other and are correlated to a certain extent with the word "desert". Naive Bayes assumes independence throughout. Furthermore, if you were to fill in the sentence on the right, this naive model will assign equal weight to the words "spring, summer, fall, winter". 

On Twitter, there are usually more positive tweets than negative ones. However, some "clean" datasets you may find are artificially balanced to have to the same amount of positive and negative tweets. Just keep in mind, that in the real world, the data could be much noisier. 

## Reading - Error Analysis

There are several mistakes that could cause you to misclassify an example or a tweet. For example, 

- Removing punctuation
- Removing words
    - - Tweet: This is not good, because your attitude is not even close to being nice. 
      - processed_tweet: [good, attitude, close, nice]        
    - - Tweet: My beloved grandmother :(
      - processed_tweet: [belov, grandmoth]
        
- Word order
    - Tweet: I am happy because I did not go. (positive)
    - Tweet: I am not happy because I did go. (negative)
- Adversarial attacks

These include sarcasm, irony, euphemisms.

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
