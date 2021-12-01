# theyes-product-classification

## Approach

I tried to simulate a taxonomy based approach for this challenge. So I follow a 3-step process:
1. I build a taxonomy of unigrams and bigrams (attributes) which would make the most sense in this scenario. I choose 50 attributes - some of them manually set, others selected based on tf-idf overall. This step can also be done manually by domain experts.
2. Image to Attributes model. This is an un supervised step where I try to predict the attributes given just the image. The reasoning here is that although almost all the images do have a description, the description might not be complete. For e.g. the description for a TOP could just talk about the material but not about other attributes like `collar`, `sleeves`, etc which might be more useful for predicting the category. Note that if an attribute is present in the description, it is very likely an actual attribute of the item. On the other hand, if an attribute is not present in the description, it doesnt imply that it's not an actual attribute, I choose probabilities 1.0 and 0.2 to denote positive and negative classes.
3. Attributes to Category model. This step should be ideally either be done via heuristics or if we have enough items for whom we have category labels, train a model. The input is a concatenated vector of 50-d predicted attribute from image and 50-d attributes from description (total 100-d vector)

## Why are you designing the solution this way? What are the aspects that you considered when designing?

Several reasons for designing the solution this way:
1. Interpretability: Since we are basing the results on the taxonomy, we can easily understand why a particular item was classified a certain way.
2. Search based on attributes: This allows us to look for certain attributes like e.g. items with "sleeve" or "sleeveless".
3. Unsupervised learning: We are leveraging the fact that we have pairs of descriptions and images. In a real world scenario, we could have millions of such rows but without any category labels. By using unsupervised learning here, we don't have to provide labels for the vast majority of items.
4. Easy to plugin domain knowledge. There are a couple of places where we can plugin expert knowledge - creating the list of attributes and also creating heuristics to convert a list of attributes to category.

## What are the cases your solution covers, how are they covered and why are they important?

The description had html content in some of them, so I used beautifulsoup to extract only the relevant text parts. I also used lemmatization to normalize the text content. This part is important since I wanted to limit the overall vocabulary and also to make sure that unigram/bigram term frequency is correct. I also used a version of soft labels to prevent the network from making a large change if it gets a false positive.

## What are the cases your solution does not cover and what are the ways you can extend your current solution for them?

The big drawback while building the current taxonomy was that we don't consider synonymous words. For e.g. I manually added "gold" and "silver" to the attributes. If we use embedding similarity to find other close words, like "platinum" and add them also to the attributes. Another way to do this more organically would be to generate image and sentence embeddings by using a CNN-LSTM or a transformer-transformer encoder decoder structure.

