## Dataset
* [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
* Source: using torchtext imdb implementation
* Downsampled to 2500 examples - training and validation 1250 each.

## Related Works
### Adversarial Learning
* [Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.](https://arxiv.org/pdf/1412.6572.pdf)
* [Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial examples in the physical world. arXiv preprint arXiv:1607.02533, 2016.](https://arxiv.org/pdf/1607.02533.pdf)
* [Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z Berkay Celik, and Ananthram Swami. The limitations of deep learning in adversarial settings. In Security and Privacy (EuroS&P), 2016 IEEE European Symposium on, pp. 372–387. IEEE, 2016.](https://arxiv.org/pdf/1511.07528.pdf)
* [Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In ICLR 2014, 2014.](https://arxiv.org/pdf/1312.6199.pdf)

### Adversarial Learning for NLP
* [Adversarial Examples for Natural Language Classification Problems](https://openreview.net/forum?id=r1QZ3zbAZ)
* [HotFlip: White-Box Adversarial Examples for Text Classification](http://aclweb.org/anthology/P18-2006)
* [Adversarial Examples for Evaluating Reading Comprehension Systems](https://www.aclweb.org/anthology/D17-1215)
* [Papernot, N., McDaniel, P., Swami, A., & Harang, R. (2016, November). Crafting adversarial input sequences for recurrent neural networks. In Military Communications Conference, MILCOM 2016-2016 IEEE (pp. 49-54). IEEE.](https://arxiv.org/pdf/1604.08275.pdf)
* [Deceiving Google’s Perspective API Built for
Detecting Toxic Comments](https://arxiv.org/pdf/1702.08138.pdf)
* [Towards Crafting Text Adversarial Samples](https://arxiv.org/pdf/1707.02812.pdf)
* [Deep Text Classification Can be Fooled](https://arxiv.org/pdf/1704.08006.pdf)
* [Generating Natural Language Adversarial Examples](https://arxiv.org/pdf/1804.07998.pdf)

### Model for Sentiment Analysis
* [Yoon Kim. Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014.](http://www.aclweb.org/anthology/D14-1181)
* [Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. ¨ Neural computation, 9(8):1735–1780, 1997.](https://www.bioinf.jku.at/publications/older/2604.pdf)

## Experiment
* Naive Bayes
* CNN
* Bidirectional RNN
* Bidirectional LSTM

## Evaluation
### Classifier Evaluation
* Accuracy, precision, recall
* Loss function analysis
  * Increment of loss
  * Change of classifier confidence (probability) (i.e. sigmoid output)

### Adversarial Example Evaluation
* Imperceptibility analysis
  * Human evaluation
  * Quantitative measurement: thought vectors
* Sentence error analysis
  * Syntactic error: word replacement incur grammatical error
  * Semantic error: meaning of the sentence change after word replacement
  * Counterfactual error: some fact in the sentence is incorrect after word replacement

## TODO
* Research question, data collection, related work (adversarial learning), Experiment (@Erica)
* Related work (adversarial learning for NLP x5), Experiment (@Alicia)
* Related work (adversarial learning for NLP x2), Evaluation, Experiment, Future work (@Tobey)
