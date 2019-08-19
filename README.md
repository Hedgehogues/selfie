https://arxiv.org/pdf/1906.02940.pdf

This implementation has fake attention. This implementation cannot work without pretrained resnet. I think autors of the paper has error or inaccuracy into paper.

My attempts to make a decoder encoder. But they were unsuccessful. Accuracy for 3x patches == 0.33 (i.e. random guessing). So it goes. Either I make a mistake somewhere, or a problem with the article.

A few observations. If ResNet is pre-trained, then everything works. If not, then of course nothing works. From the very beginning, I was confused by the designs of the scalar product that the authors propose in the article, as well as the classifier, which should distinguish:

(v, h0) + h0 ^ 2; (v, h1); (v, h2); ...; (v, hk)

scalar product to which the square of a patch is tidied up. I cannot prove it formally, but this construction does not look convincing.
