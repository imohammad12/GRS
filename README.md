# GRS
Code for our paper "[GRS: Combining Generation and Revision in Unsupervised Sentence Simplification](https://aclanthology.org/2022.findings-acl.77)" accepted at Findings of the Association for Computational Linguistics: ACL 2022.




## Setup
```
$ conda env create --file environment.yml
$ python src/main.py
```
You will also need to have a CoreNLP Server running on port 9000. You can download the package from [[here]](https://stanfordnlp.github.io/CoreNLP/download.html)

## Constituent Models
<!-- The models are mentioned below and are available in the hugging face hub. -->
We have used different models in our score function that can be also used independently from the GRS code. The models are available in the hugging face hub. You can access them from [[here]](https://huggingface.co/imohammad12).
<!-- 
### Text Simplicity
This model assigns a simplicity score to a given sentence. Consider that this model is unsupervised.  -->


## Citation
Please cite this paper if you use our code or system output.

```
@inproceedings{dehghan-etal-2022-grs,
    title = "{GRS}: Combining Generation and Revision in Unsupervised Sentence Simplification",
    author = "Dehghan, Mohammad  and
      Kumar, Dhruv  and
      Golab, Lukasz",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.77",
    pages = "949--960",
    abstract = "We propose GRS: an unsupervised approach to sentence simplification that combines text generation and text revision. We start with an iterative framework in which an input sentence is revised using explicit edit operations, and add paraphrasing as a new edit operation. This allows us to combine the advantages of generative and revision-based approaches: paraphrasing captures complex edit operations, and the use of explicit edit operations in an iterative manner provides controllability and interpretability. We demonstrate these advantages of GRS compared to existing methods on the Newsela and ASSET datasets.",
}
```
