# Neural Structured Learning for BIM
This is our implementation for the paper:

Koeun Lee, Yongsu Yu, Daemok Ha, Bonsang Koo, Kwanhoon Lee (2021). [Neural Structured Learning for BIM](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10564907) 

# Abstract
Building information modeling (BIM) element to industry foundation classes (IFC) entity mappings need to be checked to ensure the semantic integrity of BIM models. Existing studies have demonstrated that machine learning algorithms trained on geometric features are able to classify BIM elements, thereby enabling the checking of these mappings. However, reliance on geometry is limited, especially for elements with similar geometric features. This study investigated the employment of relational data between elements, with the assumption that such additions provide higher classification performance. Neural structured learning, a novel approach for combining structured graph data as features to machine learning input, was used to realize the experiment. Results demonstrated that a significant improvement was attained when trained and tested on eight BIM element types with their relational semantics explicitly represented.

# Requirements
1. Python `3.5 ~ 3.7`
2. tensorflow-gpu `1.13.0`
3. sklearn `0.20.0`
4. neural-structured-learning `1.1.0`
</br>

# Installation
```bash 
# to install NSL package,
pip3 install neural-structured-learning
```

# Example to run the codes
```bash
# to run NSL model for BIM
python3 nsl.py

# to run ADV model for BIM
python3 adv.py

# to print a graph of the prediction accuracy of the NSL or ADV model
python3 plots.py
```

