# jive_jackstraw: Jackstraw significance testing for JIVE in Python

Python implementation of jackstraw significance testing for AJIVE loadings ([arXiv](https://arxiv.org/abs/2109.12272)).

This package complements Iain Carmichael's
Python implementation of AJIVE ([py_jive](https://github.com/idc9/py_jive)), and
provides significance testing of the common loadings coming out of that software.

To install this package from Github:
```
git clone https://github.com/thomaskeefe/jive_jackstraw.git
python setup.py install
```

Usage (with numpy arrays `datablock` (n x d) and common normalized scores `cns` (n x joint_rank)):
```python
from jive_jackstraw import JIVEJackstraw

js = JIVEJackstraw()
js.fit(datablock, cns, alpha=.05, bonferroni=True)
print(js.results[0]['significant'])
```

Usage with `py_jive`, and pandas DataFrames `datablock1` and `datablock2`
```python
from py_jive.AJIVE import AJIVE
from jive_jackstraw import JIVEJackstraw

ajive = AJIVE(init_signal_ranks{'block1': 5, 'block2': 10})
ajive.fit({'block1': datablock1, 'block2': datablock2})
common_normalized_scores = ajive.common.scores()

js = JIVEJackstraw()
js.fit(datablock.values, cns.values, alpha=.05, bonferroni=True)
print(js.results[0]['significant'])
```


To run tests:
```
cd tests
python -m unittest tests.py
```

## References
Yang X, Hoadley KA, Hannig J, Marron JS (2021). Statistical inference for data integration. Submitted at Journal of Multivariate Analysis. https://arxiv.org/abs/2109.12272

Chung, N.C. and Storey, J.D. (2015) Statistical significance of variables driving systematic variation in high-dimensional data. Bioinformatics, 31(4): 545-554 http://bioinformatics.oxfordjournals.org/content/31/4/545

Feng, Q; Jiang, M; Hannig, J; Marron, JS (2018) Angle-based joint and individual variation explained. Journal of Multivariate Analysis, 166: 241-265. https://arxiv.org/pdf/1704.02060.pdf
