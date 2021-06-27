# QGG Utils
## Install
### Install qgg-utils
```sh
pip install -U git+https://github.com/p208p2002/qgg-utils.git
```
### Setup Stanza
```sh
pip install stanza
python -c"import stanza;stanza.download('en')"
```

### Setup nlg-eval
```sh
pip install git+https://github.com/voidful/nlg-eval.git@master
export LC_ALL=C.UTF-8&&export LANG=C.UTF-8&&nlg-eval --setup
```

## Usage
### Scorer
```python
from qgg_utils.scorer import SimilarityScorer,CoverageScorer,PPLScorer
import pathlib
import os

if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.absolute()
    context = open(os.path.join(current_dir,'example_question_group/context.txt')).read()
    hyps = open(os.path.join(current_dir,'example_question_group/hyps.txt')).read().split('\n')
    refs = open(os.path.join(current_dir,'example_question_group/refs.txt')).read().split('\n')

    ref_similarity_scorer = SimilarityScorer()
    self_similarity_scorer = SimilarityScorer()
    coverage_scorer = CoverageScorer()
    ppl_scorer = PPLScorer()

    # compute Ref Similarity
    print('Ref Similarity')
    for hyp in hyps:
        ref_similarity_scorer.add(hyp,refs)
    print(ref_similarity_scorer.compute(),end='\n\n')

    # compute Self Similarity
    print('Self Similarity')
    for hyp in hyps:
        _hyps = hyps[:]
        _hyps.remove(hyp)
        self_similarity_scorer.add(hyp,_hyps)
    print(self_similarity_scorer.compute(),end='\n\n')
    
    # compute Keyword Coverage
    print('Keyword Coverage')
    coverage_scorer.add(hyps,context)
    print(coverage_scorer.compute(),end='\n\n')
        
    # compute PPL
    print('PPL')
    for hyp in hyps:
        ppl_scorer.add(hyp)
    print(ppl_scorer.compute(),end='\n\n')
```

### Group Optim
```python
from qgg_utils.optim import GAOptimizer,GreedyOptimizer,RandomOptimizer,FirstNOptimizer
import os,pathlib
import random

if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.absolute()
    context = open(os.path.join(current_dir,'example_question_group/context.txt')).read()
    candidate_quesitons = open(os.path.join(current_dir,'example_question_group/hyps.txt')).read().split('\n')

    # fill in some duplicate data
    candidate_quesitons = candidate_quesitons*3
    random.shuffle(candidate_quesitons)
    candicate_pool_size = len(candidate_quesitons)
    target_question_qroup_size = 5

    # GAOptimizer
    print("GAOptimizer")
    ga_optim = GAOptimizer(candicate_pool_size,target_question_qroup_size)
    print(ga_optim.optimize(candidate_quesitons,context),end="\n\n")
    
    # GreedyOptimizer
    print('GreedyOptimizer')
    greedy_optim = GreedyOptimizer(candicate_pool_size,target_question_qroup_size)
    print(greedy_optim.optimize(candidate_quesitons,context),end="\n\n")
    
    # RandomOptimizer
    print('RandomOptimizer')
    rand_optim = RandomOptimizer(candicate_pool_size,target_question_qroup_size)
    print(rand_optim.optimize(candidate_quesitons,context),end="\n\n")
    
    # FirstNOptimizer
    print('FirstNOptimizer')
    firstn_optim = FirstNOptimizer(candicate_pool_size,target_question_qroup_size)
    print(firstn_optim.optimize(candidate_quesitons,context),end="\n\n")
```

### Negative Label Loss
```python
import torch
from qgg_utils import NegativeLabelLoss
loss = NegativeLabelLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
```
