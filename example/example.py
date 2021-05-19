import qgg_scorer
from qgg_scorer import SimilarityScorer,CoverageScorer,PPLScorer
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
