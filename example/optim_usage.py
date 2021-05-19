from qgg_utils.optim import GAOptimizer,GreedyOptimizer,RandomOptimizer,FirstNOptimizer
import os,pathlib
import random

if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.absolute()
    context = open(os.path.join(current_dir,'example_question_group/context.txt')).read()
    candidate_quesitons = open(os.path.join(current_dir,'example_question_group/hyps.txt')).read().split('\n')
    
    candidate_quesitons = candidate_quesitons*3
    random.shuffle(candidate_quesitons)

    #
    candicate_pool_size = len(candidate_quesitons)
    target_question_qroup_size = 5

    #
    print("GAOptimizer")
    ga_optim = GAOptimizer(candicate_pool_size,target_question_qroup_size)
    print(ga_optim.optimize(candidate_quesitons,context),end="\n\n")
    
    #
    print('GreedyOptimizer')
    greedy_optim = GreedyOptimizer(candicate_pool_size,target_question_qroup_size)
    print(greedy_optim.optimize(candidate_quesitons,context),end="\n\n")
    
    #
    print('RandomOptimizer')
    rand_optim = RandomOptimizer(candicate_pool_size,target_question_qroup_size)
    print(rand_optim.optimize(candidate_quesitons,context),end="\n\n")
    
    #
    print('FirstNOptimizer')
    firstn_optim = FirstNOptimizer(candicate_pool_size,target_question_qroup_size)
    print(firstn_optim.optimize(candidate_quesitons,context),end="\n\n")
