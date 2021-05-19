from nlgeval import NLGEval
from collections import defaultdict
import os
import re
import stanza
from loguru import logger
import copy
from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pathlib

current_dir = pathlib.Path(__file__).parent.absolute()

def step_len(func):
    def wrapper(*args,**kwargs):
        self = args[0]
        self.len += 1
        return func(*args,**kwargs)
    return wrapper

class Scorer():
    def __init__(self,preprocess=True,metrics_to_omit=["CIDEr","METEOR"]):
        self.preprocess = preprocess
        self.nlgeval = NLGEval(no_glove=True,no_skipthoughts=True,metrics_to_omit=metrics_to_omit)
        self.score = defaultdict(lambda : 0.0)
        self.len = 0
        if self.preprocess:
            self.nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, verbose=False)
        
        #
        self.stop_words_sign = open(os.path.join(current_dir,'stopwords-sign.txt'),'r',encoding='utf-8').read().split()
        self.stop_words_sign_rule = "|".join([re.escape(sign) for sign in self.stop_words_sign])
    
    @lru_cache(maxsize=200)
    def _preprocess(self,raw_sentence):
        result = self.nlp(raw_sentence.replace("\n\n",""))
        tokens = []
        try:
            for token in result.sentences[0].tokens:
                tokens.append(token.text.lower())
            tokenize_sentence = ' '.join(tokens)
            tokenize_sentence = re.sub(self.stop_words_sign_rule,"",tokenize_sentence)                
        except Exception as e:
            logger.warning(e)
            logger.warning(f'preprocess fail, return "" raw_sentence:{raw_sentence} result:{result}')
            return ""
        return tokenize_sentence
    
    def clean(self):
        self.score = defaultdict(lambda : 0.0)
        self.len = 0
    
    @step_len
    def add(*args,**kwargs):
        assert False,'no implement error'

    def compute(self,save_report_dir=None,save_file_name='score.txt',return_score=False):
        # 
        out_score = {}
        
        if save_report_dir is not None:
            os.makedirs(save_report_dir,exist_ok=True)
            save_score_report_path = os.path.join(save_report_dir,save_file_name)
            score_f = open(save_score_report_path,'w',encoding='utf-8')
        for score_key in self.score.keys():
            _score = self.score[score_key]/self.len
            out_score[score_key] = _score
            if save_report_dir is not None:
                score_f.write("%s\t%3.5f\n"%(score_key,_score))
        
        if return_score:
            return out_score
    
class SimilarityScorer(Scorer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ppl_scorer = PPLScorer()

    @step_len
    def add(self,hyp,refs):
        refs = refs[:]
        if self.preprocess:
            hyp = self._preprocess(hyp)
            refs = [self._preprocess(ref) for ref in refs]
        _score = self.nlgeval.compute_individual_metrics(hyp=hyp, ref=refs)
        ppl_score = self.ppl_scorer._compute_ppl(hyp)
        scaled_ppl = self.ppl_scorer._compute_scaled_ppl(hyp)
        _score['ppl'] = ppl_score
        _score['scaled_ppl'] = scaled_ppl
        for score_key in _score.keys():
            self.score[score_key] += _score[score_key]

class CoverageScorer(Scorer):
    def __init__(self,preprocess=True):
        super().__init__(preprocess=preprocess)
        self.stop_words_en = open(os.path.join(current_dir,'stopwords-en.txt'),'r',encoding='utf-8')
        self.stop_words_sign = open(os.path.join(current_dir,'stopwords-sign.txt'),'r',encoding='utf-8')
        self.stop_words = self.stop_words_en.read().split() + self.stop_words_sign.read().split()
        
        # some sign used to split context to sentence, remove them from `stopwords-sign`
        self.stop_words_sign = open(os.path.join(current_dir,'stopwords-sign.txt'),'r',encoding='utf-8').read().split()
        self.stop_words_sign.remove(',')
        self.stop_words_sign.remove('.')
        self.stop_words_sign.remove('!')
        self.stop_words_sign.remove('?')
        self.stop_words_sign_rule = "|".join([re.escape(sign) for sign in self.stop_words_sign])

    def _compute_coverage_score(self,sents:list,article:str):
        sent = ' '.join(sents)
        sent_list = re.split(r",|\.|\!|\?",sent)
        for sent in sent_list[:]:
            if sent == '': sent_list.remove(sent)
        
        # get sents keywords
        keyword_list = []
        for sent in sent_list[:]:
            sent = sent.lower()
            word_list = sent.split()
            for word in word_list:
                if word not in self.stop_words:
                    keyword_list.append(word)
        
        # process acticle into words and compute coverage
        article_sent_list = re.split(r",|\.|\!|\?",article)
        
        count_article_sent = len(article_sent_list)
        if count_article_sent == 0:
            return 0.0
        
        count_coverage = 0
        for article_sent in article_sent_list:
            article_sent = article_sent.lower().split()
            for keyword in keyword_list:
                if keyword in article_sent:
                    count_coverage += 1
                    break 
        return count_coverage/count_article_sent
    
    @step_len
    def add(self,sents:list,article:str):
        sents = sents[:]
        if self.preprocess:
            sents = [self._preprocess(sent) for sent in sents]
            article = self._preprocess(article)
        coverage_score = self._compute_coverage_score(sents,article)
        self.score['keyword_coverage'] += coverage_score

class PPLScorer(Scorer):
    def __init__(self, model_id = 'gpt2', device = 'cpu', stride=512, max_length=512):
        if '_ppl_model' not in globals():
            global _ppl_model
            _ppl_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.model = _ppl_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.stride = stride
        self.max_length = max_length
        self.device = device
        
        #
        self.score = defaultdict(lambda : 0.0)
        self.len = 0
    
    @step_len
    def add(self,sentence):
        self.score['ppl'] += self._compute_ppl(sentence)
        
    def _compute_scaled_ppl(self,sentence,alpha=0.2):
        # https://www.desmos.com/calculator/scqyyq0ody
        avg_ll = self._compute_avg_log_likelihood(sentence)
        return torch.exp(-avg_ll*alpha)
    
    def _compute_ppl(self,sentence):
        # https://huggingface.co/transformers/perplexity.html
        avg_ll = self._compute_avg_log_likelihood(sentence)
        return torch.exp(avg_ll)
        
    
    @lru_cache(maxsize=200)
    def _compute_avg_log_likelihood(self,sentence):
        stride = self.stride
        max_length = self.max_length
        encodings = self.tokenizer(sentence, return_tensors='pt')
        model = self.model

        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i    # may be different from stride on last loop
            input_ids = encodings.input_ids[:,begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:,:-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)                
                log_likelihood = outputs[0] * trg_len

            lls.append(log_likelihood)
        return torch.stack(lls).sum() / end_loc