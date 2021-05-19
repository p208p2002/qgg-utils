# qgg-scorer
## Setup
### Stanza
```sh
pip install stanza
```
```python 
import stanza
stanza.download('en')
```
### nlg-eval
```sh
pip install git+git+https://github.com/voidful/nlg-eval.git@master
export LC_ALL=C.UTF-8&&export LANG=C.UTF-8&&nlg-eval --setup
```