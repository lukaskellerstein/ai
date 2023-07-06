# Source

https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing#scrollTo=w6YTOisTFDsR

# Steps to run

open poetry shell: `poetry shell`
install dependencies (python): `poetry install`

install AI dependencies manually via pip:

```
pip install --quiet bitsandbytes
pip install --quiet --upgrade transformers
pip install --quiet --upgrade accelerate
pip install --quiet sentencepiece

pip install transformers bitsandbytes accelerate
```

Run it: `python main.py`
