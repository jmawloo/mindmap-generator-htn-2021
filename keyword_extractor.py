from rake_nltk import Rake
from collections import OrderedDict

rake_inst = Rake()

f = open("./sampletext3.txt", 'r', encoding="utf8")

text = f.read()
f.close()

rake_inst.extract_keywords_from_text(text) # void

extracted_keys = list(OrderedDict.fromkeys(rake_inst.get_ranked_phrases()))

print(extracted_keys)
