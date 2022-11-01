import spacy
import crosslingual_coreference 
text = "Benzima is the best football player in 2022, he actually deserves it!"
DEVICE = 0 # Number of the GPU, -1 if want to use CPU 
# Add coreference resolution model
coref = spacy.load(
            'en_core_web_sm', 
             disable=['ner', 
                      'tagger', 
                      'parser', 
                      'attribute_ruler',  
                      'lemmatizer'])
coref.add_pipe(
"xx_coref", 
config={"chunk_size": 2500, 
        "chunk_overlap": 2, 
        "device": DEVICE})
print(coref(text)._.resolved_text)
#OUTPUTS, Benzima is the best football player in 2022, Benzima actually deserves it!