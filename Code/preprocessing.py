import sys
import os
import pyarabic.araby as ar
from farasa.segmenter import FarasaSegmenter
import arabicstopwords.arabicstopwords as stp

sys.path.append(os.path.abspath('.'))
from Code.regex_pattern import *

segmenter = FarasaSegmenter(interactive=True)
stop_words = ['(' + word + ')' for word in list(stp.stopwords_list())]
STOPWORD_PATTERN = r'\b(' + r'|'.join(list(stp.stopwords_list())) + r')\b\s*'


def process_text(text):
    # trim text
    text = text.strip()
    # remove stop words
    text = re.sub(STOPWORD_PATTERN, '', text)
    # strip decoration inside words
    text = in_word_re.sub('', text)
    # strip shadda
    text = ar.strip_shadda(text)
    # strip tashkeel
    text = ar.strip_tashkeel(text)
    # strip tatweel
    text = ar.strip_tatweel(text)
    text = re.sub(REPEATED_CHAR, r'\1', text)
    # strip non arabic letter
    text = re.sub(NON_ARABIC_LETTER, ' ', text)
    # strip multiple spaces with single one
    text = ' '.join(text.split())
    # segment text
    text = segmenter.segment(text)
    return text
