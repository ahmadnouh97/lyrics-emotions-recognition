import sys
import os
import string
import pyarabic.araby as ar
from nltk.stem.isri import ISRIStemmer
import arabicstopwords.arabicstopwords as stp

sys.path.append(os.path.abspath('.'))
from Code.regex_pattern import *

# match decoration inside words
in_word_re = re.compile(r"[\u0600-\u0603\u0610-\u061a\u0640\u064b-\u065f\u06fa-\u06ff\u06ea-\u06ef\u06df-\u06e8\u06d4"
                        r"\u06d5\u06bf-\u06cb]")

ARABIC_LETTERS = [
    'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س',
    'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', ' '
]

NON_ARABIC_LETTER = re.compile('[^' + ''.join(ARABIC_LETTERS + [' ']) + ']')

ARABIC_PUNCTUATIONS = '''`÷×؛<>_()*&^%][،/:"؟.,'{}~¦+|!”…“–»«•'''
ENGLISH_PUNCTUATIONS = string.punctuation
ENGLISH_NUMBER = '0123456789'
PUNCTUATIONS = ARABIC_PUNCTUATIONS + ENGLISH_PUNCTUATIONS + ENGLISH_NUMBER

stop_words = ['(' + word + ')' for word in list(stp.stopwords_list())]
STOPWORD_PATTERN = r'\b(' + r'|'.join(list(stp.stopwords_list())) + r')\b\s*'


def normalize_arabic(text):
    text = re.sub("إ", "ا", text)
    text = re.sub("أ", "ا", text)
    text = re.sub("آ", "ا", text)
    text = re.sub("ا", "ا", text)
    # text = re.sub("ى", "ي", text)
    # text = re.sub("ؤ", "ء", text)
    # text = re.sub("ئ", "ء", text)
    # text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


def remove_punctuations(text):
    for c in PUNCTUATIONS:
        text = text.replace(c, " ")
    return text


def rooting(text):
    stems = list()
    for word in text.split():
        stemmer = ISRIStemmer()
        stems.append(stemmer.stem(word))
    return ' '.join(stems)


def process_text(text):
    # trim text
    text = text.strip()
    # remove stopwords
    # text = re.sub(STOPWORD_PATTERN, '', text)
    # strip decoration inside words
    text = in_word_re.sub('', text)
    # strip shadda
    text = ar.strip_shadda(text)
    # strip tashkeel
    text = ar.strip_tashkeel(text)
    # strip tatweel
    text = ar.strip_tatweel(text)
    # remove punctuations
    # text = remove_punctuations(text)
    # normalize arabic letters
    # text = normalize_arabic(text)
    # remove repeating chars
    # text = remove_repeating_char(text)
    # remove non-arabic letters
    text = re.sub(NON_ARABIC_LETTER, ' ', text)
    # rooting text
    # text = rooting(text)
    # strip multiple spaces with single one
    text = ' '.join(text.split())
    return text

# def process_text(text):
#     # trim text
#     text = text.strip()
#     # remove stop words
#     text = re.sub(STOPWORD_PATTERN, '', text)
#     # strip decoration inside words
#     text = in_word_re.sub('', text)
#     # strip shadda
#     text = ar.strip_shadda(text)
#     # strip tashkeel
#     text = ar.strip_tashkeel(text)
#     # strip tatweel
#     text = ar.strip_tatweel(text)
#     text = re.sub(REPEATED_CHAR, r'\1', text)
#     # strip non arabic letter
#     text = re.sub(NON_ARABIC_LETTER, ' ', text)
#     # strip multiple spaces with single one
#     text = ' '.join(text.split())
#     # segment text
#     text = segmenter.segment(text)
#     return text
