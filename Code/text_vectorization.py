import yaml
import os
import sys
import pyarabic.araby as ar
import tensorflow as tf

sys.path.append(os.path.abspath('.'))
from Code.config import Config
from Code.regex_pattern import NON_ARABIC_LETTER, REPEATED_CHAR_PATTERN

params = yaml.safe_load(open(Config.PARAMS_PATH))['train']

# standardization function for TextVectorization
@tf.keras.utils.register_keras_serializable()
def standardize_ar_text(text: str, strip_shadda=True):
    text = tf.strings.regex_replace(text, ar.TATWEEL, '')
    text = tf.strings.regex_replace(text, ar.HARAKAT_PATTERN.pattern, '')
    if strip_shadda:
        text = tf.strings.regex_replace(text, ar.SHADDA, '')
    text = tf.strings.regex_replace(text, REPEATED_CHAR_PATTERN, '\1')
    text = tf.strings.regex_replace(text, NON_ARABIC_LETTER.pattern, '')
    text = tf.strings.regex_replace(text, r' +', ' ')
    return text


@tf.keras.utils.register_keras_serializable()
def split_ar_text(text: str):
    text = tf.strings.regex_replace(text, '[+\s]', ' ')
    return tf.strings.split(text)


# text = 'مش عارف ب+خصوص سافيـــــــــتش لكن أنا سعيد ل+كون خيمينيز خارج تشكيل+ة أتلتيكو ال+أساسسسسسسسسسسي+ة حسب ال+توقع+ات'
# tensor = split_ar_text(text)

# text = 'ذهب إلى الحديقة ولعب بالكرة'
# tensor = standardize_ar_text(text)
# result = tensor.numpy().decode('utf-8')
# print(result)
# import re
# print(print(re.sub(STOPWORD_PATTERN, '', text)))
