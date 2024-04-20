import re

# match decoration inside words
in_word_re = re.compile(r"[\u0600-\u0603\u0610-\u061a\u0640\u064b-\u065f\u06fa-\u06ff\u06ea-\u06ef\u06df-\u06e8\u06d4"
                        r"\u06d5\u06bf-\u06cb]")

ARABIC_LETTERS = [
    'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س',
    'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', ' '
]

NON_ARABIC_LETTER = re.compile('[^' + ''.join(ARABIC_LETTERS + [' ']) + ']')

REPEATED_CHAR_PATTERN = "([ابتةثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئ])\1{2,}"
REPEATED_CHAR = re.compile(r"([ابتةثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئ])\1{2,}")
