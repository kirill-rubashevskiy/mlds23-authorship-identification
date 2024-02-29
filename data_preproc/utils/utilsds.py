import re
from pymystem3 import Mystem

# расшифровка меток
label2name = {
    0: 'А. Пушкин',
    1: 'Д. Мамин-Сибиряк',
    2: 'И. Тургенев',
    3: 'А. Чехов',
    4: 'Н. Гоголь',
    5: 'И. Бунин',
    6: 'А. Куприн',
    7: 'А. Платонов',
    8: 'В. Гаршин',
    9: 'Ф. Достоевский'
}

def clean_sentence(sentence):
    """
    заменяем все некириллические символы и не знаки препинания на пробелы

    >>> clean_sentence("как-то - 'рано' 'Marta'? пела: лЕСОМ, * &нифига(она) не ела")
    "как-то - 'рано' ''? пела: лЕСОМ,  нифига(она) не ела"
    """
    sentence = re.sub(r"[^а-яА-ЯёЁ \-\"!'(),.:;?]", '', sentence) 
    return sentence


def make_punkt(sentence):
    """
    заменяем знаки препинания на их кодовые обозначения

    >>> make_punkt(', двадцать-два, - номер, помереть: не сейчас!')
    ' CM  двадцать-два CM  DSH номер CM  помереть CL  не сейчас EXCL '
    """
    repl = [('...', ' MP '), ('..',' MP '), ('.',' PNT '), (',',' CM '), ('?',' QST '), ('!',' EXCL '), 
            (':',' CL '), (';',' SMC ')]
    for p, r in repl:
        sentence = sentence.replace(p,r)
    sentence = re.sub(r"\s?-\s|\s-\s?", ' DSH ', sentence) # не трогать тире в слове (как-то)
        
    return sentence

def make_grams(sentence):
    """
    заменяет слова в тексте на соответствующие им лексические кодировщики (часть речи, падеж и тп)

    >>> make_grams(' CM  двадцать-два CM  DSH номер CM  помереть CL  не сейчас EXCL ')
    'CM NUM=(вин|им) NUM=(вин,муж,неод|им,муж|вин,сред|им,сред) CM DSH S,муж,неод=(вин,ед|им,ед) CM V,нп=инф,сов CL PART= ADV= EXCL'
    """

    mystem_analyzer = Mystem()
    morph = mystem_analyzer.analyze(sentence)
    
    ret = []
    for lex in morph:
        if lex['text'] in ['MP', 'PNT' , 'CM', 'QST', 'EXCL', 'CL', 'SMC', 'DSH']:
            ret.append(lex['text'])
            continue
        
        try:
            if 'analysis' in lex.keys() and 'gr' in lex['analysis'][0].keys():
                ret.append(lex['analysis'][0]['gr'])
        except:
            # встретил что-то непотребное в стиле ру-ру-ру
            pass
    return ' '.join(ret)


def make_grams_brief(sentence):
    """
    заменяет слова в тексте на соответствующие им лексические кодировщики
    но уже в сокращенном варианте

    >>> make_grams_brief(' CM  двадцать-два CM  DSH номер CM  помереть CL  не сейчас EXCL ')
    'CM NUM NUM CM DSH S,муж,неод CM V,нп CL PART ADV EXCL'
    """

    mystem_analyzer = Mystem()
    morph = mystem_analyzer.analyze(sentence)
    
    ret = []
    for lex in morph:
        if lex['text'] in ['MP', 'PNT' , 'CM', 'QST', 'EXCL', 'CL', 'SMC', 'DSH']:
            ret.append(lex['text'])
            continue
        
        try:
            if 'analysis' in lex.keys() and 'gr' in lex['analysis'][0].keys():
                ret.append(lex['analysis'][0]['gr'].split('=')[0])
        except:
            # встретил что-то непотребное в стиле ру-ру-ру
            pass
    return ' '.join(ret)


def prepare_text(Text_corp, full=True):
    """
    итоговая предобработка для наших моделей
    >>> prepare_text(["Мама. Мыла раму папе"], full=True)
    ['S,жен,од=им,ед PNT V,несов,пе=прош,ед,изъяв,жен S,жен,неод=вин,ед S,муж,од=(пр,ед|дат,ед)']

    >>> prepare_text(["Мама. Мыла раму папе"], full=False)
    ['S,жен,од PNT V,несов,пе S,жен,неод S,муж,од']
    """

    res = []
    for text in Text_corp:
        text = clean_sentence(text)
        text = make_punkt(text)
        if full:
            text = make_grams(text)
        else:
            text = make_grams_brief(text)
        res.append(text)
    return res






