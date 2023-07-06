import csv
from time import sleep
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from enchant.checker import SpellChecker
from enchant.tokenize import EmailFilter, URLFilter, HashtagFilter, MentionFilter, WikiWordFilter

def write_csv_file(data, file_name):
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter = '\n', lineterminator = '\r\n')
        for row in data:
            writer.writerow(row.values())

text_clean = pd.read_csv('class/clean_text_psy.csv')

sgd_clf_psy = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), strip_accents='unicode')),
    ('sgd_clf', SGDClassifier(penalty='elasticnet', class_weight='balanced', random_state=42))
])

X_train, X_valid, y_train, y_valid = train_test_split(text_clean['text_clean'], text_clean['theme'], test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

sgd_clf_psy.fit(X_train, y_train)

predicted_sgd_psy = sgd_clf_psy.predict(X_test)
predicted_sgd_psy_val = sgd_clf_psy.predict(X_valid)

text_clean_ads = pd.read_csv('class/clean_text_ads.csv')

sgd_clf_ads = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 4), strip_accents='unicode')),
    ('sgd_clf', SGDClassifier(penalty='elasticnet', class_weight='balanced', random_state=42))
])

X_train, X_valid, y_train, y_valid = train_test_split(text_clean_ads['text_clean'], text_clean_ads['theme'], test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

sgd_clf_ads.fit(X_train, y_train)

predicted_sgd_ads = sgd_clf_ads.predict(X_test)
predicted_sgd_ads_val = sgd_clf_ads.predict(X_valid)

# checker = SpellChecker("ru_RU", filters=[EmailFilter,URLFilter, HashtagFilter, MentionFilter, WikiWordFilter])

# checker.set_text('«Сложные решения. Как управлять бизнесом, когда нет простых ответов» Автор: Бен Хоровиц 📒 Автор книги – один из опытнейших предпринимателей Кремниевой долины Бен Хоровиц – предлагает эффективные рекомендации по построению и развитию стартапов. При этом ему удается совместить теорию и практику, что повышает ценность книги для всех, независимо от этапа карьеры или жизненного цикла собственного бизнеса. Хоровиц избегает универсальных предписаний. Вместо этого он предлагает лучшие подходы к типичным ситуациям, таким как увольнение сотрудников или продажа бизнеса. Это книга для менеджеров, предпринимателей и тех, кто только собирается открыть свой бизнес. ⚡️')
# print([i.word for i in checker])


def read_posts(file_name):
    data = [[]]
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            string = f.readline()
            while string:
                # id
                # post_text
                # group_link
                # date
                # ners
                # ners_num
                # words_num
                # comments_num
                data.append({
                    'id': int(string),
                    'post_text': f.readline().strip('\n'),
                    'group_link': f.readline().strip('\n'),
                    'date': f.readline().strip('\n'),
                    'ners': f.readline().strip('{\}\n').split(', '),
                    'ners_num': int(f.readline()),
                    'words_num': int(f.readline()),
                    'comments_num': int(f.readline())
                })
                string=f.readline()
                for i in range(len(data[-1]['ners'])):
                    data[-1]['ners'][i] = data[-1]['ners'][i].strip('\'')
                data[-1]['ners']={i for i in data[-1]['ners']}
    except OSError as e:
        print(e)
    return data


def read_coms(file_name):
    data = [[]]
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            string = f.readline()
            while string:
                # id
                # comment_text
                # post_id
                # date
                # ners
                # ners_num
                # words_num
                data.append({
                    'id': int(string),
                    'comment_text': f.readline().strip('\n'),
                    'post_id': int(f.readline()),
                    'date': f.readline().strip('\n'),
                    'ners': f.readline().strip('{\}\n').split(', '),
                    'ners_num': int(f.readline()),
                    'words_num': int(f.readline())
                })
                string=f.readline()
                for i in range(len(data[-1]['ners'])):
                    data[-1]['ners'][i] = data[-1]['ners'][i].strip('\'')
                data[-1]['ners']={i for i in data[-1]['ners']}
    except OSError as e:
        print(e)
    return data



# data_posts=read_posts('posts_parsed.csv')
# data_coms=read_coms('coms_parsed.csv')
data_posts=read_posts('output1.csv')
data_coms=read_coms('output2.csv')

data_posts.pop(0)
data_coms.pop(0)

print(1)

for i in reversed(range(len(data_coms))):
    # checker.set_text(data_coms[i]['comment_text'])

    if data_coms[i]['ners_num'] == 0:
        data_coms.remove(data_coms[i])
        continue
    elif data_coms[i]['words_num'] < 4:
        data_coms.remove(data_coms[i])
        continue
    elif sgd_clf_psy.predict([data_coms[i]['comment_text']]) == 'not_psychology':
        data_coms.remove(data_coms[i])
        continue
    elif sgd_clf_ads.predict([data_coms[i]['comment_text']]) == 'advert':
        data_coms.remove(data_coms[i])
        continue
    f=0
    try:
        post = [x for x in data_posts if x['id'] == data_coms[i]['post_id']][0]
    except IndexError:
        data_coms.remove(data_coms[i])
        continue
    if not (post['ners'] & data_coms[i]['ners']):
        data_coms.remove(data_coms[i])
        continue

# write_csv_file(data_coms, 'coms_filtered.csv')
write_csv_file(data_coms, 'output4.csv')

print(2)

for i in reversed(range(len(data_posts))):
    # checker.set_text(data_posts[i]['post_text'])
    if data_posts[i]['ners_num'] == 0:
        data_posts.remove(data_posts[i])
    elif data_posts[i]['words_num'] <= 10:
        data_posts.remove(data_posts[i])
    elif data_posts[i]['comments_num'] == 0:
        data_posts.remove(data_posts[i])
    elif sgd_clf_psy.predict([data_posts[i]['post_text']]) == 'not_psychology':
        data_posts.remove(data_posts[i])
    elif sgd_clf_ads.predict([data_posts[i]['post_text']]) == 'advert':
        data_posts.remove(data_posts[i])

# write_csv_file(data_posts, 'posts_filtered.csv')
write_csv_file(data_posts, 'output3.csv')