import requests
from csv import writer
from time import strftime
from time import gmtime
from time import sleep
from re import findall
from re import sub
import spacy
from pymorphy2 import MorphAnalyzer

TOKEN_USER = 'vk1.a.fsFgV7zP9wLwaRKsEzBb1gLF_hHUGHqyMFdSHtmjZENYwYdVZEfdijvc7Gm2wyIcPdu-XuoNjwoy1j5Js8kojFgUH6nLgIYGYsuEenF3-K1iOunjaY1F4UgsP3un0iC1rrRDiuaK03mltGHvqXOWZYtBI6-cSD2eI4h4Z40aB_tsY-yLa90f6uridaChe5dFsdKX7GbIs8zDtoOcSTsqDw'
VERSION = '5.131' 
DOMAIN = 'localhost'
GROUPS_COUNT = 100 # максимум 1000
POSTS_IN_BLOCK = 100 # максимум 100
POSTS_BLOCKS_COUNT = 10
COMS_IN_BLOCK = 100 # максимум 100
COMS_BLOCKS_COUNT = 10
nlp = spacy.load("ru_core_news_sm")

print(1)

def write_csv_file(data, file_name):
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        write = writer(csvfile, delimiter = '\n', lineterminator = '\r\n')
        for row in data:
            write.writerow(row)
            
def norm(x):
    morph = MorphAnalyzer()
    p = morph.parse(x)[0]
    return p.normal_form

response1 = requests.get('https://api.vk.com/method/groups.search', 
params={'access_token': TOKEN_USER,
        'v': VERSION,
        'q': 'психоло',
        'type': 'group',
        'sort': 6,#сортировка по числу участников
        'count': GROUPS_COUNT})

print(2)

# try:
#     with open('json_coms.txt', 'w', encoding='utf-8') as f:
#         print(response1.text, file = f)
#         print(response2.text, file = f)
# except OSError as e:
#     print(e)

groups = response1.json()['response']['items']

f=1
data_posts = []
data_coms = []
data_subcoms = []
for group in groups:
    for i in range(POSTS_BLOCKS_COUNT):
        res_posts = requests.get('https://api.vk.com/method/wall.get', 
        params={'access_token': TOKEN_USER,
                'v': VERSION,
                'owner_id': -group['id'],
                'offset': POSTS_IN_BLOCK*i,
                'count': POSTS_IN_BLOCK})
        sleep(0.1)
        if f==1:
            with open('json_posts.txt', 'w', encoding='utf-8') as file:
                print(res_posts.text, file=file)
        try:
            data_posts.append(res_posts.json()['response'])
        except KeyError:
            continue
    print(f)
    f+=1

print(3)
            
print_data_posts = [] 
for i in range(len(data_posts)):
    for post in data_posts[i]['items']:
        doc = nlp(post['text'].lower())
        post_ners = {''}
        for ent in doc.ents:
            post_ners.add(norm(ent.text))
        post_text = sub(r'\n+', ' ', post['text'])
        post_text = sub(r' +', ' ', post_text)
        if post_text=='':
            post_text=' '
        ners_num=len(post_ners)-1
        if len(post_ners)>1:
            post_ners.remove('')
        try:
            group = [x for x in groups if x['id'] == -post['owner_id']][0]
        except IndexError:
            print(0)
            continue
        # id
        # post_text
        # group_link
        # date
        # ners
        # ners_num
        # words_num
        # comments_num
        print_data_posts.append([
            post['id'], 
            post_text, 
            'vk.com/'+group['screen_name'], 
            strftime('%d %B %Y', gmtime(post['date'])), 
            post_ners, 
            ners_num,
            len(findall(r'\w+', post['text'])), 
            post['comments']['count']])
        
write_csv_file(print_data_posts, 'posts_parsed.csv')
# write_csv_file(print_data_posts, 'output1.csv')

print(4)

j=0
for i in range(len(data_posts)):
    for post in data_posts[i]['items']:
        for k in range(COMS_BLOCKS_COUNT):
            try:
                if post['comments']['count'] <= k*COMS_IN_BLOCK:
                    break
            except KeyError:
                break
            res_coms = requests.get('https://api.vk.com/method/wall.getComments', 
            params={'access_token': TOKEN_USER,
                    'v': VERSION,
                    'owner_id': post['owner_id'],
                    'post_id': post['id'],
                    'offset': COMS_IN_BLOCK*k,
                    'count': COMS_IN_BLOCK})
            sleep(0.1)
            try:
                data_coms.append(res_coms.json()['response'])
            except KeyError:
                None
            for com in data_coms[j]['items']:
                try:
                    if com['thread']['count']==0:
                        continue
                except KeyError:
                    continue
                res_subcoms = requests.get('https://api.vk.com/method/wall.getComments',
                params={'access_token': TOKEN_USER,
                        'v': VERSION,
                        'owner_id': post['owner_id'],
                        'post_id': post['id'],
                        'count': 100,
                        'comment_id': com['id']})
                try:
                    data_subcoms = res_subcoms.json()['response']
                    if data_subcoms['count'] > 0:
                        data_coms.append(data_subcoms)
                except KeyError:
                    None
            j+=1

print(5)

print_data_coms = []
for i in range(len(data_coms)):
    for com in data_coms[i]['items']:
        doc = nlp(com['text'].lower())
        com_ners = {''}
        for ent in doc.ents:
            com_ners.add(norm(ent.text))
        com_text = sub(r'\n+', ' ', com['text'])
        com_text = sub(r' +', ' ', com_text)
        if com_text=='':
            com_text=' '
        ners_num=len(com_ners)-1
        if len(com_ners)>1:
            com_ners.remove('')
        # id
        # comment_text
        # post_id
        # date
        # ners
        # ners_num
        # words_num
        try:
            print_data_coms.append([
                com['id'], 
                com_text, 
                com['post_id'], 
                strftime('%d %B %Y', gmtime(com['date'])), 
                com_ners, 
                ners_num,
                len(findall(r'\w+', com['text']))])
        except KeyError:
            continue
        
write_csv_file(print_data_coms, 'coms_parsed.csv')
write_csv_file(print_data_coms, 'output2.csv')