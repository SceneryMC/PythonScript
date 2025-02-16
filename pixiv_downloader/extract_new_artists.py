import json
from collections import defaultdict
from secret import pd_tags, pd_user_list


users = set(_id for _id, _ in pd_user_list)
processed = 1
ls = []
for tags, _type in pd_tags[:28]:
    path = f'text_files/search_results/{tags[0]}.json'
    with open(path, 'r', encoding='utf-8') as f:
        d_new = json.load(f)
    d_users = defaultdict(list)
    for _id, info in d_new.items():
        user_id = info['user']['id']
        if info['user']['id'] not in users and set(e['name'] for e in info['tags']) & set(tags):
            d_users[user_id].append(info['total_bookmarks'])
    results = [(_id, max(count)) for _id, count in d_users.items()]
    ls.append((tags[0], [(_id, count) for _id, count in results if count >= 2000]))
processed_ids = set()
processed_characters = {}
for character, result in ls:
    if character in processed_characters:
        processed_ids |= set(_id for _id, count in result)
for character, result in ls:
    if character not in processed_characters:
        print(character, [(_id, count) for _id, count in result if _id not in processed_ids])

