import time

from pixivpy3 import AppPixivAPI

from secret import proxies, pd_token

api = AppPixivAPI(proxies=proxies)
api.auth(refresh_token=pd_token)

with open('artists.txt', 'r', encoding='utf-8') as f:
    for line in f:
        index = line.find(' ')
        ls = eval(line[index + 1:])
        for _id, _ in ls:
            while True:
                r = api.user_follow_add(_id, restrict='private')
                print(_id, r)
                time.sleep(10)
                if 'error' not in r:
                    break
                time.sleep(60)

