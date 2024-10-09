from download_marked import api
from pixiv_downloader.maintain_symlink import user_id_to_name
from pixiv_downloader.utils import replace_filename, get_info_with_retry
from secret import pd_pid


def fetch_following_list():
    users = []
    json_result = get_info_with_retry(api.user_following, keyword='user_previews', user_id=pd_pid, restrict="private")
    while True:
        users.extend((d.user.id, replace_filename(d.user.name)) for d in json_result.user_previews)
        if json_result.next_url is None:
            break
        next = api.parse_qs(json_result.next_url)
        json_result = get_info_with_retry(api.user_following, keyword='user_previews', **next)
    return users


if __name__ == '__main__':
    result = dict(fetch_following_list())
    current = set(user_id_to_name.keys())
    print(list((k, result[k]) for k in set(result.keys()) - current))
