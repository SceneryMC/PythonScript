import os
import subprocess
from secret import pd_user_list, pd_path


def get_last_downloaded_user():
    downloaded_folders = os.listdir(os.path.dirname(pd_path))
    for _id, name in pd_user_list[::-1]:
        if name in downloaded_folders:
            return name
    return ''


if __name__ == '__main__':
    for i in range(100):
        user = get_last_downloaded_user()
        print(f'----------------第{i}次循环，从【{user}】继续下载！----------------')
        result = subprocess.run([r'C:\Users\13308\PycharmProjects\SceneryMCPythonScript\venv\Scripts\python.exe', 'download_marked.py'],
                                encoding='utf-8',
                                input=f'ul\nFalse\n{user}\n')
        if result.returncode == 0:
            break