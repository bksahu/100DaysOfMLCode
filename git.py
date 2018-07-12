import git
import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

repo = git.Repo(path)
#repo.git.pull()
print(repo.git.status())

commit_message = repo.untracked_files[0] + ' added'

def unstaged_files():
    untracked_files = repo.untracked_files
    changedFiles = [item.a_path for item in repo.index.diff(None)]
    return changedFiles + untracked_files     

repo.git.add([file for file in unstaged_files()])

repo.git.commit('-m', commit_message)

