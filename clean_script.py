import io
import os
from nbstripout import strip_output, read, write, NO_CONVERT

all_notebooks = []
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('ipynb'):
            all_notebooks.append(os.path.join(root, f))

print('{} notebooks found'.format(len(all_notebooks)))

for filename in all_notebooks:
    print(filename)
    try:
        with io.open(filename, 'r', encoding='utf8') as f:
            nb = read(f, as_version=NO_CONVERT)
        nb = strip_output(nb, False, False)
        with io.open(filename, 'w', encoding='utf8') as f:
            write(nb, f)
    except Exception as e:
        # print(e.message)
        continue
