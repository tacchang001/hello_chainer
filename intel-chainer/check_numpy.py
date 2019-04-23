import numpy as np
import ideep4py
import chainer

np.show_config()

print(chainer.backends.intel64.is_ideep_available())

x = np.ones((3, 3), dtype='f')
with chainer.using_config('use_ideep', 'auto'):
    y = chainer.functions.relu(x)
print(type(y.data))
