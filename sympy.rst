sympy
================

matplotlib 是 python 中一个非常强大的 2D 函数绘图模块，它提供了子模块 pyplot 和 pylab 。pylab 是对 pyplot 和 numpy 模块的封装，更适合在 IPython 交互式环境中使用。

对于一个项目来说，官方建议分别导入使用，这样代码更清晰，即：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  import numpy as np
  import matplotlib.pyplot as plt

而不是

.. code-block:: python
  :linenos:
  :lineno-start: 0

  import pylab as pl
