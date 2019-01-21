pandas
================

NumPy 的 ndarray 数据处理要求数据类型一致，且不能缺失，不可为数据项添加额外标签等，为了解决 ndarray 的强类型限制，Panda 对 NumPy 的 ndarray 对象进行了扩展。

建立在 NumPy 数组结构上的 Pandas， 提供了 Series 和 DataFrame 对象，为极度繁琐和耗时的“数据清理”（data munging）任务提供了捷径。

笔者环境使用的 pandas 版本为 0.20.3，使用 Anaconda 提供的集成数据处理环境。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  import pandas as pd
  print(pd.__version__)
  
  >>>
  0.20.3

