pandas
================

NumPy 的 ndarray 数据处理要求数据类型一致，且不能缺失，不可为数据项添加额外标签等，为了解决 ndarray 的强类型限制，Panda 对 NumPy 的 ndarray 对象进行了扩展。

建立在 NumPy 数组结构上的 Pandas， 提供了 Series 和 DataFrame 对象，为极度繁琐和耗时的“数据清理”（data munging）任务提供了捷径。

笔者使用 Anaconda 提供的集成数据处理环境，查看 pandas 版本：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  import pandas as pd
  print(pd.__version__)
  
  >>>
  0.20.3

基本数据结构
-------------------

pandas 在 NumPy 的 ndarray 对象基础上封装了三个基本数据结构 Series、 DataFrame 和 Index。 Pandas 在这些基本数据结构上实现了许多功能和方法。

Series 对象
~~~~~~~~~~~~~~~~ 

Series 对象是一个带索引标签（Labels）的一维数组，打印查看很像只有一列的表，可以看做向量。可以使用 list 作为参数，来生成对应 Series 对象，例如：

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  sdata = pd.Series([1, 2, 3.14])
  print(sdata)
  
  >>>
  0    1.00  # 默认使用从 0 开始的整数作为索引
  1    2.00
  2    3.14
  dtype: float64

  print(type(sdata).__name__)
  print(sdata.dtype) # dtype 属性记录成员的类型
  
  >>>
  Series
  float64

可以使用索引访问 Series 对象成员，如果使用切片返回的是一个 Series 对象。

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  print(type(sdata[1]).__name__, sdata[1])
  print(type(sdata[0:-1]).__name__)  
  
  >>>
  float64 2
  Series

如同查看列表长度一样，可以使用 len() 查看成员数目：

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  print(len(sdata))
  
  >>>
  3

Series 索引
``````````````

Series 对象和一维 NumPy 数组的本质差异在于索引：

- NumPy 数组通过隐式定义的整数索引获取数值。
- Pandas 的 Series 对象用显式定义的 RangeIndex 索引与数值关联。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 打印 RangeIndex 类型
  print(sdata.index)
  
  >>>
  RangeIndex(start=0, stop=3, step=1)

显式索引让 Series 对象拥有了更具弹性的索引方式。 索引不再局限于整数，可以是任意想要的类型。例如用字符串作为索引：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  sdata = pd.Series([1, 2, 3.14], index=['num1', 'num2', 'pi'])
  print(sdata)
  
  >>>
  num1    1.00
  num2    2.00
  pi      3.14
  dtype: float64
  
  # 使用字符串作为索引  
  print(sdata['pi'])
  
  >>>
  3.14

Series 成员可以是其他任何对象，也可以是不同对象，这看起来很像字典，此时它的类型为 object：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  sdata = pd.Series({'a': 1, 'b': 2, 'c': 'abc'})
  print(sdata)
  
  >>>
  a      1
  b      2
  c    abc
  dtype: object

Series 是特殊字典
``````````````````

字典是一种将任意键映射到一组任意值或对象的数据结构，而 Series 对象是一种将类型键映射到一组类型值的数据结构。Pandas Series 的类型信息使得它在某些操作上比 Python 的字典更高效。

可以直接用 Python 的字典创建一个 Series 对象：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  id_dicts = {'John': 100,
              'Tom' : 101,
              'Bill': 102}
  ids = pd.Series(id_dicts)
  print(ids['Bill'])
  
  >>>
  102
  
  # 元素顺序按照索引字母大小进行排序
  print(ids)
  
  >>>
  Bill    102
  John    100
  Tom     101
  dtype: int64

和字典不同，Series 对象还支持数组形式的操作， 比如切片：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 注意切片索引顺序不是按照字典中元素定义顺序，而是按照 Series 对象的索引顺序
  sub_ids = ids['Bill':'John']
  print(sub_ids)
  
  >>>
  Bill    102
  John    100
  dtype: int64

创建 Series 对象
``````````````````

::

  pd.Series(data, index=index)

创建 Series 对象的格式如上所示，index 可选，指定索引序列，默认值为整数序列；data 参数支持多种数据类型：列表，字典或者一维的 ndarray 对象。

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  ndata = np.arange(1, 4, 1)
  sdata = pd.Series(ndata)
  print(sdata)
  
  >>>
  0    1
  1    2
  2    3
  dtype: int32

data 也可以是一个数值， 创建 Series 对象时会重复填充到每个索引上：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  sdata = pd.Series(1, index=['a', 'b', 'c'])
  print(sdata)
  
  >>>
  a    1
  b    1
  c    1
  dtype: int64

当参数为字典时，可以通过显式指定索引筛选需要的成员：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  subsdata = pd.Series({'a': 1, 'b': 2, 'c': 'abc'}, index=['a', 'c'])
  print(subsdata)
  
  >>>
  a    1
  c    abc
  dtype: object

.. admonition:: 注意

  Series 对象只会保留显式定义的键值对。

Series.index 属性获取所有行索引信息：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 获取行索引信息
  print(subsdata.index)

  >>>
  Index(['a', 'c'], dtype='object')

DataFrame 对象
~~~~~~~~~~~~~~~~

如果将 Series 类比为带索引的一维数组（或者含有一列数据的带有行标签的单列表）， 那么 DataFrame 就可以看作是一种既有行索引，又有列名的二维数组（或者含有行标签和列标签的表，每一列都是一个 Series 对象）。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  id_dicts = {'John': 100,
              'Tom' : 101,
              'Bill': 102}
  
  age_dicts = {'John': 20,
               'Tom' : 21,
               'Bill': 19}

  studentd = pd.DataFrame({'id':  pd.Series(id_dicts),
                          'age': pd.Series(age_dicts)})
  print(studentd)
  
  >>>
        age   id
  Bill   19  102
  John   20  100
  Tom    21  101

从示例中可以看出 DataFrame 是一组 Series 的集合，每一列都是一个 Series 对象。

DataFrame 索引
``````````````````

在 NumPy 的二维数组里， data[0] 返回第一行；而在 DataFrame 中， data['col0'] 返回第一列。 因此，DataFrame 是一种通用字典，而不是通用数组。

.. code-block:: python
  :linenos:
  :lineno-start: 0

  # 使用列名字访问特定列
  print(studentd['age'])
  
  >>>
  Bill    19
  John    20
  Tom     21
  Name: age, dtype: int64
  
  # 指定列名和行名
  print(studentd['age']['John'])
  
  >>>
  20

创建DataFrame对象 
```````````````````

上面的示例指定列名和 Series 对象创建多列，也可以创建单列的 DataFrame 对象：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 以下两种创建方式等价
  ids = pd.Series(id_dicts)
  
  # 通过 Series 对象字典创建
  studentd = pd.DataFrame({'id': ids})
  studentd = pd.DataFrame(ids, columns=['id'])

通过字典列表创建: 任何元素是字典的列表都可以变成 DataFrame。 

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  # 创建字典列表
  num = [{'num0': i, 'num*3': 3 * i} for i in range(3)]
  print(num)
  
  >>>
  [{'num0': 0, 'num*3': 0}, {'num0': 1, 'num*3': 3}, {'num0': 2, 'num*3': 6}]
  
  # 创建 DataFrame 对象
  print(pd.DataFrame(num))
  
  >>>
     num*3  num0
  0      0     0
  1      3     1
  2      6     2

如果字典中有些键不存在，Pandas 会用 NaN（不是数字或此处无数，Not a number） 来表示：

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  numd = pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
  print(numd)
  
       a  b    c
  0  1.0  2  NaN
  1  NaN  3  4.0

通过 NumPy 二维数组创建。 假如有一个二维数组， 就可以创建一个可以指定行列索引值的 DataFrame。 如果不指定行列索引值，那么行列默认都是整数索引值：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  narray = np.random.randint(3, size=(3, 2))
  print(narray)
  
  >>>
  [[2 0]
   [2 2]
   [2 1]]
   
  df = pd.DataFrame(narray,
                    columns = ['foo', 'bar'],
                    index=['a', 'b', 'c'])
  print(df)
  
  >>>
     foo  bar
  a    2    0
  b    2    2
  c    2    1

查看行索引和列索引：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  print(df.index)
  print(df.columns)  
  
  >>>
  Index(['a', 'b', 'c'], dtype='object')
  Index(['foo', 'bar'], dtype='object')
    
通过 NumPy 结构化数组创建：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  A = np.ones(3, dtype=[('A', 'i8'), ('B', 'f8')])
  print(A)
  
  >>>
  [(1,  1.) (1,  1.) (1,  1.)]
  
  print(pd.DataFrame(A))
  
  >>>
     A    B
  0  1  1.0
  1  1  1.0
  2  1  1.0

更改行或列名
~~~~~~~~~~~~~~~

通过属性更改
```````````````

可以通过 df.index 和 df.columns 查看行名和列名，同样可以通过这些属性更改行或列的名称，对于 Series 来说只有行名：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  narray = np.random.randint(3, size=(3, 2))
  df = pd.DataFrame(narray,
                    columns = ['foo', 'bar'],
                    index=['a', 'b', 'c'])
  
  # 更新行索引标签                 
  df.index = [0, 1, 2]
  print(df.index)
  
  # 更新列索引标签
  df.columns = ['a', 'b']
  print(df.columns)
  
  >>>
  Int64Index([0, 1, 2], dtype='int64')
  Index(['a', 'b'], dtype='object')
  
  print(df)

  >>>
     a  b
  0  0  0
  1  1  0
  2  0  2

注意行或列的标签个数和 DataFrame 对象的行数或列数必须一致，否则会报错。

Index 对象
~~~~~~~~~~~~~~

Pandas 的 Index 对象可以将它看作是一个不可变数组或有序集合， Index 对象可以包含重复值。 

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 可以包含重复值
  ind = pd.Index([2, 3, 5, 7, 7, 11])
  print(type(ind).__name__)
  
  >>>
  Int64Index
  
  # 索引访问元素
  print(ind[1])
  >>>
  3
  
  # 切片访问返回 Index 对象
  print(ind[::2])
  
  >>>
  Int64Index([2, 5, 7], dtype='int64')
  
Index 对象不支持对数据的修改：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  ind[1] = 1
  
  >>>
  TypeError: Index does not support mutable operations

Index 对象还有许多与 NumPy 数组相似的属性：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(ind.size, ind.shape, ind.ndim, ind.dtype)
  
  >>>
  6 (6,) 1 int64

排序操作
````````````

Index 对象支持对元素的排序：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  ind = pd.Index([2, 4, 5, 1, 11])
  print(ind)
  
  >>>
  Int64Index([2, 4, 5, 1, 11], dtype='int64')
  
  ind = ind.sort_values()
  print(ind)
  
  >>>
  Int64Index([1, 2, 4, 5, 11], dtype='int64')

集合操作
``````````````````

Pandas 对象被设计用于实现多种操作， 如连接（join） 数据集，其中会涉及许多集合操作。 Index 对象遵循 Python 标准库的集合（set） 数据结构的许多习惯用法， 包括并集、 交集、 差集等：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  indA = pd.Index([1, 3, 5, 7, 9])
  indB = pd.Index([2, 3, 5, 7, 11])
  
  # 交集，等价于 indA.intersection(indB)
  print(indA & indB)
  
  >>>
  Int64Index([3, 5, 7], dtype='int64')
  
  # 并集
  print(indA | indB)
  
  >>>
  Int64Index([1, 2, 3, 5, 7, 9, 11], dtype='int64')
  
  # 异或
  print(indA ^ indB)
  
  >>>
  Int64Index([1, 2, 9, 11], dtype='int64')

Index 对象进行集合操作的结果还是 Index 对象。它可以是一个空对象。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  indA = pd.Index([1, 3, 5, 7, 9])
  indB = pd.Index([2])
  
  # 交集
  print(indA & indB)
  
  >>>
  Int64Index([], dtype='int64')

数据选择和扩展
---------------

NumPy 数组可以通过索引，切片，花式索引和掩码操作进行各类选择，Pandas 的 Series 和 DataFrame 对象具有相似的数据获取与调整操作。

Series数据选择
~~~~~~~~~~~~~~~~~

访问数据
```````````````

将Series看作字典，和字典一样， Series 对象提供了键值对（索引）的映射：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 使用 in 或者 not in 判断键是否存在
  sdata = pd.Series([1, 2, 3.14], index=['num1', 'num2', 'pi'])
  
  # 等价于 sdata.index
  print(sdata.keys())
  print('pi' in sdata) # 等价于 'pi' in sdata.keys()
  
  >>>
  Index(['num1', 'num2', 'pi'], dtype='object')
  True
  
  # 判断值是否存在，Series.values 是 ndarray 类型
  print(sdata.values, type(sdata.values).__name__)
  print(1 in sdata.values)
  
  >>>
  [ 1.    2.    3.14] ndarray
  True

  # Series.items() 返回 zip 类型，可以转换为 list
  print(sdata.items())
  print(list(sdata.items()))
  
  >>>
  <zip object at 0x0000020B8A3DCF08>
  [('num1', 1.0), ('num2', 2.0), ('pi', 3.1400000000000001)]

Series 不仅有着和字典一样的接口， 而且还具备和 NumPy 数组一样的数组数据选择功能，包括索引、掩码、花式索引等操作，例如：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 将显式索引作为切片，结果包含最后一个索引
  subs = sdata['num1':'num2']
  print(subs)
  
  >>>
  num1    1.0
  num2    2.0
  dtype: float64
  
  # 将隐式整数索引作为切片，结果不含最后一个索引
  print(sdata[0:2])
  print(sdata[-1:0:-1])
  
  >>>
  num1    1.0
  num2    2.0
  dtype: float64

  pi      3.14
  num2    2.00
  dtype: float64
    
  # 掩码，返回 bool 类型的 Series 掩码对象
  print((sdata > 1) & (sdata < 4))
  
  >>>
  num1    False
  num2     True
  pi       True
  dtype: bool
  
  # Series 掩码对象作为索引
  subs = sdata[(sdata > 1) & (sdata < 4)]
  print(subs)
  
  >>>
  num2    2.00
  pi      3.14
  dtype: float64
  
  # 花式索引
  subs = sdata[['num1', 'pi']]
  print(subs)
  
  >>>
  num1    1.00
  pi      3.14
  dtype: float64

切片是绝大部分混乱之源。 需要注意的是，当使用显式索引（即 data['a':'c']） 作切片时， 结果包含最后一个索引； 而当使用隐式索引（即 data[0:2]） 作切片时， 结果不包含最后一个索引。

索引标签默认是无序的，也即根据创建时标签声明的顺序来排列，我们可以对它进行排序，以方便切片操作：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  sdata = pd.Series([1, 2, 3.14], index=['num2', 'num1', 'pi'])
  print(sdata)
  
  >>>
  num2    1.00
  num1    3.14
  pi      2.00
  dtype: float64
  
  # 对索引进行排序
  sdata = sdata.sort_index()
  print(sdata['num1':'num2'])
  
  >>>
  num1    2.0
  num2    1.0
  dtype: float64

索引器
````````````

切片和取值的习惯用法经常会造成混乱。如果 Series 是显式整数索引，那么 data[1] 这样的取值操作会使用显式索引，而 data[1:3] 样的切片操作却会使用隐式索引。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  sdata = pd.Series([1, 2, 3.14], index=[1, 2, 3])
  print(sdata[1]) # 显式索引，使用 sdata[0] 将报错
  
  >>>
  1.0
  
  print(sdata[0:2]) # 隐式索引，不含 sdata[2]
  
  >>>
  1    1.0
  2    2.0
  dtype: float64

由于整数索引很容易造成混淆，所以 Pandas 提供了一些索引器（indexer） 属性来作为取值的方法。它们不是 Series 对象的函数方法， 而是暴露切片接口的属性。

第一种索引器是 loc 属性， 表示取值和切片都是显式的：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  sdata = pd.Series([1, 2, 3.14], index=[1, 2, 3])
  print(sdata.loc[1])   # 显式索引
  
  >>>
  1.0
  
  print(sdata.loc[1:2]) # 显式索引
  
  >>>
  1    1.0
  2    2.0
  dtype: float64

第二种是 iloc 索引属性，表示取值和切片都是隐式索引（从 0 开始， 左闭右开区间）：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  sdata = pd.Series([1, 2, 3.14], index=[1, 2, 3])
  print(sdata.iloc[1])  # 隐式索引

  >>>  
  2.0
  
  print(sdata.iloc[1:2])# 隐式索引
  
  >>>
  2    2.0
  dtype: float64
  
第三种取值属性是 ix，它是前两种索引器的混合形式，从 0.20.0 版本开始，ix 索引器不再被推荐使用。

Python 代码的设计原则之一是“显式优于隐式”。 使用 loc 和 iloc 可以让代码更容易维护， 可读性更高。 特别是在处理整数索引的对象时， 我强烈推荐使用这两种索引器。 它们既可以让代码阅读和理解起来更容易， 也能避免因误用索引 / 切片而产生的小 bug。

扩展数据
```````````````

Series 对象还可以用字典语法调整数据。可以通过增加新的索引值扩展 Series：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  sdata['e'] = 2.72
  print(sdata)
  
  >>>
  num1    1.00
  num2    2.00
  pi      3.14
  e       2.72
  dtype: float64

DataFrame数据选择
~~~~~~~~~~~~~~~~~~

访问数据
```````````````

既可以通过字典方式也可以通过属性方式访问 DataFrame :

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  studentd = pd.DataFrame({'id':  pd.Series(id_dicts),
                          'age': pd.Series(age_dicts)})
  print(studentd['id']['John']) # 字典键方式访问
  
  >>>
  100
  
  print(studentd['id']) # 列属性方式访问

  >>>
  Bill    102
  John    100
  Tom     101
  Name: id, dtype: int64
 
  print(studentd['id']['John']) # 列属性和行属性方式访问
  
  >>>
  100 

虽然属性形式的数据选择方法很方便， 但是它并不是通用的。 如果列名不是纯字符串， 或者列名与 DataFrame 的方法同名， 那么就不能用属性索引。 例如， DataFrame 有一个 pop() 方法， 如果用data.pop 就不会获取 'pop' 列， 而是显示为方法。

另外， 还应该避免对用属性形式选择的列直接赋值（即可以用data['pop'] = z，但不要用 data.pop = z）防止覆盖方法名。

和前面介绍的 Series 对象一样，还可以用字典形式的语法调整对象，如果要增加一列可以这样做：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 等价于 studentd['newcol'] = studentd.id + studentd.age
  studentd['newcol'] = studentd['id'] + studentd['age']
  print(studentd)
  
  >>>
        age   id  newcol
  Bill   19  102     121
  John   20  100     120
  Tom    21  101     122

将DataFrame看作二维数组，用 values 属性按行查看数组数据：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(studentd.values, '\n', type(studentd.values).__name__)
  
  >>>
  [[ 19 102]
   [ 20 100]
   [ 21 101]] 
   ndarray

由于返回值是 ndarray 类型，所以可以对其进行任何矩阵操作：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 获取行数据（获取一列数据要传递列索引）
  print(studentd.values[0])
  
  >>>
  [ 19 102]
  
  print(studentd.values.T)
  
  >>>
  [[ 19  20  21]
   [102 100 101]]
   
  print(studentd.keys())

keys() 方法返回列名组成的索引类型 Index：

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  >>>
  Index(['age', 'id'], dtype='object')

使用索引器
``````````````

索引器的作用在于指明使用隐式索引还是显示索引。通过 iloc 索引器，可以像对待 NumPy 数组一样索引 Pandas 的底层数组（Python 的隐式索引），DataFrame 的行列标签会自动保留在结果中：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(studentd.iloc[:1, :2])
  
  >>>
        age   id
  Bill   19  102

任何用于处理 NumPy 形式数据的方法都可以用于这些索引器。例如，可以在 loc 索引器中结合使用掩码与花式索引方法：  

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 选择 age >= 20 的学生的 id 信息
  print(studentd.loc[studentd.age >= 20, ['id']])
  
  >>>
        id
  John  100  
  Tom   102  

切片选择
```````````````

如果对单个标签取值就选择列，而对多个标签用切片就选择行：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 列选取，返回 Series 对象
  print(studentd['age'])
  
  >>>
  Name: age, dtype: int64
  
  # 行选取，返回 DataFrame 对象
  print(studentd['John':'Tom'])
  
  >>>
        age   id
  John   20  100
  Tom    21  101

切片也可以不用索引值， 而直接用行数来实现：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(studentd[1:3])

  >>>
        age   id
  John   20  100
  Tom    21  101

与之类似，掩码操作也可以直接对每一行进行过滤，而不需要使用 loc 索引器：  

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  print(studentd[studentd.age >= 20])
  
  >>>
        age   id
  John   20  100
  Tom    21  101

更新数据
`````````````

任何一种索引方法都可以用于调整数据， 这一点和 NumPy 的常用方法是相同的：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  studentd.loc['John', 'age'] = 23 
  print(studentd)
  
  >>>
        age   id
  Bill   19  102
  John   23  100
  Tom    21  101

  # 更新第一行的值全为 5
  studentd.iloc[0] = 5
  print(studentd)  
    
        age   id
  Bill    5    5
  John   20  100
  Tom    21  101  

数值运算
-----------

NumPy 的基本能力之一是快速对每个元素进行运算，既包括基本算术运算（加、 减、 乘、 除） ， 也包括更复杂的运算（三角函数、 指数函数和对数函数等），参考 :ref:`array_scalar`。 Pandas 继承了 NumPy 的功能，也即这些函数同样可以作用在 Pandas 对象上。

除此之外，Pandas 也实现了一些高效技巧：一元运算作用在 Pandas 对象上时会保留索引和列标签；而对于二元运算（如加法和乘法），Pandas 在传递通用函数时会自动对齐索引进行计算。这就意味着，保存数据内容与组合不同来源的数据——两处在NumPy 数组中都容易出错的地方在 Pandas 中很容易实现。

一元运算
~~~~~~~~

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  sdata = pd.Series(np.arange(4))
  print(sdata * 2)
  
  >>>  
  0    0
  1    2
  2    4
  3    6
  dtype: int32

可以发现 np 函数作用在 Pandas 对象上的返回值还是 Pandas 对象，会保留原标签。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  df = pd.DataFrame(np.arange(4).reshape(2, 2), columns=['a', 'b'])
  print(np.sin(df / 4 * np.pi))
  
  >>>
       a         b
  0  0.0  0.707107
  1  1.0  0.707107

二元运算
~~~~~~~~

当在两个 Series 或 DataFrame 对象上进行二元计算时，Pandas 会在计算过程中对齐两个对象的索引。当处理不完整的数据时，这一点非常方便。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  sdata0 = pd.Series(np.arange(3))
  sdata1 = pd.Series(np.arange(2))
  print(sdata0)
  print(sdata1)
  
  >>>
  0    0
  1    1
  2    2
  dtype: int32
  0    0
  1    1
  dtype: int32

首先生成两个索引不同的 Series 对象，然后进行相加：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(sdata0 + sdata1)
  
  >>>
  0    0.0
  1    2.0
  2    NaN
  dtype: float64
 
结果数组的索引是两个输入数组索引的并集。对于缺失位置的数据，Pandas 会用 NaN 填充，表示“此处无数”。这是 Pandas 表示缺失值的方法。

如果用 NaN 值不是我们想要的结果， 那么可以用适当的对象方法代替运算符。 例如， A.add(B) 等价于 A + B， 也可以设置参数自定义 A 或 B 缺失的数据：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # sdata1 中缺失的索引 2 的值将使用 0 替代
  print(sdata0.add(sdata1, fill_value=0))
  
  >>>
  0    0.0
  1    2.0
  2    2.0  # 0 + 2
  dtype: float64

在计算两个 DataFrame 时，类似的索引对齐规则也同样会出现在共同（并集）列中：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  df0 = pd.DataFrame(np.arange(4).reshape(2,2), columns=list('BA'))  
  df1 = pd.DataFrame(np.arange(2).reshape(2,1), columns=list('A'))

  print(df0)
  print(df1)
  
  >>>
     B  A
  0  0  1
  1  2  3
     A
  0  0
  1  1
  
  # 填充缺省值 NaN
  print(df0 + df1)
  >>>
     A   B
  0  1 NaN
  1  4 NaN
  
  # 指定缺省值
  print(df0.sub(df1, fill_value=0))
  >>>
     A    B
  0  1  0.0
  1  2  2.0

两个对象的行列索引可以是不同顺序的，结果的索引会自动按顺序排列。 

Python运算符与Pandas方法的映射关系:

  ============ ============
  Python运算符 Pandas 对象方法
  ============ ============
  \+            add()
  \-            sub()、 subtract()
  \*            mul()、 multiply()
  /            truediv()、 div()、 divide()
  //           floordiv()
  %            mod()
  \*\*           pow()
  ============ ============
 
DataFrame与Series的运算
~~~~~~~~~~~~~~~~~~~~~~~

DataFrame 与 Series 之间的运算遵循 NumPy 中二维数组和一维数组之间的广播运算规则。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  narray0 = np.array([2,2])
  narray1 = np.array([[1,1],[2,2]])
  print(narray0 + narray1)
  
  >>>
  [[3 3]
   [4 4]]
   
  sdata = pd.Series(narray0, index=list('AB'))
  print(sdata)
  
  >>>
  A    2
  B    2
  dtype: int32
  
  df = pd.DataFrame(narray1, columns=list('AB'))
  print(df)
  
  >>>
     A  B
  0  1  2
  1  1  2
  
  print(sdata + df)

  >>>
     A  B
  0  3  3
  1  4  4

根据 NumPy 的广播规则，让二维数组减自身的一行数据会按行计算。如果想按列计算，就需要利用前面介绍过的运算符方法， 通过 axis 参数设置：

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  # 默认按行计算  
  print(df + df.iloc[0])
  
  >>>
     A  B
  0  2  2
  1  3  3
  
  # 按列相加
  print(df.add(df['A'], axis=0))
  
  >>>
     A  B
  0  2  2
  1  4  4

这些行列索引的保留与对齐方法说明 Pandas 在运算时会一直保存这些数据内容， 从而避免在处理数据类型有差异和 / 或维度不一致的 NumPy 数组时可能遇到的问题。

缺失值处理
-----------

现实中采集的数据很少是干净整齐的，许多目前流行的数据集都会有数据缺失的现象。

通常有两种方式表示缺失值： 

1. 通过一个覆盖全局的掩码表示缺失值，例如 R 语言为每个元素保留 1 bit 用于标记缺失值。
2. 用一个标签值（sentinel value） 表示缺失值，比如用 NaN（不是一个数） 表示缺失的浮点数。

Pandas 选择用标签方法表示缺失值，包括两种 Python 原有的缺失值： 浮点数据类型的 NaN 值， 以及 Python 的 None 对象。

None
~~~~~~~~~~

None 是一个 Python 内置对象，经常在代码中表示缺失值。 

.. code-block:: python
  :linenos:
  :lineno-start: 0

  print(None, type(None).__name__)
  
  >>>
  None NoneType
  
由于 None 是一个 Python 对象，只能用于 'object' 数组类型（即由 Python 对象构成的数组），不能用于其他类型的数组：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(np.array([1, None, 3, 4], dtype=object))
  
  >>>
  [1 None 3 4]
  
  # 如果不是 object 类型将报错
  print(np.array([1, None, 3, 4], dtype=int))

  >>>
  TypeError: int() argument must be a string, a bytes-like 
  object or a number, not 'NoneType'

这里 dtype=object 表示 NumPy 认为由于这个数组是 Python 对象构成的，因此将其类型判断为 object。虽然这种类型在某些情景中非常有用，对数据的任何操作最终都会在 Python 层面完成，但是在进行常见的快速操作时，这种类型比其他原生类型数组要更耗时。

由于 Python 没有对 None 对象定义加减等运算操作，所以在包含 None 的数组上执行这类操作均会报错。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  narray = np.array([1, None, 3, 4], dtype=object)
  print(narray.sum())
  
  >>>
  TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'

在 Pandas 中，None 被自动转化为 NaN 类型，由于 NaN 是特殊的浮点数，所以生成的对象类型默认为浮点型 float64： 

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  ps = pd.Series([1, 2, None])
  print(ps)
  
  >>>
  0    1.0
  1    2.0
  2    NaN
  dtype: float64

当为一个整型对象的元素赋值为 None 时，类型自动转换为 float64：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  ps = pd.Series([1, 2])
  print(ps.dtype)
  
  >>>
  int64
  
  ps[0] = None
  print(ps)
  
  >>>
  0    NaN
  1    2.0
  dtype: float64

NaN
~~~~~~~~~~~~~

NaN（全称 Not a Number，不是一个数字），是一种按照 IEEE 浮点数标准设计、在任何系统中都兼容的特殊浮点数。表示未定义或不可表示的值。

IEEE 754-1985中，用指数部分全为1、小数部分非零表示NaN。以32位IEEE单精度浮点数的NaN为例，按位表示即：S111 1111 1AXX XXXX XXXX XXXX XXXX XXXX，S为符号位，符号位S的取值无关紧要；A是小数部分的最高位（the most significant bit of the significand），其取值表示了 NaN 的类型：X 不能全为0，并被称为 NaN 的payload。

通常返回 NaN 的运算有如下三种：

1. 至少有一个参数是 NaN 的运算

2. 不定式

  - 下列除法运算：0/0、∞/∞、∞/−∞、−∞/∞、−∞/−∞
  - 下列乘法运算：0×∞、0×−∞
  - 下列加法运算：∞ + (−∞)、(−∞) + ∞
  - 下列减法运算：∞ - ∞、(−∞) - (−∞)

3. 产生复数结果的实数运算。例如：

  - 对负数进行开偶次方的运算
  - 对负数进行对数运算
  - 对正弦或余弦到达域以外的数进行反正弦或反余弦运算

由于 NaN 是特殊的浮点数，所以当数组成员包含 NaN 时，其类型为浮点型，默认为 float64。 

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  narray = np.array([1, np.nan, 3, 4])
  print(narray.dtype)
  
  >>>
  float64
  
  # 作用在 NaN 上的运算总是返回 NaN
  print(narray.sum())
  
  >>>
  nan
  
  # 指定类型为 int 将报错
  narray = np.array([1, np.nan, 3, 4], dtype=int)
  
  >>>
  ValueError: cannot convert float NaN to integer

NumPy 同时提供了一类特殊的累计函数，参考 :ref:`converge`，它们可以忽略缺失值的影响：

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  print(np.nansum(narray))
  
  >>>
  8.0
  
  print(np.nanmin(narray), np.nanmax(narray))
  
  >>>
  1.0 4.0

.. admonition:: 注意

  NaN 是一种特殊的浮点数， 不是整数、 字符串以及其他数据类型。

np.nan 表示常量 NaN，如果在创建 Pandas 对象时，包含 np.nan 成员，则对象 dtype 自动转化为 float64 类型，同样赋值操作也会改变 dtype：

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  ps = pd.Series([1, 2, np.nan])
  print(ps.dtype)
  
  >>>
  float64
  
  ps = pd.Series([1, 2])
  print(ps.dtype)
  
  >>>
  int64
  
  ps[0] = np.nan
  print(ps)
  
  >>>
  0    NaN
  1    2.0
  dtype: float64

缺失值转换规则
~~~~~~~~~~~~~~~

Pandas对不同类型缺失值的转换规则：

  ========= ====================== ========
  类型      缺失值转换规则         NA标签值
  ========= ====================== ========
  floating  无变化                 np.nan
  object    无变化                 None 或 np.nan
  integer   强制转换为 float64     np.nan
  boolean   强制转换为 float64     np.nan
  boolean   无变化                 None
  ========= ====================== ========

以 bool 类型为例，分别对元素赋值 None 和 np.nan，观察类型变化：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  ps = pd.Series([1, 0, 1], dtype=bool)
  
  # 赋值为 None 等价于 False 类型不变
  ps[1] = None
  print(ps)
  
  >>>
  0     True
  1    False
  2     True
  dtype: bool 
  
  # 赋值为 np.nan 类型转换为 float64
  ps[1] = np.nan
  print(ps)

  >>>
  0    1.0
  1    NaN
  2    1.0
  dtype: float64 

缺失值函数
~~~~~~~~~~~~

Pandas 提供了一些列用于处理确实值的函数或方法。例如发现缺失值，替换缺失值等。

发现缺失值
```````````````

Pandas 数据结构有两种有效的方法可以发现缺失值：isnull() 和 notnull()。每种方法都返回布尔类型的掩码数据，例如：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  ps = pd.Series([1, np.nan, None])
  print(ps.isnull())
  
  >>>
  0    False
  1     True
  2     True
  dtype: bool
  
  # 与 isnull() 方法相反
  print(ps.notnull())
  
  >>>
  0     True
  1    False
  2    False
  dtype: bool

布尔类型掩码数组可以直接作为 Series 或 DataFrame 的索引使用：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  ps = pd.Series([1, np.nan, 2, None])
  print(ps[ps.notnull()])
  
  >>>
  0    1.0
  2    2.0
  dtype: float64

以上操作同样适用于 DataFrame 对象：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  df = pd.DataFrame([1, np.nan, None])
  print(df.isnull())
  
  >>>
         0
  0  False
  1   True
  2   True

剔除缺失值
`````````````

dropna() 用于剔除缺失值，它返回一个数组副本。在 Series 上使用它非常简单：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 剔除缺失值
  ps = pd.Series([1, np.nan, 2, None])
  print(ps.dropna())
  
  >>>
  0    1.0
  2    2.0
  dtype: float64

由于 Series 是一维的，任何元素是 NaN 都可以直接删除这一元素（相当于一列），而在 DataFrame 上使用它们时需要设置一些参数， 例如：

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  df = pd.DataFrame([[1, np.nan, 2],
                      [2, 3, 5],
                      [np.nan, 4, 6]])
  print(df)
  
  >>>
       0    1  2
  0  1.0  NaN  2
  1  2.0  3.0  5
  2  NaN  4.0  6

无法从 DataFrame 中单独剔除一个值，要么是剔除缺失值所在的整行，要么是整列。根据实际需求，来剔除整行或整列，DataFrame 中的 dropna() 会有一些参数可以配置。
默认情况下， dropna() 会剔除任何包含缺失值的整行数据：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(df.dropna())
  
  >>>
       0    1  2
  1  2.0  3.0  5

可以设置按不同的坐标轴剔除缺失值， 比如 axis=1（或 axis='columns'） 会剔除任何包含缺失值的整列数据：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(df.dropna(axis='columns'))
  
  >>>
     2
  0  2
  1  5
  2  6

这么做也会把非缺失值一并剔除，因为可能有时候只需要剔除全部是缺失值的行或列，或者绝大多数是缺失值的行或列。可以通过设置 how 或 thresh 参数来满足，它们可以设置剔除行或列缺失值的数量阈值。

默认设置是 how='any'， 也就是说只要有缺失值就剔除整行或整列（通过 axis 设置坐标轴）。还可以设置 how='all'， 这样就只会剔除全部是缺失值的行或列了：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  df[3] = np.nan
  print(df)
  
  >>>
       0    1  2   3
  0  1.0  NaN  2 NaN
  1  2.0  3.0  5 NaN
  2  NaN  4.0  6 NaN
  
  df = df.dropna(axis='columns', how='all')
  print(df)
  
  >>>
       0    1  2
  0  1.0  NaN  2
  1  2.0  3.0  5
  2  NaN  4.0  6

还可以通过 thresh 参数设置行或列中非缺失值的最小数量，从而实现更加个性化的配置：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  df = df.dropna(axis='rows', thresh=3)
  print(df)
  
  >>>
       0    1  2   3
  1  2.0  3.0  5 NaN

第 1 行与第 3 行被剔除了， 因为它们只包含两个非缺失值。

填充缺失值
``````````````````

有时可能并不想移除缺失值，而是想把它们替换成有效的数值。 有效的值可能是像 0、 1、 2 那样单独的值，也可能是经过填充（imputation） 或转换（interpolation） 得到的。 虽然你可以通过isnull() 方法建立掩码来填充缺失值， 但是 Pandas 为此专门提供了一个 fillna() 方法， 它将返回填充了缺失值后的数组副本。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  ps = pd.Series([1, np.nan, 2, None], index=list('abcd'))
  print(ps)
  
  >>>
  a    1.0
  b    NaN
  c    2.0
  d    NaN
  dtype: float64

我们将用一个单独的值来填充缺失值， 例如用 -1：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(ps.fillna(-1))
  
  >>>
  a    1.0
  b   -1.0
  c    2.0
  d   -1.0
  dtype: float64

可以用缺失值前面的有效值来从前往后填充（forward-fill）：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(ps.fillna(method='ffill'))
  
  >>>
  a    1.0
  b    1.0
  c    2.0
  d    2.0
  dtype: float64

也可以用缺失值后面的有效值来从后往前填充（back-fill） ：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(ps.fillna(method='bfill'))
  
  a    1.0
  b    2.0
  c    2.0
  d    NaN
  dtype: float64

无论是从前往后还是从后往前，NaN 之后或之前如果都是 NaN 则无法实现填充。

DataFrame 的操作方法与 Series 类似， 只是在填充时需要设置坐标轴参数 axis：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  df = pd.DataFrame([[1, np.nan, 2],
                     [2, 3, 5],
                     [np.nan, np.nan, np.nan]])
  print(df)
  
  >>>
       0    1    2
  0  1.0  NaN  2.0
  1  2.0  3.0  5.0
  2  NaN  NaN  NaN
  
  # 从前向后填充行
  print(df.fillna(method='ffill', axis=1))
  
  >>>
       0    1    2
  0  1.0  1.0  2.0
  1  2.0  3.0  5.0
  2  NaN  NaN  NaN
  
  # 从后向前填充行
  print(df.fillna(method='bfill', axis=1))
  
  >>>
       0    1    2
  0  1.0  2.0  2.0
  1  2.0  3.0  5.0
  2  NaN  NaN  NaN

需要注意的是，假如在从前往后填充时，需要填充的缺失值前面没有值，那么它就仍然是缺失值，这个机制是递归填充。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 从前向后填充列
  print(df.fillna(method='ffill', axis=0))
  
  >>>
       0    1    2
  0  1.0  NaN  2.0
  1  2.0  3.0  5.0
  2  2.0  3.0  5.0
 
  # 从后向前填充列
  print(df.fillna(method='bfill', axis=0))
  
  >>>
       0    1    2
  0  1.0  3.0  2.0
  1  2.0  3.0  5.0
  2  NaN  NaN  NaN

数据加载
--------------

Pandas 提供了丰富的数据加载接口，例如 pd.read_csv，pd.read_json，pd.read_sql 等。

csv 文件数据
~~~~~~~~~~~~

CSV 是逗号分隔值（Comma-Separated Values有时也称为字符分隔值，因为分隔字符也可以不是逗号）的缩写，其文件以纯文本形式存储表格数据（数字和文本）。可以使用记事本直接打开它，或者使用 Excel 打开。

名为 students.csv 的示例文件内容如下：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  age,id,name
  20,100,John
  21,101,Tom
  19,102,Bill

读取数据
`````````````

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 参数 header 默认值为 0，表示以第一行为列索引
  # 等价于 df = pd.read_csv('students.csv')
  df = pd.read_csv('students.csv', header=0)
  print(df)
  
  >>>
     age   id  name
  0   20  100  John
  1   21  101   Tom
  2   19  102  Bill

read_csv() 方法具有非常丰富的参数，常用参数说明如下：

- sep：分隔符，默认是‘,’，CSV文件的分隔符
- header:列名所在 csv 中的行（列索引），默认第一行为列名（默认header=0），header=None 说明第一行不是列名，它会生成新的整数列名。
- names：当 csv 文件没有列名时候，可以用 names 加上要用的列名
- index_col：要用的行名（index），int或sequence或False，默认为 None，即默认添加从 0 开始的 index，若要用第一列作为行索引则 index_col = 0。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # header 为 None，表示 csv 第一行数据作为普通数据  
  df = pd.read_csv('students.csv', header=None)
  print(df)
  
  >>>
       0    1     2
  0  age   id  name
  1   20  100  John
  2   21  101   Tom
  3   19  102  Bill
  
  # 使用 names 指定列名
  df = pd.read_csv('students.csv', header=None, names=['a', 'b', 'c'])
  print(df)
  
       a    b     c
  0  age   id  name
  1   20  100  John
  2   21  101   Tom
  3   19  102  Bill  
  
  # 指定 csv 文件第一列为行名
  df = pd.read_csv('students.csv', header=0, index_col=0)
  print(df)
  
  >>>
        id  name
  age           
  20   100  John
  21   101   Tom
  19   102  Bill 

分块读取
```````````

read_csv() 的 chunksize 参数支持指定每次读取的行数，返回的是一个可迭代的对象 TextFileReader，这对于读取超大文件特别有用：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 每次读取两行
  tfr = pd.read_csv('students.csv', header=0, chunksize=2)
  print(type(tfr).__name__)
  
  >>>
  TextFileReader
  
  for chunk in tfr:
      print('------------------')
      print(chunk)
      
  >>>    
  ------------------
     age   id  name
  0   20  100  John
  1   21  101   Tom
  ------------------
     age   id  name
  2   19  102  Bill

可以看到每次从 TextFileReader 迭代对象读取时都会带上列名。

保存数据
`````````````

to_csv() 用于写出数据到文件，注意 index 参数指明是否写出行信息：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  studentdf = pd.DataFrame({'id': [100,101,102],
                            'name':['John', 'Tom', 'Bill'],
                            'age': [20, 21, 19]})
  studentdf.to_csv('students.csv', index=False)

其他数据类型
~~~~~~~~~~~~

以下列出 Pandas 支持的数据文件类型，以及读取和保存的方法：
  
  ============= ============ ================ ===============
  数据格式类型    描述       读取方法         写出方法
  ============= ============ ================ ===============
  text          CSV          read_csv         to_csv
  text          JSON         read_json        to_json
  text          HTML         read_html        to_html
  text          本地粘贴板   read_clipboard   to_clipboard
  binary        Excel        read_excel       to_excel
  binary        HDF5         read_hdf         to_hdf
  binary        Feather包    read_feather     to_feather
  binary        Parquet      read_parquet     to_parquet
  binary        Msgpack      read_msgpack     to_msgpack
  binary        Stata        read_stata       to_stata
  binary        SAS          read_sas        
  binary        Pickle       read_pickle      to_pickle
  SQL           SQL          read_sql         to_sql
  SQL           谷歌BigQuery read_gbq         to_gbq
  ============= ============ ================ ===============

查看行数据
~~~~~~~~~~~~~

head(n) 方法用于查看从头部开始的 n 行数据： 

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  df = pd.read_csv('students.csv', header=0)
  print(df.head(2))
  
  >>>
     age   id  name
  0   20  100  John
  1   21  101   Tom

tail(n) 方法用于查看尾部的 n 行数据：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(df.tail(2))
  
  >>>
     age   id  name
  1   21  101   Tom
  2   19  102  Bill

当我们对一个很大的数据文件一无所知时，可以打开前几行观察数据的类型，列标签等。

层级索引
--------------

一级索引的 Series 看起来很像一维数组，且是单列数组。DataFrame 可以看做有两个索引的二维数组。

通过层级索引（hierarchical indexing，也被称为多级索引，multi-indexing）配合多个有不同等级（level）的一级索引一起使用，这样就可以将高维数组转换成类似一维 Series 和二维 DataFrame 对象的形式。

多级索引的 Series
~~~~~~~~~~~~~~~~~

多级索引的 Series，索引是一个二维数组，相当于多个索引决定一个值，类似于 DataFrame 的行索引和列索引：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  ps = pd.Series([90, 80, 95, 91, 92, 88], index=[['John'] * 3 + ['Tom'] * 3, 
                  ['Maths', 'English', 'Chemistry'] * 2])
  print(ps)
  
  >>>
  John  Maths        90
        English      80
        Chemistry    95
  Tom   Maths        91
        English      92
        Chemistry    88
  dtype: int64

  print(ps.index)
  
  >>>
  MultiIndex(levels=[['John', 'Tom'], ['Chemistry', 'English', 'Maths']],
           labels=[[0, 0, 0, 1, 1, 1], [2, 1, 0, 2, 1, 0]])

此时的索引类型为 MultiIndex。MultiIndex 里面的 levels 属性表示索引的等级，可以看到 John 和 Tom 处在第一级，各科课程名称为第二级。

labels 标签包含了各个索引等级对应的数据的整数索引。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 使用一级索引查看数据
  print(ps['Tom'])
  
  >>>  
  Maths        91
  English      92
  Chemistry    88
  dtype: int64
  
  # 使用切片查看二级索引 Maths 数据
  print(ps[:, 'Maths'])
  
  >>>
  John    90
  Tom     91
  dtype: int64

多级索引 Series 转 DataFrame
````````````````````````````

unstack() 方法可以快速将一个多级索引的 Series 转化为普通索引的 DataFrame：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  df = ps.unstack()
  print(type(df).__name__)
  
  >>>
  DataFrame
  
  print(df)
  
  >>>
        Chemistry  English  Maths
  John         95       80     90
  Tom          88       92     91

stack() 方法实现相反的转换：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  print(df.stack())
  
  >>>
  John  Chemistry    95
        English      80
        Maths        90
  Tom   Chemistry    88
        English      92
        Maths        91
  dtype: int64

增加索引层级
`````````````

如果我们可以用含多级索引的一维 Series 数据表示二维数据，那么我们就可以用 Series 或 DataFrame 表示三维甚至更高维度的数据。 多级索引每增加一级，就表示数据增加一维， 利用这一特点就可以轻松表示任意维度的数据了。

假如上面示例中的学生成绩是 2012 年数据，我们要添加 2013 年的数据，只需要增加一个新的索引层级即可：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  newps = pd.DataFrame({'2012': ps, '2013': [98,87,93, 90,91,84]})
  print(newps)
  
  >>>
                  2012  2013
  John Maths        90    98
       English      80    87
       Chemistry    95    93
  Tom  Maths        91    90
       English      92    91
       Chemistry    88    84

当然我们可以使用 stack() 转化为 Series 类型：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(newps.stack())

  >>>
  John  Maths      2012    90
                   2013    98
        English    2012    80
                   2013    87
        Chemistry  2012    95
                   2013    93
  Tom   Maths      2012    91
                   2013    90
        English    2012    92
                   2013    91
        Chemistry  2012    88
                   2013    84    

这一实现效果令人惊喜。求取各科平均成绩非常简单：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 求取两年各科平均成绩
  average = (newps['2013'] + newps['2012']) / 2
  print(average.unstack())
  
  >>>
        Chemistry  English  Maths
  John       94.0     83.5   94.0
  Tom        86.0     91.5   90.5

创建多级索引
~~~~~~~~~~~~~

有多种方式创建多级索引 MultiIndex 对象：

- MultiIndex.from_arrays 转换由 arrays 组成的 list 为 MultiIndex
- MultiIndex.from_tuples 转换元组为 MultiIndex
- MultiIndex.from_product 由迭代对象的笛卡尔积生成 MultiIndex

array 转多级索引
`````````````````

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # names 指明每个层级的名称
  arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
  pm = pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
  print(pm)
  
  >>>
  MultiIndex(levels=[[1, 2], ['blue', 'red']],
             labels=[[0, 0, 1, 1], [1, 0, 1, 0]],
             names=['number', 'color'])
  
  # 查看多级索引的属性
  print(pm.levels)
  print(pm.labels)
  print(pm.names)
  
  >>>
  [[1, 2], ['blue', 'red']]
  [[0, 0, 1, 1], [1, 0, 1, 0]]
  ['number', 'color']

元组转多级索引
`````````````````

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  tuples = [(1, 'red'), (1, 'blue'),(2, 'red'), (2, 'blue')]
  pm = pd.MultiIndex.from_tuples(tuples, names=('number', 'color'))
  print(pm)
  
  >>>
  MultiIndex(levels=[[1, 2], ['blue', 'red']],
             labels=[[0, 0, 1, 1], [1, 0, 1, 0]],
             names=['number', 'color'])

笛卡尔积转多级索引
```````````````````

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  numbers = [0, 1, 2]
  colors = ['green', 'purple']
  pm = pd.MultiIndex.from_product([numbers, colors], names=['number', 'color'])
  print(pm)
  
  >>>
  MultiIndex(levels=[[0, 1, 2], ['green', 'purple']],
             labels=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
             names=['number', 'color'])

使用笛卡尔积方式创建 MultiIndex 对象，层次是比较清晰的。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 增加模拟数据，查看层级和层级名称
  pf = pd.DataFrame(np.arange(6), index=pm)
  print(pf)
  
  >>>
  number color    
  0      green   0
         purple  1
  1      green   2
         purple  3
  2      green   4
         purple  5

多级列索引
``````````

上面的示例均是创建多级行索引，当然也可以创建多级列索引。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  students = ['John', 'Tom']
  subjects = ['Maths', 'English', 'Chemistry']

  # 创建多级行索引  
  index = pd.MultiIndex.from_product([students, subjects], names=['student', 'subject'])
  
  # 创建多级列索引
  columns = pd.MultiIndex.from_product([['2012', '2013'], ['first_half', 'latter_half']],
                                        names=['year', 'half'])
  
  # 模拟成绩数据，一共是 6 行 4 列
  pf = pd.DataFrame(99 - np.random.randint(20, size=(6, 4)), index=index, columns=columns)
  print(pf)
  
  >>>
  year                    2012                   2013            
  half              first_half latter_half first_half latter_half
  student subject                                                
  John    Maths             82          81         86          81
          English           83          81         88          94
          Chemistry         81          85         85          81
  Tom     Maths             97          80         95          82
          English           89          99         94          92
          Chemistry         97          92         92          84

有上例可以看出多级行列索引的创建非常简单。我们可以方便查看各级索引的数据：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 查询 2012 年上半年成绩数据
  print(pf['2012','first_half'].unstack())
  
  subject  Chemistry  English  Maths
  student                           
  John            89       88     88
  Tom             83       84     99

如果想获取包含多种标签的数据，需要通过对多个维度（姓名、科目等标签）的多次查询才能实现，这时使用多级行列索引进行查询会非常方便。

多级索引排序和切片
~~~~~~~~~~~~~~~~~~~

Series多级索引排序
``````````````````

和单级索引一样，多级索引顺序是按照声明顺序确定的，也即是无序的，如果按照索引字母顺序排序，将方便切片操作：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  students = ['Tom', 'John']
  subjects = ['Maths', 'English', 'Chemistry']
  index = pd.MultiIndex.from_product([students, subjects], names=['student', 'subject'])
  
  ps = pd.Series(np.arange(6) + 90, index=index)
  print(ps)
  
  >>>
  student  subject  
  Tom      Maths        90
           English      91
           Chemistry    92
  John     Maths        93
           English      94
           Chemistry    95
  dtype: int32

可以看到，默认的索引顺序和声明中索引顺序相同，但是使用切片 [start:end:step] 操作时，start 要小于 end，否则返回空对象，如果是乱序的，我们每次切片时都要记住声明的标签顺序，且声明顺序一旦更改，切片相关的代码就要更新，如果对索引进行排序，就不会再出现这类问题：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  # 默认使用 level=0 排序
  ps = ps.sort_index()
  print(ps)
  
  >>>
  student  subject  
  John     Chemistry    95
           English      94
           Maths        93
  Tom      Chemistry    92
           English      91
           Maths        90
  dtype: int32

经过排序后，可以发现第一级索引和第二季索引都被更新了。 可以使用 level 参数指定优先进行排序的索引层：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 使用 subject 索引排序
  print(ps.sort_index(level=1))
  
  >>>
  student  subject  
  John     Chemistry    95
  Tom      Chemistry    92
  John     English      94
  Tom      English      91
  John     Maths        93
  Tom      Maths        90
  dtype: int32  

逆序排序
```````````````

sort_index() 方法的 ascending 参数可以指定升序或者降序排列，例如 ascending = False 将降序排列：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(ps.sort_index(ascending=False))
  
  >>>
  student  subject  
  Tom      Maths        90
           English      91
           Chemistry    92
  John     Maths        93
           English      94
           Chemistry    95
  dtype: int32

Series多级索引访问
````````````````````

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 使用已排序数据
  print(ps)
  
  >>>
  student  subject  
  John     Chemistry    95
           English      94
           Maths        93
  Tom      Chemistry    92
           English      91
           Maths        90
  dtype: int32  
  
  # 直接通过访问属性方式访问
  print(ps.Tom.Maths)
  
  >>>
  90

  # 通过数组访问方式，这类似于 2 维的 DataFrame 访问方式
  print(ps.loc['Tom', 'Maths']) 
  
  >>>
  90
  
  # 切片方式访问
  print(ps.loc[:, 'Maths'])
  
  >>>
  student
  John    93
  Tom     90
  dtype: int32
  
  # 二级切片索引访问
  print(ps.loc['Tom', 'Chemistry':'English'])
  
  >>>
  student  subject  
  Tom      Chemistry    92
           English      91
  dtype: int32  
  
使用 loc 或者 iloc 属性进行多级索引操作，应该为所有层级指定索引，例如 ps.loc[:, :]，而不是 ps.loc[:]。

排序直接修改
````````````````````

以上示例排序结果不对对象直接修改，如果需要结果直接作用在排序对象上，可以传入 inplace = True，此时无返回值。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  ps.sort_index(level=1, inplace=True)
  print(ps)
  
  >>>
  
  student  subject  
  John     Chemistry    95
  Tom      Chemistry    92
  John     English      94
  Tom      English      91
  John     Maths        93
  Tom      Maths        90
  dtype: int32

DataFrame 多级索引排序
```````````````````````

与 Series 对象类似，DataFrame 同样支持多级索引的排序，唯一不同点在于它有行索引和列索引，可以接受 axis 参数：

- axis = 0，对行索引进行排序，Series 只有行索引，所以 axis 永远为 0.
- axis = 1，对列索引进行排序。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  students = ['Tom', 'John']
  subjects = ['Maths', 'English', 'Chemistry']  
  columns = pd.MultiIndex.from_product([['2013', '2012'], ['first_half', 'latter_half']],
                                        names=['year', 'half'])
  
  # 使用固定的模拟成绩数据，以观察排序影响
  pf = pd.DataFrame(75 + np.arange(24).reshape(6, 4), index=index, columns=columns)
  print(pf)
  
  >>>
  year                     2013                   2012           
  half              latter_half first_half latter_half first_half
  student subject                                                
  Tom     Maths              75         76          77         78
          English            79         80          81         82
          Chemistry          83         84          85         86
  John    Maths              87         88          89         90
          English            91         92          93         94
          Chemistry          95         96          97         98

我们使用上面的示例数据，为了查看排序效果，我们把所有索引标签的顺序都颠倒了。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 行标签排序
  print(pf.sort_index(axis=0))

  >>>
  year                     2013                   2012           
  half              latter_half first_half latter_half first_half
  student subject                                                
  John    Chemistry          95         96          97         98
          English            91         92          93         94
          Maths              87         88          89         90
  Tom     Chemistry          83         84          85         86
          English            79         80          81         82
          Maths              75         76          77         78

行标签排序后，对列标签顺序无影响，同样列标签排序对行标签顺序无影响：
 
.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 列标签排序  
  print(pf.sort_index(axis=1))
  
  >>>
  year                    2012                   2013            
  half              first_half latter_half first_half latter_half
  student subject                                                
  Tom     Maths             78          77         76          75
          English           82          81         80          79
          Chemistry         86          85         84          83
  John    Maths             90          89         88          87
          English           94          93         92          91
          Chemistry         98          97         96          95
  
  # 同时对行和列排序
  sorted_pf = pf.sort_index(axis=0).sort_index(axis=1)
  print(sorted_pf)
  
  >>>
  year                    2012                   2013            
  half              first_half latter_half first_half latter_half
  student subject                                                
  John    Chemistry         98          97         96          95
          English           94          93         92          91
          Maths             90          89         88          87
  Tom     Chemistry         86          85         84          83
          English           82          81         80          79
          Maths             78          77         76          75

无论是行排序还是列排序，均对行或列的所有层级标签依次进行了排序，我们当然可以使用 level 指定优先排序的索引层：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 优先使用 subject 排序
  print(pf.sort_index(axis=0, level=1))

  >>>
  year                     2013                   2012           
  half              latter_half first_half latter_half first_half
  student subject                                                
  John    Chemistry          95         96          97         98
  Tom     Chemistry          83         84          85         86
  John    English            91         92          93         94
  Tom     English            79         80          81         82
  John    Maths              87         88          89         90
  Tom     Maths              75         76          77         78
  
  # # 优先使用半学期 half 排序        
  print(pf.sort_index(axis=1, level=1))
  
  >>>
  year                    2012       2013        2012        2013
  half              first_half first_half latter_half latter_half
  student subject                                                
  Tom     Maths             78         76          77          75
          English           82         80          81          79
          Chemistry         86         84          85          83
  John    Maths             90         88          89          87
          English           94         92          93          91
          Chemistry         98         96          97          95

DataFrame多级索引访问
`````````````````````````

DataFrame 不支持属性访问方式。所以通常使用 loc 显式索引方式访问：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 使用已排序数据
  pf.sort_index(inplace=True)
  print(pf)
  
  >>>
  year                     2013                   2012           
  half              latter_half first_half latter_half first_half
  student subject                                                
  John    Chemistry          95         96          97         98
          English            91         92          93         94
          Maths              87         88          89         90
  Tom     Chemistry          83         84          85         86
          English            79         80          81         82
          Maths              75         76          77         78
  
  # 访问 Tom 的各科成绩
  print(pf.loc['Tom', :])
  
  >>>
  year             2013                   2012           
  half      latter_half first_half latter_half first_half
  subject                                                
  Chemistry          83         84          85         86
  English            79         80          81         82
  Maths              75         76          77         78
  
  # 查看 Tom 的 2012 年各科成绩
  print(pf.loc['Tom', :]['2012'])
  
  >>>  
    half       latter_half  first_half
  subject                           
  Chemistry           85          86
  English             81          82
  Maths               77          78
  
  # 查看 Tom 的 2012 年下半年各科成绩
  # 等价于 print(pf.loc['Tom', '2012']['latter_half'])
  print(pf.loc['Tom', :]['2012']['latter_half'])
  
  >>>
    subject
  Chemistry    85
  English      81
  Maths        77
  Name: latter_half, dtype: int32

注意体会 DataFrame 的数组访问方式，第一维索引的形式有几种：

- ['Tom', '2012']：指定行索引和列索引
- ['Tom', 'Maths']：均指定行索引
- ['Tom', :]：均指定行索引

显然第一维索引无法指定的列索引的第二层索引，就要通过增加第二维索引来访问，例如 ['Tom', '2012']['latter_half']。

索引调整和重置
~~~~~~~~~~~~~~~

我们可以通过对象的 index 和 columns 属性更新行或列标签。也可以调整索引的顺序。

reindex
``````````````

reindex() 是 pandas 对象的一个重要方法，其作用是在当前对象基础上创建一个新索引的新对象。它通常和 set_index() 配合使用：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  students = ['Tom', 'John']
  subjects = ['Maths', 'English', 'Chemistry']
  index = pd.MultiIndex.from_product([students, subjects], names=['student', 'subject'])
  ps = pd.Series(np.arange(6) + 90, index=index)
  
  # 重制索引
  flat_ps = ps.reset_index() 
  print(flat_ps)

  >>>
    student    subject   0
  0     Tom      Maths  90
  1     Tom    English  91
  2     Tom  Chemistry  92
  3    John      Maths  93
  4    John    English  94
  5    John  Chemistry  95  

  # 逆向转换
  print(flat_ps.set_index(['student','subject']))
  >>>
  
  student subject      
  Tom     Maths      90
          English    91
          Chemistry  92
  John    Maths      93
          English    94
          Chemistry  95
  
可以传入 drop = True 丢弃所有索引，此时变为 Series 对象，只保留数据：

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  flat_ps = ps.reset_index(drop = True) 
  print(flat_ps)
  
  >>>
  0    90
  1    91
  2    92
  3    93
  4    94
  5    95
  dtype: int32

多级索引数据统计
~~~~~~~~~~~~~~~~~

前面已经介绍过一些 Pandas 自带的数据累计方法，比如 mean()、sum() 和 max()。而对于层级索引数据，可以设置参数 level 实现对数据子集的累计操作。

首先我们准备如下用于统计的带有多级索引的数据：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # Series 类型的数据
  student  subject         # 行标签名
  Tom      Maths        90
           English      91
           Chemistry    92
  John     Maths        93
           English      94
           Chemistry    95
  dtype: int32
  
  print(ps.index)
  
  >>>
  MultiIndex(levels=[['John', 'Tom'], ['Chemistry', 'English', 'Maths']],
             labels=[[1, 1, 1, 0, 0, 0], [2, 1, 0, 2, 1, 0]],
             names=['student', 'subject'])
           
可以使用 level 指定要统计的行标签名或者整数索引，来进行统计：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  print(ps.mean(level='student')) # 等价于 ps.mean(level=0)
  
  >>>
  student
  John    94  # (93+94+95) / 3 = 94
  Tom     91  # (90+91+92) / 3 = 91
  dtype: int32

结合 axis 参数， 就可以对 DataFrame 列索引进行类似的统计操作：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 准备以下 DataFrame 数据
  year                     2013                   2012           
  half              latter_half first_half latter_half first_half
  student subject                                                
  John    Chemistry          95         96          97         98
          English            91         92          93         94
          Maths              87         88          89         90
  Tom     Chemistry          83         84          85         86
          English            79         80          81         82
          Maths              75         76          77         78   

  # 查看多级索引信息
  print(pf.index)
  
  >>>
  MultiIndex(levels=[['John', 'Tom'], ['Chemistry', 'English', 'Maths']],
             labels=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
             names=['student', 'subject'])

  # 获取每个学生的平均成绩，axis=0 指定对行统计
  print(pf.mean(level='student', axis=0))
  
  >>>
  year           2013                   2012           
  half    latter_half first_half latter_half first_half
  student                                              
  John             91         92          93         94
  Tom              79         80          81         82

  # 获取年平均成绩，axis=1 指定对列统计
  print(pf.mean(level='year', axis=1))
  
  >>>
  year               2012  2013
  student subject              
  John    Chemistry  97.5  95.5
          English    93.5  91.5
          Maths      89.5  87.5
  Tom     Chemistry  85.5  83.5
          English    81.5  79.5
          Maths      77.5  75.5

