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

Series 对象是一个带索引数据构成的一维数组。 可以使用 list 作为参数，来生成对应 Series 对象，例如：

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
  print(sdata.dtype)
  
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

字典是一种将任意键映射到一组任意值的数据结构，而 Series 对象是一种将类型键映射到一组类型值的数据结构。Pandas Series 的类型信息使得它在某些操作上比 Python 的字典更高效。

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

DataFrame 对象
~~~~~~~~~~~~~~~~

如果将 Series 类比为带索引的一维数组， 那么 DataFrame 就可以看作是一种既有行索引， 又有列名的二维数组。

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
  
从示例中可以看出 pd.DataFrame 指定每列信息，它是一个指定列名的 Series 对象。它是一组 Series 的集合。

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
   
  d = pd.DataFrame(narray,
                   columns = ['foo', 'bar'],
                   index=['a', 'b', 'c'])
  print(d)
  
  >>>
     foo  bar
  a    2    0
  b    2    2
  c    2    1

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

将Series看作字典，和字典一样， Series 对象提供了键值对的映射：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 使用 in 或者 not in 判断键是否存在
  sdata = pd.Series([1, 2, 3.14], index=['num1', 'num2', 'pi'])
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

Series 不仅有着和字典一样的接口， 而且还具备和 NumPy 数组一样的数组数据选择功能， 包括索引、 掩码、 花哨的索引等操作，例如：

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

索引器
``````````````````

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
