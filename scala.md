

# Scala

## 学习资源

https://www.journaldev.com/7444/introduction-to-scala-programming-language

https://www.javatpoint.com/scala-tutorial

https://alvinalexander.com/scala/



## 简介

C ：面向过程。

C++，Java：面向对象。

Shell/Javascript/Python：解释性函数式编程语言。

------

面向对象：一切皆类和对象

函数式编程语言：以函数/方法为核心。

------

动态编程语言：定义变量时无需指定类型，动态确定

静态编程语言：需定义变量类型

------

编译型：C，C++，Java。

解释型（脚本）：Shell， Javascript，Python。

Scala: **动态类型的编译型的函数式和面向对象相结合的高编程语言**。

.scala->scalac->.class->运行在JVM上。Spark使用 scala 开发。



面向对象和函数范式编程语言相似度，从左向右：

java(oop)->kotlin(oop+minor fp)->scala(oop+fp)->clojure(fp+minor oop)->haskell(fp)

## 环境安装

1.JDK

2.Scala

3.IDEA IDE

4.Scala开发插件

## 基础语法

### 基本特征

- 句尾无需指定';'。

  类成员必须设置默认值（为何不像Java自动初始化？）。

  类成员默认为 public，一个文件中可以有多个 public类。

  无参方法，可以不适用括号调用，这与访问属性（字段）造成混乱。

  访问数组元素使用(index)，而不是[index]。

#### 术语障碍

将会遭遇大量的术语障碍（Terminology Barrier），包括

nomenclature：科学术语

terminology：术语 

abbreviation：缩写

acronym：首字母缩写

Naming conventions：命名规范



#### 分号推断

句尾无需指定';'，如果一行有多条语句，需要 ';' 分割。这在书写多行代码时要注意不合适的换行位置，可能导致副作用，例如以下语句被当做两条语句：

```scala
x
+y
```

解决方法可以加圆括号，或者将操作符放在句尾：

```scala
（x
+y）

// 或者
x+
y
```

#### 神奇的句法

真正的一切皆对象，所有对象均有相应的方法，链式调用方法：

```
scala> 3.1415926.toDouble.toInt
res36: Int = 3
```

#### 无操作符

Scala中没有传统意义上的操作符，类似 +，-，*，/ 这样的字符可以被作为方法名调用，例如：

```scala
scala> (1).+(2)
res62: Int = 3
```

上例中 1 就是值为 1 的 Int 对象，+ 是 Int 对象的方法，2 是将 Int  对象 2 作为参数传入 + 方法中。

Scala 中所有操作都是方法调用。所以数组访问也是通过方法，而使用()，不使用 []。数组只是类的实例，访问时调用 apply 工厂方法；赋值时则调用 update 方法：

```scala
scala> val a = new Array[Int](2)
a: Array[Int] = Array(0, 0)

// 以下两种访问方式等价
scala> a(0)
res68: Int = 0

scala> a.apply(0)
res69: Int = 0

// 以下两种赋值方法等价
scala> a(0) = 1
scala> a.update(0, 1)

scala> a
res72: Array[Int] = Array(1, 0)
```

实际上数组初始化调用的是 Array 类的工厂方法 apply（定义在 Array 的伴生对象）中。

```scala
// 以下两种方式等价
scala> val a = Array[Int](2,3)
a: Array[Int] = Array(2, 3)

scala> val a = Array.apply(2, 3)
a: Array[Int] = Array(2, 3)
```

### 特殊符号汇总

| 符号    | 示例代码                                                     | 意义                                                         |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ->      | val map = Map("key1" -> 1,"key2" -> 2)                       | 定义映射                                                     |
| <-      | for(i <- "string")println(i)                                 | 遍历可遍历对象                                               |
| =>      | foreach(i => println(i))                                     | 定义匿名函数                                                 |
| ::      | 1 :: list1 或                                val b = "x" :: "y" :: "z" :: Nil | 列表扩展元素                                                 |
| :::     | list1 ::: list2                                              | 拼接列表对象                                                 |
| _N      | tuple._1                                                     | 访问元组元素，N 从1开始， 元组中的元素类型可以不同 ，所以不能使用统一的 () 运算符（apply函数） |
| :+和+:  | val ac = 1 +: 2.2 +: 'f +: Nil             val ca = Nil :+ "scala" :+ 2 :+ true | **:+ 和 +:**  其中  `:+` 方法用于在尾部追加元素， `+:` 方法用于在头部追加元素 ** |
| ++      | "scala" ++ "java"                                            | 类似:::，++ 方法除了连接集合还可以连接字符串                 |
| /: , :\ |                                                              | 分别对应 foldLeft 和 foldRight 函数                          |
| ==      | a == true                                                    | 调用对象 equals 方法                                         |

### ()与{}

scala里面大小括号并不是一回事儿，虽然说很多时候**看起来可以替换**。大括号之所以在小括号的地方能使用，是因为该小括号仅需一个参数，故小括号可以省略，而大括号的内容最后会被推演为一个结果值，并作为小括号的参数给予传递。两个参数的小括号就无法直接用大括号替代。

二者还是通常意义上的定位和用法，只不过scala有一套省略策略。

**专用大括号的场景，此时不能使用小括号**：

1. 大括号{}用于代码块，定义新的作用域，计算结果是代码最后一行；
2. 大括号用于 for 语句中，可以换行书写每一部分：生成器，定义和过滤器(守卫)，换行符取代分号，代码更清晰。
3. 大括号{}用于定义普通函数或者匿名函数； 

```scala
def function():Unit = {           // 定义函数  
    val s = {"Hello" + " world!"} // 代码块，一行代码时，可以省略
    println(s)
}

def function()= println("Hello world!") // 一行代码时省略大括号

// 定义匿名函数，此处省略了参数类型的定义
// scala> val f = (_:Any) => {print(_)}
scala> val f = {print(_)}
f: Any => Unit = $$Lambda$1748/618436185@5da7b947

scala> f("123")
123
// 只有一行可省略大括号
scala> val f = print(_)
f: Any => Unit = $$Lambda$1749/1710639062@5c3daa21
```

3. 在Scala中，被“{}”包含的case语句可以被看成是一个函数字面量定义的偏函数，它可以被用在任何普通的函数字面量适用的地方。  

  ```scala
// case 定义偏函数时大括号不可省略
val tupleList = List[(String, String)]()
val filtered = tupleList.takeWhile{case (s1, s2) => s1 == s2 }
  ```

**看起来混用的场景**：某些高阶函数需要函数作为参数，此时要注意**是偏函数还是匿名函数的使用**，匿名函数看起来总是使用小括号，而偏函数总是使用大括号。例如：

```scala
scala> val list = List(0,1,2,3,4)
list: List[Int] = List(0, 1, 2, 3, 4)

// 使用匿名函数，当只有一条语句时，可以省略大括号
scala> list.filter(_ > 3)
res106: List[Int] = List(4)

scala> list.filter({_ > 3}) // 未省略大括号
res107: List[Int] = List(4)

// 使用 case 定义的偏函数，当函数只有一个参数时可以省略小括号
scala> list.filter{case x => x > 3}
res113: List[Int] = List(4)

scala> list.filter({case x => x > 3}) // 未省略小括号
res114: List[Int] = List(4)
```

注意 case 语句定义的字面量函数将会根据传入函数的参数声明，自动编译转换为偏函数，或者匿名函数：

```scala
// map 参数是匿名函数，所以这里再遇到 "seven" 会报错
List(1, 3, 5, "seven") map { case i: Int => i + 1 } // won't work 

// collect 是偏函数
List(1, 3, 5, "seven") collect { case i: Int => i + 1 } 
```

偏函数只对会作用于指定类型的参数或指定范围值的参数实施计算，超出它的界定范围之外的参数类型和值它会忽略。就像上面例子中一样，case i: Int => i + 1只声明了对Int参数的处理，在遇到”seven”元素时，不在偏函数的适用范围内，所以这个元素被过滤掉了。

case 语句定义的匿名函数还可以用来解析嵌套变量类型：

````scala
scala> val list = List((1,2),(3,4))
list: List[(Int, Int)] = List((1,2), (3,4))

// foreach 接收普通函数做参数，通过 _N 访问元组元素并不优雅
scala> list.foreach(x => println((x._1-x._2)*(x._1+x._2)))
-3
-7

//一种改进方式，函数体内部解析单个元组给 x,y 变量
scala> list.foreach(tuple => {val(x,y) = tuple; println((x-y)*(x+y))})

// case 语句被解析为普通函数，这种方式更清晰
scala> list.foreach{case (x,y) => println((x-y)*(x+y))}
````

for 语句也可用于解析复合类型：

```scala
for((x,y) <- list)println((x-y)*(x+y))
```

不过以上方式均不是推荐的方式，推荐方式采用函数式编程的map 函数：

```scala
scala> val result = list.map{case (x,y)=>(x-y)*(x+y)}
result: List[Int] = List(-3, -7)
```

### 字符串插值

Scala 提供了丰富的字符串插值，作用类似于 shell 脚本：

```scala
// s 计算表达式
scala> s"Result is ${1*2 + 3}"
res62: String = Result is 5

// s 前缀引用变量名，$res62 等价于 ${res62}
scala> s"Helo $res62"
res63: String = Helo Result is 5

// 原始输出
scala> raw"\\\\"
res64: String = \\\\

// f 格式化显示
scala> f"${math.Pi}%.5f"
res65: String = 3.14159
```

### 定义变量

Java 中定义变量时必须声明类型：

```java
int a = 123;
```

Scala 中可以指定类型，也可以不指定，此时会根据赋的值（准确说是初始化表达式的值）进行动态推断：

```scala
scala> var a:String = "123" // 指定类型
a: String = 123

scala> var a = "123"        // 自动推断类型
a: String = 123

scala> var a,b:Int = 5      // 定义多个变量
a: Int = 5
b: Int = 5

// 定义多个不同类型的变量
scala> var (a,b):(Int,String) = (5,"123")
a: Int = 5
b: String = 123
```

一旦变量类型确定，不可以再改变变量的类型，且必须赋初值，所以不能进行声明操作。

var (variable) 用于声明变量。

### 不可变变量

val (value) 用于声明不可变变量（Immutable variables），类似于Java中的 final 修饰符。不可变变量具有一旦初始化，不可再改变之意。注意在 Scala 中没有常量（constant）的概念，所以全局定义的不可变变量可以当做常量使用。

```scala
val PI = 3.14
val Int:PI = 3.14

// a 不可再引用其他数组，但是数组内容是可变的
val a = Array[Int](10)
```

scala 编程语言中，推荐使用 val，不推荐使用 var：

- 明确哪些量不可变，将减少代码错误，同时使得逻辑更清晰，实际上程序中的变量并没有想象的需要那么多
- 增强可读性，可以被垃圾回收机制快速回收
- 线程安全，适合并发编程

#### 延迟初始化常量

lazy 关键字用于定义延迟初始化常量：

```scala
lazy val a:Int = 123 + 456
```

延迟初始化常量只有在使用时才被赋初值。

var 类型不可用 lazy 修饰。

##数据类型

###基础类型

Scala 基础数据类型类似 Java，共有 9 种：

- 内置有 7 种数值类型（numeric type）：包括 5 种整数类型（intergal type）Byte、Char、Short、Int、Long 和 2 种浮点类型 Float 和Double。 
- 两个非数值类型 Boolean 和 String（String 类型来自 java.lang）。
- 和 Java 不同，Scala 没有基本类型（primitive type），Scala是纯面型对象的。

以上9种基础类型均使用字面量（Literal）来赋值，字面量是在代码中直接写入常量值的一种方式。

###字面量

字面量（literal）是对值的字面表示方式：例如 10，0x0a，true，"hello, world"，用于给变量或常量进行赋值。所以函数字面量（function literal）就是指函数名（函数变量）对应的函数值，也即函数体中的代码。

####整数字面量

整数字面量有两种形式：

1. 十进制：以非 0 开头的 0-9 数字
2. 16进制：以 0x 或者 0X 开头的 0-9和A-F（大小写不敏感）表示的数字

```scala
scala> var a = 10
a: Int = 10

scala> var b = 0xab
b: Int = 171

scala> var c = 0xAB
c: Int = 171
```

Scala 不支持 8 进制。如果整数字面量以 L 或者 l 结尾，则是  Long 型的，否则为 Int 型。Int 型的字面量可以赋值给 Short 和 Byte 类型（值在对应类型的合法区间）。

```scala
scala> var c = 0xABL
c: Long = 171

// Int 字面量赋值给 Byte 类型
scala> var c:Byte = 100
c: Byte = 100

scala> var c:Byte = 1000
<console>:14: error: type mismatch;
 found   : Int(1000)
 required: Byte
       var c:Byte = 1000
```

####浮点字面量

浮点数字面量由十进制数字，可选的小数点和可选的 E 或者 e 组成：

```scala
scala> var f = 1.0
f: Double = 1.0

scala> var f = 1e1
f: Double = 10.0

scala> var f = .1e1
f: Double = 1.0
```

如果浮点字面量以 F 或 f 记为，则它是 Float型的，默认为 Double 类型。

```scala
scala> var f=.1e1f
f: Float = 1.0
```

####字符字面量

字符字面量由一对单引号和中间的任意 Unicode 字符组成，例如：

```scala
scala> var c = 'a'
c: Char = a

scala> var c = 'A'
c: Char = A

scala> var c = '\u0042'
c: Char = B
```

一些字符字面量由特殊的转义字符序列表示：

```scala
scala> var c = '\\'
c: Char = \
```

#### 字符串字面量

由双引号引起来的字符组成：

```scala
scala> var str = "str\"123"
str: String = str"123
```

如果字符串包含许多需要转义的字符，可以使用三双引号来表示字符串的开始和结束，这种字符串被称为原生字符串（raw string）。其中的任何字符不会被转义处理。

```scala
scala> var str = """\\\hello"""""
str: String = \\\hello""
```

中间不可以包含三空格，否则处理可能出错。如果字符串较长可以分多行书写：

```scala
var str = """hello """ +
		 """world"""
println(str)
```

#### 布尔字面量

类型 Boolean 有两个字面量：true 和 false。

###引用类型

![](./types.png)

Any是所有类型的超类型，也称为顶级类型。它定义了一些通用的方法如equals、hashCode和toString。Any有两个直接子类：AnyVal和AnyRef。

AnyVal代表值类型。有9个预定义的非空的值类型分别是：Double、Float、Long、Int、Short、Byte、Char、Unit和Boolean。Unit是不带任何意义的值类型，它仅有一个实例可以像这样声明：()。所有的函数必须有返回，所以说有时候Unit也是有用的返回类型。

AnyRef代表引用类型。所有非值类型都被定义为引用类型。在Scala中，每个用户自定义的类型都是AnyRef的子类型。如果Scala被应用在Java的运行环境中，AnyRef相当于java.lang.Object。

#### Option,None和Some

None和Some类型均是 Option 的子类，

```scala
def toInt(in: String): Option[Int] = {
    try {
        Some(Integer.parseInt(in.trim))
    } catch {
        case e: NumberFormatException => None
    }
}

// 使用 toInt
toInt("123") match {
    case Some(i) => println(i)
    case None => println("That didn't work.")
}
```

示例 toInt 尝试将字符串转换为整型，如果字符串无法转换则返回 None类型，否则返回 Some 类型。这样就避免了返回 Null 类型的常量 null，而导致代码异常。

```scala
val bag = List("1", "2", "foo", "3", "bar")
val sum = bag.flatMap(toInt).sum
```

大部分函数，例如 flatMap 可以很好地直接处理 None 类型。

```scala
// flatMap 等价于 map + flatten 操作
scala> bag.map(toInt)
res42: List[Option[Int]] = List(Some(1), Some(2), None, Some(3), None)

scala> bag.map(toInt).flatten
res43: List[Int] = List(1, 2, 3)
```

获取 Option 类型的值常用 getOrElse，这样可以在解析出错时提供一个默认值：

```scala
val x = toInt("1").getOrElse(0)
```

尽管 Option 类型提供了 get 方法，但是 None 类型不支持 get 方法，会导致 NoSuchElementException 异常。

```scala
case object None extends Option[Nothing] {
  def isEmpty = true
  def get = throw new NoSuchElementException("None.get")
}
```



### 隐式类型转换



![1559700420030](convert.png)

值类型可以按照上面的方向进行自动转换，是单向的。

```scala
val x: Long = 987654321
val y: Float = x 
val face: Char = '☺'
val number: Int = face
```

逆向转换是不允许的：

```scala
scala> a
res22: Int = 1

scala> a = 1.1
<console>:12: error: type mismatch;
 found   : Double(1.1)
 required: Int
       a = 1.1

// 需要类型显式转换
scala> a = 1.1.toInt
a: Int = 1
```

Scala 的类型转换均是通过对象方法进行，使用 1.1.toType，而不是 (Int)1.1 的形式。 

数字和字符串相加会进行隐式转换（Implicit Type Conversion）：

```scala
scala> 1 + " string"
res23: String = 1 string
```

这在 Python 中是不被允许的，必须显式转换（Explicit Type Conversion），格式为 str(1) + " string"。

### 官网文档错误

1. https://docs.scala-lang.org/zh-cn/tour/basics.html

方法：注意返回类型是怎么在函数列表和一个冒号`: Int`之后声明的。 

注意返回类型`Int`是怎么在函数列表和一个冒号`:`之后声明的。 

1. https://docs.scala-lang.org/zh-cn/tour/classes.html

类定义

一个最简的类的定义就是关键字`class`+标识符，类名必须是大写。

类名首字母应大写。

1. https://docs.scala-lang.org/zh-cn/tour/classes.html

使用`private`访问修饰符可以在函数外部隐藏它们。 

使用`private`访问修饰符可以在类外部隐藏它们。 

## 控制结构

Scala 中的所有控制结构都会返回某种值作为结果，这式函数式编程语言采取的策略，程序被认为是用来计算出某个值，因此程序的各个组成部分也应该计算出某个值。

while 和 do-while 被称为“循环”，而不是表达式，因为它们总是返回类型 Unit 类型的值()。

### if 表达式

Scala 中的 if/else 表达式具有值，也即各分支最后一条表达式的值：

```scala
scala> var num = 10
num: Int = 10

scala> val cmp = if(num > 0) 1 else 0
cmp: Int = 1
```

如果两个分支中的值类型不同，则返回混合类型 Any。

```scala
scala> num = -1
num: Int = -1

scala> val cmp = if(num > 0) 1 else "less"
cmp: Any = less
```

如果缺少 else 分支，则该分支返回 Unit 类型，它的值总是 ()，可以看成是“没有值”的占位符。

```scala
// 该语句等价于 val cmp = if(num > 0) 1 else ()
scala> val cmp = if(num > 0) 1
cmp: AnyVal = ()
```

Scala是强类型的，所以它不会自动把 0/1 解释为 false/true，其他内置空对象如空列表 Nil 也不会被自动解释为 false，条件分支中必须是一个可以产生布尔值的表达式，这一点和 Java  保持了统一。

Scala的条件判定常用于初始化初值：

```scala
val filename = if(!args.isEmpty) args(0) else "default.txt"
```

这里的优势在于，如果首先给 filename 赋初值，然后再从参数中获取用户的设定，那么 filename就必须设置为 var 类型的。val 明确告诉用户 filename 是不可变的。

### 块语句

{}块可以包含多条表达式，块语句具有值，其结果是最后一个表达式的值。

```scala
// 首先定义两个点
scala> val (x0,y0)=(0,0)
x0: Int = 0
y0: Int = 0

scala> val (x1,y1)=(2,4)
x1: Int = 2
y1: Int = 4

// 导入数学函数
scala> import math._
import math._

// 分布求出两点距离
scala> val distance = {val dx = x1 - x0; val dy = y1 - y0; sqrt(dx * dx + dy * dy)}
distance: Double = 4.47213595499958
```

赋值语句的值是 Unit 类型，所以这样做不会得到预期的结果：

```scala
x = y = 1 // x 最终被赋值为Unit类型的（）
```



### 输入和输出

打印输出有三个函数：

1. print
2. println 带换行
3. printf 支持格式化字符串 

```scala
scala> print("Input a num:")
Input a num:
scala> print("Hello " + "John")
Hello John
scala> print("1 + 1 = " + 2)
1 + 1 = 2
scala> println("print with a linefeed")
print with a linefeed

scala> printf("I am %d years old!", 10)
I am 10 years old!
```

Scala 定义了一系列从控制台读取函数，它们定义在 scala.io.StdIn 包中，分为两大类：

1. readLine 可以带一个提示符参数
2. readInt、readDouble、readByte、readShort、readLong、readFloat、readChar 和 readBoolean 读取特定类型的值。

```scala
scala> val ID = scala.io.StdIn.readLine("Your ID: ")
Your ID: ID: String = 12345678

scala> val age = scala.io.StdIn.readInt()
age: Int = 10
```

### while 循环

Scala支持 while 循环，例如：

```scala
scala> var sum = 0
sum: Int = 0

scala> var till = 100
till: Int = 100

scala> while(till > 0){
     | sum += till
     | till -= 1
     | }

scala> sum
res23: Int = 5050
```

但是 Scala 没有提供 break 和 continue 语句来退出循环，可以使用Breaks 对象中的break 方法：

```scala
import scala.util.control.Breaks._
breakable {
    while(...){
        if(...)break;
        ...
    }
}
```

### do while 循环

先执行后检查。使用 do-while 读取标准输入的文本行，直到读到空行：

```scala
var line = ""
do {
    line = readLine()
    println("Read: " + line)
} while(line != "")
```

while 循环和 do while 循环均返回 Unit类型的 ()。

###for 循环

Scala提供了风格迥异的for 循环结构：

```scala
scala> var sum = 0
sum: Int = 0

scala> for(i <- 1 to 5)
     |     sum += i

scala> sum
res2: Int = 15
```

1 to 5 返回 [1,5]的Range对象，它具有迭代属性：

```
scala> 1 to 5
res3: scala.collection.immutable.Range.Inclusive = Range 1 to 5

```

可以指定 by 指定步长，例如：

```scala
scala> for(i <- 10 to 1 by -1)
     |     print(i + " ")
10 9 8 7 6 5 4 3 2 1
```

遍历字符串或数组时，通常需要使用 [0, n) 区间，可以使用 until 代替 to方法，until 返回一个不含上限的区间。

```scala
scala> for(i <- 0 until 10) // 不含结尾10
     |     print(i + " ")
0 1 2 3 4 5 6 7 8 9
```

遍历字符串无需使用下标，而是使用 <- 符号直接遍历：

```scala
scala> for(ch <- "Hello") sum += ch
```

由于Scala 支持函数式编程，这大大减少了对循环的使用。

### 高级for循环

可以对**变量 <- 表达式**形式复用来提供多个生成器，用分号将它们分割开。例如：

```scala
scala> for(i <- 1 to 9; j <- 1 to i){ print(j + "x" + i + "=" + j * i + "\t"); if (j == i)println("")}
1x1=1
1x2=2   2x2=4
1x3=3   2x3=6   3x3=9
1x4=4   2x4=8   3x4=12  4x4=16
1x5=5   2x5=10  3x5=15  4x5=20  5x5=25
1x6=6   2x6=12  3x6=18  4x6=24  5x6=30  6x6=36
1x7=7   2x7=14  3x7=21  4x7=28  5x7=35  6x7=42  7x7=49
1x8=8   2x8=16  3x8=24  4x8=32  5x8=40  6x8=48  7x8=56  8x8=64
1x9=9   2x9=18  3x9=27  4x9=36  5x9=45  6x9=54  7x9=63  8x9=72  9x9=81
```

每个生成器可以带一个守卫，以 if 开头的 Boolean 表达式：

```scala
scala> for(i <- 1 to 3; j <- 1 to 3)print((10 * i + j) + " ")
11 12 13 21 22 23 31 32 33

// 带守卫的生成器，去除重复组合11,22和33，注意 if 前没有分号
scala> for(i <- 1 to 3; j <- 1 to 3 if i != j)print((10 * i + j) + " ")
12 13 21 23 31 32
```

循环中的变量无需使用 var 定义。

如果 for 循环体以 yield 开始，则循环会构造出一个集合（Vector），每次迭代生成集合中的一个值：

```scala
scala> for(i <- 1 to 5) yield i % 2
res30: scala.collection.immutable.IndexedSeq[Int] = Vector(1, 0, 1, 0, 1)
```

这里循环叫做 for 推导式。它生成的集合类型与第一个生成器类型一致。

```scala
scala> for(c <- "Hello"; i <- 0 to 1) yield(c + i).toChar
res31: String = HIeflmlmop

scala> for(i <- 0 to 1; c <- "Hello") yield(c + i).toChar
res32: scala.collection.immutable.IndexedSeq[Char] = Vector(H, e, l, l, o, I, f, m, m, p)
```

### for推导式

for 推导式（comprehension ）是 Scala 中对普通 for loop 循环的升级，它实际上由三部分构成：

```scala
for {  // 使用大括号，可以使用换行分割各部分
    p <- persons             // 生成器
    n = p.name               // 定义
    if (n startsWith "To")   // 过滤器
} yield
```

for 推导式第一个部分总是生成器，可以有多个生成器，<- 符号左侧是一个模式匹配表达式，这方便解构某个对象，生成器对应高阶函数 map。

定义部分用于简写某个对象；过滤器为false时，不会进入推导式内部进行处理。定义和过滤器不是必须的。它们分别对应高阶函数 flatMap 和 withFilter。

```scala
case class Person(name: String, age: Int)
val people = List(
    Person("Tom", 10),
    Person("Jack", 20)
)
// 这里只有生成器，被编译器翻译为people.map(people => p)
for(p <-people)println(p)
// 输出
Person(Tom,10)
Person(Jack,20)
```

添加过滤器，在定义和过滤器中均可以引用生成器生成的临时变量：

```scala
for{
    p <-people
    if p.name == "Tom"
}println(p)

//输出
Person(Tom,10)
```

含有定义的 for 推导式：

```scala
// 等价于 for(p <-people;name = p.name;if name == "Tom")println(name)
for{
    p <-people
    name = p.name
    if name == "Tom"
}println(name)

//输出
Tom
```

出现在生成器，定义中的临时变量可以在内部自由使用，以完成元素转换。

for 推导式中如果没有 yield语句，则返回为()，否则返回 List 类型数据，可能是空列表 Nil。

```scala
// (1) works because `foreach` is defined   // for 依赖于 foreach 方法
for (p <- peeps) println(p)

// (2) `yield` works because `map` is defined // yield 表达式依赖于 map 方法
val res: Sequence[Int] = for {
    i <- ints
} yield i * 2
res.foreach(println)

// (3) `if` works because `withFilter` is defined // 过滤器依赖于 withFilter 方法
val res = for {
    i <- ints
    if i > 2
} yield i*2

// (4) works because `flatMap` is defined，多生成器依赖于 flatMap 方法
val mutualFriends = for {
    myFriend <- myFriends        // generator
    adamsFriend <- adamsFriends  // generator
    if (myFriend.name == adamsFriend.name)
} yield myFriend
```

### foreach

foreach 是序列对象的方法，它接受函数作为参数，提供典型的函数式编程句法：

```scala
scala> str.foreach(i => print(i + " "))
H e l l o
```

foreach 作为对象的方法，而参数部分是一个函数字面量（function literal），可以看做是一个匿名函数，它接受一个名为 i 的参数，而函数体为 print 语句。Scala 根据对象类型String 自动推断出参数类型为 Char。可以指明元素类型：

```scala
scala> str.foreach((i: Char) => print(i + " "))
H e l l o
```

Scala 支持函数字面量的简写，当只有单个参数时，且无需对参数进行特殊处理，可以不写出形参和实参名。例如：

```scala
scala> str.foreach(println)
H
e
l
l
o
```

遍历映射：

```scala
scala> val m = Map[String, Int]("a" -> 1, "b" -> 2, "c" -> 3)
m: scala.collection.immutable.Map[String,Int] = Map(a -> 1, b -> 2, c -> 3)

// 参数为元组
scala> m.foreach(p => println(">>> key=" + p._1 + ", value=" + p._2))
>>> key=a, value=1
>>> key=b, value=2
>>> key=c, value=3

// case 定义偏函数
scala> m.foreach {case (key, value) => println(">>> key=" + key + ", value=" + value)}
>>> key=a, value=1
>>> key=b, value=2
>>> key=c, value=3
```

### try 表达式

try 表达式用于异常处理。方法如果不能正常地返回值，也可以通过抛出异常终止执行。在方法中要么捕获异常，要么自我终止，让异常传播到更上层调用者。

异常层层向上传播，主键展开调用者，直到某个方法处理该异常或者再没有上层调用者为止。与 Java 类似，抛出异常的方法如下：

```scala
throw new IllegalArgumentException
```

1. 首先创建异常对象
2. 然后使用 throw 关键字抛出

抛出异常的表达式的值类型是 Nothing。

```scala
import java.io.IOException
try {
    val f = new FileReader("input.txt")
}catch{
    case ex: FileNotFoundException => println("Not find file")
    case ex: IOException => println("Other Errors")
}
```

如果异常既不是 FileNotFoundException 也不是 IOException，try-catch 将会终止执行，异常将向上传播。

如果是否抛出异常都要执行的代码放在 finally 子句中。例如打开后关闭文件的处理：

```scala
var java.io.FileReader
val file = new FileReader("input.txt")
try {
    // 文件操作
} finally {
    file.close() // 确保关闭文件
}
```

finally 子句中执行清理工作。

### match 表达式

Scala 的 match 表达式类似其他语言中的 switch 语句。但是它支持任意的模式匹配。

```scala
"pepper" match {
    case "salt" => println("salt")
    case _ => println("unknown") // 匹配所有其他选项
}
```

match 可以匹配任意常量，字符串。每条 case 语句中隐含了 break 语句。

case 中的每个分支的最后一个表达式是返回值，match 语句会返回匹配分支语句的值。

```scala
var hah = "pepper" match {
	case "salt" => "salt"
	case _ => "unknown"
}
```

如果没有匹配任何分支，例如没有提供默认分支，则会报异常：

```scala
Exception in thread "main" scala.MatchError
```

### 作用域

Scala 作用域规则和 Java类似，{} 将定义新的作用域。Java 不可以在内嵌作用域定义一个跟外部作用域内相同名称的变量。Scala 可以，内嵌作用域中的变脸会遮挡（shadow）外部作用域中相同名称的变量。

```scala
val a = 10
{
	val Int a = 100
    println(a) // 打印 100
}
```

##函数

与 Java 只可以定义类和类方法不同，Scala 支持定义函数。例如：

```scala
scala> def add(a:Int, b:Int):Int = a + b
add: (a: Int, b: Int)Int

scala> add(1,2)
res34: Int = 3
```

参数类型是必须提供的，但是返回类型可以不提供，编译器可以自动推断。递归函数则必须提供返回值类型。

如果函数体需要多个表达式，可以使用代码块，块中最后一个表达式的值就是函数的返回值。

```scala
// 定义阶乘函数
scala> def fac(n:Int) = {
     |     var r = 1
     |     for(i <- 1 to n) r = r * i
     |     r
     | }
fac: (n: Int)Int

scala> fac(3)
res35: Int = 6
```

可以使用 return 跳出函数体，但通常无需这样做。

### 默认参数和命名参数

Scala 的默认参数和命名参数机制与 Python 类似，可以给参数提供默认值：

```scala
def decorate(str: String, left: String = "[", right: String ="]"):String = {
	left + str + right
}

// 使用默认参数
scala> decorate("Hello")
res39: String = [Hello]

// 覆盖默认参数
scala> decorate("Hello", "<", ">")
res38: String = <Hello>

// 使用命名参数
scala> decorate(str="Hello", right = "<", left = ">")
res40: String = >Hello<

// 从左向右实参依次对应形参，未提供实参使用默认值
scala> decorate("Hello", "<")
res41: String = <Hello]
```

可以混用未命名参数和命名参数：

```scala
scala> decorate("Hello", right = "<")
res42: String = [Hello<
```

### 变长参数

可以用任意多的整型参数调用如下函数：

```scala
def sum(args: Int*):Int = {
	var result = 0
    // 函数得到一个类型为 Seq 的参数；如果 args 中没有参数，则不仅进入循环
	for(arg <- args)
		result += arg
	result
}

scala> sum(1,2,3)
res0: Int = 6

// 返回默认值 0 
scala> sum()
res1: Int = 0
```

sum 函数不能直接接受序列，例如 sum(1 to 5) 是错误的。需要告诉编译器把它当做参数序列处理，也即追加 :_*，例如：

```scala
scala> sum(1 to 5)
<console>:13: error: type mismatch;
 found   : scala.collection.immutable.Range.Inclusive
 required: Int
       sum(1 to 5)
             ^
scala> sum(1 to 5:_*)
res3: Int = 15
```

### 局部函数

在类中可以定义私有方法还封装不对外暴露的处理。这些函数可以通常是助手函数，完成单一的内聚处理。这一方式问题是助手函数的名称为污染整个类的命名空间。

如果希望类的使用者不要意识到这些函数的存在，那么可以在函数内部定义函数，就像局部变量一样，局部函数只在它的代码块中可见。

```scala
object learnScala {
	// 打印文件中所有行宽度超过 width 的行
	def processFile(fname:String, width:Int): Unit ={
        // 这里定义的变量，内部函数也可访问
        // 内部函数，可以访问外部函数的参数
		def processLine(line:String)={
			if(line.length > width)
				println(fname + ": " + line.trim)
		}
		
		import scala.io.Source
		val source = Source.fromFile(fname)
		for (line <- source.getLines())
			processLine(line) // 调用内部函数处理
	}
	
	def main(args: Array[String]): Unit = {
		val half = new Rational(1,2)
		val half1 = new Rational(1,2)
		
		processFile("E:\\learn.scala", 20)
	}
}
```

### 一等函数

```scala
scala> val add1 = (x:Int) => x+1
add1: Int => Int = $$Lambda$1025/5338884@28a6e171

scala> add1(10)
res0: Int = 11
```

=>右侧部分被称为函数字面量。=> 符号表示该函数将左侧的内容转换成右侧的内容。函数值是对象，可以存放在常量或者变量中。

如何理解函数值是对象，以下两种定义是等价的，根据参数的不同，将创建 FunctionN 对象。

```scala
val succ = (x: Int) => x + 1
val anonfun1 = new Function1[Int, Int] {
    def apply(x: Int): Int = x + 1
}

assert(succ(0) == anonfun1(0))
```

这里的 succ 被称为匿名函数（Lamda 函数），函数同样具有函数名，函数类型和值：

```scala
scala> val succ = (x: Int) => x + 1
  succ: Int => Int = $$Lambda$1944/804626931@12bcdde0
  ----- ----------   --------------------------------
//函数名  函数类型    函数值（对应存储的函数体中的代码地址）

scala> val sum = (a: Int, b: Int) => a + b
sum: (Int, Int) => Int = $$Lambda$1945/892350438@1bf914d6

// 函数和普通变量一样，具有变量名，类型和值（字面量），所以函数名可以作为一个普通的变量名使用，例如作为其他高阶函数的参数或者返回值
scala> val a = 10
a: Int = 10
```

对比以上两个函数的类型，可以看出函数类型由函数的入参和出参的类型构成。

函数中可以使用外部变量，如果外部变量有变化，函数值也会跟随变化，例如：

```scala
// 乘法因子
scala> var factor = 2
factor: Int = 2

scala> val multiplier = (i: Int) => i * factor
multiplier: Int => Int = $$Lambda$1052/1606886748@20a47036

scala> multiplier(1)
res1: Int = 2

// 改变乘法因子
scala> factor = 10
factor: Int = 10

scala> multiplier(1)
res2: Int = 10
```

所有的集合类都提供了 foreach 方法，它接收一个函数作为入参，并对它的每个元素调用这个函数。入参可以使用函数字面量定义。

```scala
val list = List(1,2,3)
list.foreach((x:Int) => println(x))
```

####简写函数字面量

由于可以根据调用调用对象推测出元素类型，可以省去参数类型声明：

```scala
scala> list.foreach((x) => println(x))
1
2
3
```

另外也可以去除参数两侧的圆括号。

#### 占位符

可以使用下划线作为占位符，表示一个或者多个参数，只要满足每个参数只在函数字面量中出现一次即可，此时省去了参数声明部分和 => 符号。

```scala
scala> list.foreach(println(_))
1
2
3
```

但是使用占位符时，如果编译器没有足够多信息推断确实的参数类型，则要定义每个占位符的类型：

```scala
scala> val f =(_:Int) + (_:Int)
f: (Int, Int) => Int = $$Lambda$1214/1980711696@3b9d85c2

scala> f(1,2) // 接受两个参数的函数
res9: Int = 3
```

多个占位符表示多个参数，而不是对单个参数的重复使用。

#### 偏应用函数

偏应用函数(Partial Applied Function)也叫部分应用函数，跟偏函数(Partial Function)从英文名来看只有一字之差，但他们二者之间却有天壤之别。 

部分应用函数, 是指一个函数有n个参数, 而我们为其提供少于n个参数, 那就得到了一个部分应用函数。 

```scala
scala> def sum(a:Int, b:Int):Int = a + b
sum: (a: Int, b: Int)Int

// 给定部分参数的默认值，占位符取代需要传入的参数
scala> val sum1 = sum(_,1) 
sum1: Int => Int = $$Lambda$1341/1759280963@137efccc

scala> sum1(10)
res29: Int = 11
```

#### 闭包

运行时从函数字面量创建出来的函数对象被称作闭包（closure）。没有自由变量的函数字面量，例如 (X:Int) =>X+1 称为闭合语（closed term），无需外部变量，已经自闭合。

```scala
// 乘法因子
scala> var factor = 2
factor: Int = 2

scala> val multiplier = (i: Int) => i * factor
multiplier: Int => Int = $$Lambda$1052/1606886748@20a47036
```

上例不同，它有外部变量 factor，所以需要创建上下文，编译器创建了一个闭包，用于包含（或“覆盖”） multiplier 与它引用的外部变量的上下文信息，从而也就绑定了外部变量本身。函数值是通过闭合这个开放语（open term）的动作产生的。

外部变量的修改可以被闭包感知，也即引用的是变量而不是它的值。

此外闭包中可以修改外部自变量，并被闭包外感知。

```scala
scala> var sum = 0
sum: Int = 0

// 每次都修改闭包外变量 sum，所以最后 sum 记录了所有元素的和
scala> list.foreach(sum += _)

scala> sum
res32: Int = 6
```

Java 的内部类不允许访问外部作用的可修改变量。 

### 高阶函数

高阶函数指参数为函数或者返回值为函数的函数。

高阶函数的函数参数定义方式与普通变量一样，函数名加函数类型：

```scala
// callback 为参数名，() => Unit 为函数类型：没有入参，返回类型为 Unit
def callFunc(callback: () => Unit): Unit ={
    callback()
}

def sayHello(): Unit ={
	println("Hello world!")
}

// sayHello _ 进行了方法向函数的显式转换，Eta-expansion 扩展不推荐空参方法向函数转换
scala> callFunc(sayHello _)
Hello world!

scala> val sayBye = ()=> println("Good bye!")
sayBye: () => Unit = $$Lambda$1963/2043763469@4fd6744c

// 此处由于 sayBye 默认为函数类型，所以不用显式转换
scala> callFunc(sayBye)
Good bye!
```

从示例可以看出，传入的函数参数必须和声明的函数参数类型相匹配。显然callFunc中参数名callback可以是任意合法的标识符，所以通常使用 f 来简化函数参数的定义。

```scala
callFunc(f: () => Unit) // 无参数时不能省略定义参数列表的小括号
strLen(f:(String) => Int)
sum(f:(Int, Int) => Int)

// 对于只有一个参数的函数参数，可以省略定义参数列表的小括号，例如：
strLen(f: String => Int)
list2Elemet(f: List[Person] => Person)
```

其中 callFunc(f: () => Unit): Unit 被称为函数签名（functionsignatures），它由函数变量名，类型签名和返回值类型组成，对于callFunc中的 f 函数来说， () =>Int 部分被称为类型签名（type signatures）。

函数的等价定义：

````scala
// 显式定义函数类型签名：(Int, Int) => Int
scala> val sum: (Int, Int) => Int = (a, b) => a + b 
sum: (Int, Int) => Int = $$Lambda$1969/592030716@218642f6

// 隐式定义函数类型签名：(Int, Int) => Int
scala> val sum = (a:Int, b:Int) => a + b
sum: (Int, Int) => Int = $$Lambda$1970/1378644910@bad1c5
````

![img](./signature.jpg)

####map

map 方法可以将集合中的成员一次转换为另一种类型的成员，并返回新类型的集合：

```scala
scala> Array(1,2,3).map(_ * 2)
res79: Array[Int] = Array(2, 4, 6)

scala> List(1,2,3).map("<" + _ + ">")
res82: List[String] = List(<1>, <2>, <3>)
```

传入的函数参数将调用者中的每一个成员转换成新集合中的成员，新旧集合元素类型可能改变。

#### flatten

flatten 将集合中的每个集合元素展开一层（it converts a “list of lists” to a single list.），返回的集合类型不变，但是元素类型改变了：

```scala
val xs = List(
           Set(1, 2, 3),
           Set(1, 2, 3)
         ).flatten
// xs == List(1, 2, 3, 1, 2, 3)

val ys = Set(
           List(1, 2, 3),
           List(3, 2, 1)
         ).flatten
// ys == Set(1, 2, 3)

// flatten 只展开一层
scala> val xs=List(Set(List(1,2,3), 1,2)).flatten
xs: List[Any] = List(List(1, 2, 3), 1, 2)
```

另外 flatten 会自动过滤掉 None 元素和空列表：

```scala
scala> val strings = Seq("1", "2", "foo", "3", "bar")
strings: Seq[java.lang.String] = List(1, 2, foo, 3, bar)

scala> val mapped = strings.map(toInt)
mapped: Seq[Option[Int]] = List(Some(1), Some(2), None, Some(3), None)

scala> mapped.flatten
res87: Seq[Int] = List(1, 2, 3)
```



#### flatMap

flatMap 实际上是先进行 map 然后再进行 flatten，例如：

```scala
scala> List("abc", "123").map(_.toCharArray).flatten
res83: List[Char] = List(a, b, c, 1, 2, 3)

scala> List("abc", "123").flatMap(_.toCharArray)
res86: List[Char] = List(a, b, c, 1, 2, 3)
```

传入 flatMap的函数作用在 map 上。

```scala
// 字符串转数字
def toInt(s: String): Option[Int] = {
    try {
        Some(Integer.parseInt(s.trim))
    } catch {
        case e: Exception => None
    }
}

scala> val strings = Seq("1", "2", "foo", "3", "bar")
strings: Seq[java.lang.String] = List(1, 2, foo, 3, bar)

scala> val mapped = strings.map(toInt)
mapped: Seq[Option[Int]] = List(Some(1), Some(2), None, Some(3), None)

scala> mapped.flatten
res87: Seq[Int] = List(1, 2, 3)

// 以上操作等价于
scala> strings.flatMap(toInt)
res1: Seq[Int] = List(1, 2, 3)

// 过滤空列表
scala> Array(List(1,2,3), Nil, List(4,5,6)).flatten
res96: Array[Int] = Array(1, 2, 3, 4, 5, 6)
```

查看高阶函数签名，可以明确它需要传入的函数类型：

```scala
def flatMap[B](f: A => Sequence[B]): Sequence[B]
```



### 传名和传值参数

传名（call-by-name）和传值（call-by-value）参数是变量作为参数传递给方法时是否预先求值：

传名参数：在参数类型声明前标记了 => 的参数，比如（x: => Int）。相应的入参并不会在方法调用前求值，而是在方法体内，该参数的名字每次被使用的时候求值。传名参数只能用于方法中，不能用于匿名函数中。

传值参数是在参数类型声明前没有被 => 标记的参数，比如（x:Int）。相应的入参在方法被实际调用前预先求值。

```scala
scala> def funByValue(x:Int):Int = {println("in function"); 2 * x }
funByValue: (x: Int)Int

// 传名参数，无需给出参数类表
scala> def funByName(x: => Int):Int = {println("in function"); 2 * x }
funByName: (x: => Int)Int

// 传值参数，会首先对参数进行求值，然后再进入函数体
scala> funByValue({println("print from block"); 1})
print from block
in function
res46: Int = 2

// 传名参数，首先进入函数体，然后再调用参数的时候再对参数求值
scala> funByName({println("print from block"); 1})
in function
print from block
res47: Int = 2

// 进入函数体后再求值 x 的值
scala> def funByValue(x: => Int) = x*10
func: (x: => Int)Int

// 与 C 语言中的宏替换不同，进入函数体后求 x 的值为 2 然后 2 * 10 = 20
scala> funByValue(1+1)
res55: Int = 20
```

所以如果你需要执行同一段代码，那么选择传名参数就有必要，否则传值参数只能得到计算后的值。如果把代码块封装为函数，那么效果看起来和传递函数变量一样。有时候直接使用代码块更自然：

```scala
scala> def myAssert(boolBlock: => Boolean) = if(!boolBlock)throw new AssertionError
myAssert: (boolBlock: => Boolean)Unit

scala> myAssert(5<3) // 定义成代码块可以直接使用代码块
java.lang.AssertionError
  at .myAssert(<console>:11)
  ... 28 elided

// 定义成传递函数
scala> def myAssert(boolBlock: () => Boolean) = if(!boolBlock())throw new AssertionError
myAssert: (boolBlock: () => Boolean)Unit

scala> myAssert(() => 5<3) // 定义成函数则要把代码块封装为匿名函数
java.lang.AssertionError
  at .myAssert(<console>:11)
  ... 28 elided
```

传名参数的应用不单如此，更重要的应用在于函数的柯里化（currying），可以创建新的语言控制结构。

类似传名机制的延迟求值机制还有 lazy 和 def。

#### 延迟求值

def：类似于每一次重新赋值，如果用 def 定义函数，则是每一次重新获得一个函数，做call-by-name操作。

```scala
// def 定义变量，指向一个代码块
scala> def block = {println("print from block"); 1}
block: Int

// 每次调用均重复求值
scala> block
print from block
res48: Int = 1

scala> block
print from block
res49: Int = 1
```

val：获得一次，并立即执行，且在生命周期内不能再被修改，使用的是call-by-value操作。

```scala
// val 定义的不可变变量，只在第一次定义时求值
scala> val block = {println("print from block"); 1}
print from block
block: Int = 1

// 直接返回变量值
scala> block
res50: Int = 1
```

var：在生命周期内可以被再次赋值。
lazy val：惰性执行，也就是赋值(绑定)的时候先不会执行，等到需要的时候再执行。

```scala
// lazy 修饰的 val 变量只在第一次使用时求值
scala> lazy val block = {println("print from block"); 1}
block: Int = <lazy>

// 第一次使用时求值，有打印
scala> block
print from block
res51: Int = 1

// 再次使用时，不用再次求值，无打印
scala> block
res52: Int = 1
```

#### 延迟方法

在Scala中类的方法可以分为即时计算的(*strict*)或者延后计算的(non-*strict* or *lazy*)。例如 withFilter 就是延后计算的，而 filter 就是即时计算的。

```scala
object Test1WithFilterLazy extends App {
    def lessThan30(i: Int): Boolean = {
        println(s"\n$i less than 30?")
        i < 30
    } 

    def moreThan20(i: Int): Boolean = {
        println(s"$i more than 20?")
        i > 20
    } 

    val a = List(1, 25, 40)
    val q0 = a.withFilter(lessThan30)
    println("filtered by lessThan30")
    val q1 = q0.withFilter(moreThan20)
    println("filtered by moreThan20")
    for (r <- q1) println(s"$r") // 此处才真正执行withFilter中的过滤方法
}
```

打印结果可能出人意料：

```scala
filtered by lessThan30
filtered by moreThan20

1 less than 30?
1 more than 20?

25 less than 30?
25 more than 20?
25

40 less than 30?
```

如果 withFilter 方法换成 filter，则会顺序打印。

#### 柯里化

柯里化函数支持多个参数列表，一个最简单的例子：

```scala
scala> def sum(a: Int, b: Int, c: Int) = a + b + c
sum: (a: Int, b: Int, c: Int)Int

scala> sum(1,2,3)
res62: Int = 6

// 柯里化的 sum 函数
scala> def cSum(a: Int)(b: Int)(c: Int) = a + b + c
sum: (a: Int)(b: Int)(c: Int)Int

scala> cSum(1)(2)(3) // 必须为每个参数列表独立提供参数
res61: Int = 6
```

当调用 cSum 时，实际上是一个链式调用，依次从左向右使用单个参数列表调用函数，并返回一个函数值，直至最后一个函数返回（也即使用多个单参数列表函数迭代求值）。

```scala
result = f(x)(y)(z)

// 等价于
f1 = f(x)
f2 = f1(y)
result = f2(z)
```

上面的例子看起来没有什么特殊，但是当传入不同代码块作为参数时，神奇的效果就出现了。

```scala
var i = 0
while(i < 2)({ // 这里使用小括号依然可以正确编译，为什么？
	println("hello")
	i += 1
})
```

使用柯里化模拟 while 控制结构：

```scala
def whilst(testCondition: => Boolean)(codeBlock: => Unit) {
    while(testCondition)
    	codeBlock
}

var i = 0
whilst(i < 2){ 由于只有一个参数，可以去掉小括号
	println("hello")
	i += 1
}

// 体会下这两种定义和使用的不同
def whilst(testCondition: => Boolean, codeBlock: => Unit):Unit = {
	while(testCondition)
		codeBlock
}

var i = 0
whilst(i < 2, ({
	println("hello")
	i += 1
}))
```

隐式参数传递(只可以有一个隐式参数列表，且只能是最后一个参数列表，如果定义了多个隐式参数，那么编译器将报错)：

```scala
scala> def printIntIfTrue(a: Int)(implicit boo: Boolean) = if (boo) println(a)
printIntIfTrue: (a: Int)(implicit boo: Boolean)Unit

scala> implicit val b = true // 定义隐式参数，且和形参名称可以不一致
b: Boolean = true

scala> printIntIfTrue(1)
1
```

不推荐这种令逻辑混乱的调用，推荐使用偏应用函数（partially-applied）方式：

```scala
scala> val alwaysPrint = (x:Int) => printIntIfTrue(x)(true)
alwaysPrint: Int => Unit = $$Lambda$1443/1103795260@1abfd287

scala> alwaysPrint(1)
1
```

#### 柯里化普通函数

可以调用普通函数的.curried 方法柯里化函数。

```scala
scala> def add(x: Int, y: Int) = x + y
add: (x: Int, y: Int)Int

scala> val addFunction = add _ // 方法转换为函数
addFunction: (Int, Int) => Int = $$Lambda$1482/1820567481@3c215934

scala> (add _).isInstanceOf[Function2[_, _, _]]
res8: Boolean = true

scala> val addCurried = addFunction.curried // 柯里化
addCurried: Int => (Int => Int) = scala.Function2$$Lambda$1504/621738848@4ec357c3

scala> addCurried(1)(2)
res9: Int = 3
```



### 过程

没有返回值的函数被称为过程（procedure），实际上过程返回Unit类型的()，只是定义的时候可以不显式地指明返回值，并且省去 = 号：

```scala
def box(s:String){
    var border = "-" * s.length + "--\n"
    println(border + "|" + s + "|\n" + border)
}

scala> box("hello")
-------
|hello|
-------
```

##偏函数

在Scala中，被“{}”包含的一系列case语句可以被看成是一个函数字面量，它可以被用在任何普通的函数字面量适用的地方。  多个 case 语句可以组合成一个偏函数。

偏函数(Partial Function)，是一个数学概念它不是"函数"的一种, 它跟函数是平行的概念。  Scala中的Partia Function是一个Trait，其的类型为PartialFunction[A,B]，其中接收一个类型为A的参数，返回一个类型为B的结果。 

Scala 中的偏函数不是函数，这与 Python 中的偏函数（等价于 Scala中的偏应用函数）不同。

```scala
// 定义 Int=>String 映射的偏函数
scala> val pf:PartialFunction[Int,String] = {
     |   case 1=>"One"
     |   case 2=>"Two"
     |   case 3=>"Three"
     |   case _=>"Other"
     | }
pf: PartialFunction[Int,String] = <function1>

scala> pf(1)
res0: String = One

scala> pf(2)
res1: String = Two
```

偏函数内部有一些方法，比如isDefinedAt、OrElse、 andThen、applyOrElse等等。



### isDefinedAt

判断传入来的参数是否在这个偏函数所处理的范围内。  刚才定义的pf来尝试使用isDefinedAt()，只要是数字都是正确的，因为有case _=>"Other"这一句。如果换成其他类型就会报错。 

```scala
scala> pf.isDefinedAt(1)
res4: Boolean = true

scala> pf.isDefinedAt(2)
res5: Boolean = true

scala> pf.isDefinedAt("1")
<console>:13: error: type mismatch;
 found   : String("1")
 required: Int
       pf.isDefinedAt("1")
                      ^

scala> pf.isDefinedAt(100)
res7: Boolean = true
```

再定义一个PartialFunction ：

```scala
scala> val anotherPF:PartialFunction[Int,String] = {
     |    case 1=>"One"
     |    case 2=>"Two"
     |    case 3=>"Three"
     | }
anotherPF: PartialFunction[Int,String] = <function1>

scala> anotherPF.isDefinedAt(1)
res8: Boolean = true

scala> anotherPF.isDefinedAt(4)
res11: Boolean = false
```

### orElse

orElse : 将多个偏函数组合起来使用，效果类似case语句。 

```scala
scala> val onePF:PartialFunction[Int,String] = {
     |   case 1=>"One"
     | }
onePF: PartialFunction[Int,String] = <function1>

scala> val twoPF:PartialFunction[Int,String] = {
     |   case 2=>"Two"
     | }
twoPF: PartialFunction[Int,String] = <function1>

scala> val newPF = onePF orElse twoPF orElse threePF orElse otherPF
newPF: PartialFunction[Int,String] = <function1>

scala> newPF(1)
res0: String = One
```

### andThen

andThen: 相当于方法的连续调用，比如g(f(x))。 

```scala
scala> val pf1:PartialFunction[Int,String] = {
     |   case i if i == 1 => "One"
     | }
pf1: PartialFunction[Int,String] = <function1>

scala> val pf2:PartialFunction[String,String] = {
     |   case str if str eq "One" => "The num is 1"
     | }
pf2: PartialFunction[String,String] = <function1>

scala> val num = pf1 andThen pf2
num: PartialFunction[Int,String] = <function1>

scala> num(1)
res4: String = The num is 1
```

pf1的结果返回类型必须和pf2的参数传入类型必须一致，否则会报错。 

### applyOrElse

applyOrElse：它接收2个参数，第一个是调用的参数，第二个是个回调函数。如果第一个调用的参数匹配，返回匹配的值，否则调用回调函数。 

```scala
scala> onePF.applyOrElse(1,{num:Int=>"two"})
res5: String = One

scala> onePF.applyOrElse(2,{num:Int=>"two"})
res6: String = two
```

在这个例子中，第一次onePF匹配了1成功则返回的是"One"字符串。第二次onePF匹配2失败则触发回调函数，返回的是"Two"字符串。

## 数组

### 定长数组

长度不变的数组使用  Array  定义。Scala 对数组元素的访问使用()，而不是通常的[]，这是数组 apply 方法的缩写形式。

```scala
scala> val nums = new Array[Int](10) // 所有元素初始化为 0
nums: Array[Int] = Array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

scala> nums(0)
res24: Int = 0

// nums(0) 是 apply 方法的缩写形式
scala> nums.apply(0)
res25: Int = 0

// String 类型的定长数组初始化为 null
scala> val strs = new Array[String](2)
strs: Array[String] = Array(null, null)

// 给定初始值，无需 new，且类型可以自动推断
scala> val months = Array("Jan", "Feb")
months: Array[String] = Array(Jan, Feb)

// val 类型数组可以改变数组内容，但不能改变引用
scala> strs(0) = "Hello"

scala> strs
res7: Array[String] = Array(Hello, null)

// 超出数组范围，则抛出异常
scala> strs(2)
java.lang.ArrayIndexOutOfBoundsException: 2
  ... 28 elided

// 不可以改变 val 类型的引用
scala> strs = ("1", "2")
<console>:12: error: reassignment to val
       strs = ("1", "2")
```

### 变长数组

如果需要对长度按需变化的数组，对应 Java 中的 ArrayList，Scala提供等价的 ArrayBuffer。

变长数组也称为数组缓冲。

```scala
// ArrayBuffer 位于 scala.collection.mutable 包中
scala> import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ArrayBuffer

// 定义空数组缓冲
scala> val b = ArrayBuffer[Int]()
b: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer()

// 尾部追加
scala> b += 1
res11: b.type = ArrayBuffer(1)

// 链式处理
scala> b += (2,3) -= (2, 3)
res51: b.type = ArrayBuffer(1)

// 头部删除
scala> b -= 2
res13: b.type = ArrayBuffer(1)

// ++= 追加固定数组
scala> b ++= Array(8, 13, 21)
res17: b.type = ArrayBuffer(1, 3, 8, 13, 21)

// 尾部删除
scala> b.trimEnd(2)

scala> b
res19: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(1, 3, 8)

// 索引位置 2 处插入 6
scala> b.insert(2,6)

scala> b
res21: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(1, 3, 6, 8)

// 索引位置 2 删除，并返回
scala> b.remove(2)
res22: Int = 6

scala> b
res23: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(1, 3, 8)

// 转固定数组
scala> b.toArray
res24: Array[Int] = Array(1, 3, 8)

// 转变长数组
scala> var a = b.toArray
a: Array[Int] = Array(1, 3, 8)

scala> a.toBuffer
res25: scala.collection.mutable.Buffer[Int] = ArrayBuffer(1, 3, 8
```

定义为 Any 类型的数组类似 Python 中的 List，可以存储任何其他类型的对象。

```
scala> var b = ArrayBuffer[Any]()
b: scala.collection.mutable.ArrayBuffer[Any] = ArrayBuffer()

```

### 遍历数组和变长数组

Scala 对遍历数组和变长数组提供了统一的处理方式：

```scala
// 遍历定长数组
scala> var a = Array(1,2,3)
a: Array[Int] = Array(1, 2, 3)

scala> for (i <- a)print(i + " ")
1 2 3

// 遍历变长数组
scala> val b = ArrayBuffer[String]()
b: scala.collection.mutable.ArrayBuffer[String] = ArrayBuffer()

scala> b += "Hello"
res18: b.type = ArrayBuffer(Hello)

scala> b += "world"
res19: b.type = ArrayBuffer(Hello, world)

scala> for (i <- b)print(i + " ")
Hello world
```

当然也可以使用索引来遍历：

```scala
scala> for (i <- 0 until b.length)print(b(i) + " ")
Hello world
```

也可以指定索引的步长，

```scala
scala> a
res33: Array[Int] = Array(0, 1, 2, 3, 4, 5)

scala> for(i <- 0 to a.length-1 by 2)print(a(i) + " ")
0 2 4
scala> for(i <- 0 until (a.length-1, 2))print(a(i) + " ")
0 2 4
```

或者进行逆序索引：

```scala
scala> for(i <- a.length-1 to 0 by -2)print(a(i) + " ")
5 3 1

scala> for(i <- (0 until a.length).reverse)print(a(i) + " ")
5 4 3 2 1 0
```

理解 Range 类型有助于理解索引的机制：

```scala
// Range 的 revere方法反转区间
scala> var a = (0 until 5).reverse
a: scala.collection.immutable.Range = Range 4 to 0 by -1

// by 方法指定区间步长
scala> var a = (0 until 5).by(2)
a: scala.collection.immutable.Range = inexact Range 0 until 5 by 2

// 链式调用
scala> var a = (0 until 5).by(2).reverse
a: scala.collection.immutable.Range = Range 4 to 0 by -2

scala> for (i <- a)print(i + " ")
4 2 0
```

### 数组处理

使用 for 和 yield 组成的推导式生成器，可以转换数组或者变长数组，原数组保持不变，产生的新数组与原数组类型相同。

```scala
scala> a
res47: Array[Int] = Array(1, 2, 3, 4, 5)

// a 为数组，b也为数组
scala> var b = for(i <- a)yield i * 2
b: Array[Int] = Array(2, 4, 6, 8, 10)

scala> a
res49: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(2, 4, 6, 8, 10)
// a 为变长数组，b也为变长数组
scala> var b = for(i <- a)yield i * 2
b: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(4, 8, 12, 16, 20)
```

可以使用卫士来处理特定条件的元素：

```scala
scala> a
res51: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(2, 4, 6, 8, 10)

// 提取所有 4 的倍数的元素
scala> var b = for(i <- a if i % 4 == 0)yield i
b: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(4, 8)
```

函数式编程可以更优雅地实现类似处理：

```scala
scala> var a = Array[Int](1,2,3,4,5)
a: Array[Int] = Array(1, 2, 3, 4, 5)

// 过滤所有偶数
scala> a.filter(_ % 2 == 0)
res60: Array[Int] = Array(2, 4)

// 过滤结果 * 2 映射处理
scala> a.filter(_ % 2 == 0).map(2 * _)
res61: Array[Int] = Array(4, 8)

// 过滤+映射+聚合
scala> a.filter(_ % 2 == 0).map(2 * _).reduce(_+_)
res62: Int = 12
```

### 常用算法

大部分的业务需求集中在过滤，排序，求和等处理上。Scala 对象内建了大量方法处理这些任务。这些作用在数组上的方法对变长数组同样适用：

```scala
scala> a
res21: Array[Int] = Array(1, 2, 3, 4, 5)

// sum 方法只针对数值类型数组
scala> a.sum
res22: Int = 15
```

max,min方法返回数组中的最大值和最小值，除了数值型数组外，还支持布尔数组和字符串数组，它们具有 Ordered 特质。

```scala
scala> b
res27: Array[Boolean] = Array(true, false)

scala> b.max
res28: Boolean = true

scala> b.min
res29: Boolean = false

scala> var c = Array[String]("hello", "world")
c: Array[String] = Array(hello, world)

scala> c.max
res31: String = world
```

sorted 方法可以将原数组排序，并返回新数组，类型保持一致。

```scala
scala> var a = Array[Int](1,9,2,4)
a: Array[Int] = Array(1, 9, 2, 4)

// 默认从小到达排序
scala> a.sorted
res39: Array[Int] = Array(1, 2, 4, 9)

// 指定排列顺序
scala> a.sortWith(_<_)
res40: Array[Int] = Array(1, 2, 4, 9)

scala> a.sortWith(_>_)
res41: Array[Int] = Array(9, 4, 2, 1)
```

scala.util.Sorting.quickSort   方法则之际对原数组排序。

```scala
scala> scala.util.Sorting.quickSort(a)

scala> a
res43: Array[Int] = Array(1, 2, 4, 9)
```

mkString 方法可以将数组转换为字符串，可以指定连接符：

```scala
scala> a.mkString(", ")
res45: String = 1, 2, 4, 9

// 不要使用 toString 方法
scala> a.toString
res44: String = [I@2a48b0eb
                
scala> c
res49: Array[String] = Array(hello, world)

scala> c.toString
res50: String = [Ljava.lang.String;@4066edd3
```

### 多维数组

多维数组通过数组的数组实现，需要使用数组的 ofDim 方法：

```scala
// 创建 2 行 3 列数组
scala> val matrix = Array.ofDim[Int](2, 3)
matrix: Array[Array[Int]] = Array(Array(0, 0, 0), Array(0, 0, 0))

// 通过两个圆括号访问
scala> matrix(0)(0) = 1

scala> matrix
res53: Array[Array[Int]] = Array(Array(1, 0, 0), Array(0, 0, 0))
```

可以通过定义数组的数组，也即每个元素都是数组类型，这样就创建了行长度不一样的数组，也即不规则数组。

```scala
// 创建一维数组，每个元素都是数组类型
scala> val triangle = new Array[Array[Int]](3)
triangle: Array[Array[Int]] = Array(null, null, null)

// 为每个数组元素继续分配数组空间
scala> for(i <- 0 until triangle.length) {
     |   triangel(i) = new Array[Int](i + 1)
     | }

scala> triangle
res63: Array[Array[Int]] = Array(Array(0), Array(0, 0), Array(0, 0, 0))
```

## List链表

List类与数组类似，但元素不可变，被设计为线程安全的，允许函数式编程。Python 中列表元素是可变的。

```scala
// 创建空链表
scala> List() == Nil
res10: Boolean = true
```

链表的最后一个元素总是 Nil，它表示一个空链表。出空链表外，每个链表元素由值和指针构成。

![](list.png)

```scala
scala> val list = List(1,2,3)
list: List[Int] = List(1, 2, 3)

scala> list.length
res74: Int = 3

scala> list(0)
res75: Int = 1

// 列表对象不支持 update 方法，无法更新值
scala> list(0)  = 1
<console>:13: error: value update is not a member of List[Int]
       list(0)  = 1
```

因为不可变，List对象行为类似 Java 中的字符串，可以使用 ":::"方法进行拼接，

```scala
scala> val list1 = List(1,2,3)
list1: List[Int] = List(1, 2, 3)

// 拼接返回新 List 对象
scala> val list2 = list ::: list1
list2: List[Int] = List(1, 2, 3, 1, 2, 3)
```

“::”  用于在列表前追加元素：

```scala
// 列表前追加
scala> val list2 = 1 :: list
list2: List[Int] = List(1, 1, 2, 3)

// 列表前追加不同类型，结果列表自动变为 "Any"
scala> val list2 = "s" :: list
list2: List[Any] = List(s, 1, 2, 3)
```

"::"是右操作元，所以调用的是右侧 list 对象的方法。通常一个方法名的最后一个字符是冒号，则该方法的调用会发生在它的右操作元上。

Nil 表示空列表，所以快速创建列表的方式为：

```scala
scala> Nil
res0: scala.collection.immutable.Nil.type = List()

// 结尾必须使用 Nil 对象以提供右操作
scala> var list = 1::2::Nil
list: List[Int] = List(1, 2)
```

推荐使用前插入，这可以复用现有列表的数据。尽管列表提供了 append 追加操作 “:+”，但是效率不高，列表越大，效率越低。

如果要大量进行追加，推荐使用可变列表 ListBuffer，支持追加操作，完成后再调用 toList 转换为支持函数式操作的不可变列表对象。

## 元组

与列表类似，元组也是不可变的，但是元素可以支持不同的类型。注意尽管列表可以定义为 Any 类型，看起来可以存放各种对象，但是它们均是 Any类型，而不是 Int，String类型。

```scala
scala> var list = List(1, "string")
list: List[Any] = List(1, string)

scala> list(1)
res6: Any = string
```

元组中可以存放不同类型的数据：

```scala
scala> val tuple = (1, "string")
tuple: (Int, String) = (1,string) // 类型是一个 tuple2 元组 (Int, String)
```

由于 apply 方法只能返回同一种类型元素，但是元组中元素是可以不同类型的，所以需要通过 ._n 的方式访问元素，n >=1，之所以从 1 开始值遵循了其他支持静态类型元组的语言设定的传统，例如 Haskell。

相对于List，元组只提供了较少的方法。

###多变量定义

Scala使用元组进行多变量定义和初始化：

```scala
scala> var (a,b,c) = (1,"str",false)
a: Int = 1
b: String = str
c: Boolean = false
```

变量个数和值的个数必须一致：

```scala
scala> var (a,b,c) = (1,"str")
<console>:14: error: constructor cannot be instantiated to expected type;
 found   : (T1, T2, T3)  // 变量类型类表
 required: (Int, String) // 值类型列表
       var (a,b,c) = (1,"str")
```

如果只取部分数据赋值，可以使用 _ 占位符，

```scala
// 忽略部分元素
scala> val (a,b,_) = (1,2,3)
a: Int = 1
b: String = hello

scala> val (a,_,b) = (1,2,3)
a: Int = 1
b: Int = 3
```

### 多值返回

元组的值不可改变。它可以用于函数返回不止一个值的情况。StringOps 的 partition 方法返回一对字符串，包含满足和不满足该条件的字符：

```scala
scala> "Hello World".partition(_.isUpper)
res109: (String, String) = (HW,ello orld)
```

## 映射

映射是键值对的集合，类似 Python 中的字典类型 dict。为了支持函数式编程风格，Scala 同时提供了映射（map）和 集合（Set）的可变和不可变对象版本的支持。

### 创建映射

```scala
// 创建不可变映射
scala> val scores = Map("Tom"->90, "Jick"->80)
scores: scala.collection.immutable.Map[String,Int] = Map(Tom -> 90, Jick -> 80)

// 创建可变映射
scala> val mutable_scores = scala.collection.mutable.Map("Tom"->90, "Jick"->80)
mutable_scores: scala.collection.mutable.Map[String,Int] = Map(Tom -> 90, Jick -> 80)
```

键值对的键在一个映射对象中是唯一的，如果出现重复则只保留最后一个:

```scala
scala> val scores = Map("Tom"->90, "Tom"->80)
scores: scala.collection.immutable.Map[String,Int] = Map(Tom -> 80)
```

可以使用圆括号代替 -> 符号：

```scala
scala> val scores = Map(("Tom", 90), ("Jick", 80))
scores: scala.collection.immutable.Map[String,Int] = Map(Tom -> 90, Jick -> 80)
```

### 获取映射值

Scala中很多对象操作都类似于函数，也即使用 () 符号。这在函数和映射对象上体现尤为明显。

```scala
scala> val TomScore = scores("Tom")
TomScore: Int = 90
```

如果映射中没有对应的键，则报异常：

```scala
scala> val tomScore = scores("Jack")
java.util.NoSuchElementException: key not found: Jack
  at scala.collection.immutable.Map$Map2.apply(Map.scala:138)
  ... 28 elided
```

可以使用 contains 方法检查是否包含某个建：

```scala
scala> scores.contains("Tom")
res74: Boolean = true

scala> scores.contains("Jack")
res75: Boolean = false

scala> val TomScore = if(scores.contains("Tom"))scores("Tom")else 0
TomScore: Int = 90
```

这种检测获取的方法被封装为 getOrElse 方法：

```scala
scala> val TomScore = scores.getOrElse("Tom", 0)
TomScore: Int = 90

scala> val JackScore = scores.getOrElse("Jack", 0)
JackScore: Int = 0
```

直接使用 get方法则会返回一个 Option 对象，值要么为 Some，要么为 None：

```
scala> val JackScore = scores.get("Jack")
JackScore: Option[Int] = None

scala> val TomScore = scores.get("Tom")
TomScore: Option[Int] = Some(90)

```

### 更新映射值

只有可变映射对象才可更新值：

```scala
scala> mutable_scores("Tom") = 70

scala> mutable_scores
res80: scala.collection.mutable.Map[String,Int] = Map(Tom -> 70, Jick -> 80)

// 新增键值对
scala> mutable_scores("Jack") = 70

scala> mutable_scores
res82: scala.collection.mutable.Map[String,Int] = Map(Jack -> 70, Tom -> 70, Jick -> 80)
```

也可以使用类似可变变组扩充的 +-符号：

```scala
// 添加键值对，如果键已经存在，则更新值
scala> mutable_scores += ("Bill"-> 100)
res83: mutable_scores.type = Map(Bill -> 100, Jack -> 70, Tom -> 70, Jick -> 80)

// 删除键值对
scala> mutable_scores -= "Bill"
res84: mutable_scores.type = Map(Jack -> 70, Tom -> 70, Jick -> 80)
```

### 迭代映射

```scala
scala> for ((k,v) <- scores)println(k + "'s score is " + v)
Tom's score is 90
Jick's score is 80
```

如果只需要查询键或者值，可以用 keySet 和 values 方法。

```scala
scala> scores.keySet
res90: scala.collection.immutable.Set[String] = Set(Tom, Jick)

scala> for(i <- scores.keySet) println(i)
Tom
Jick
```

keySet 返回一个集合，显然集合中每个元素都是唯一的；values 方法返回一个可迭代对象：

```scala
scala> scores.values
res94: Iterable[Int] = MapLike.DefaultValuesIterable(90, 80)

scala> for(i <- scores.values) println(i)
90
80
```

如果要翻转映射：也即交换键和值的位置，可以使用 yield 生成器：

```scala
scala> for ((k,v) <- scores) yield (v, k)
res96: scala.collection.immutable.Map[Int,String] = Map(90 -> Tom, 80 -> Jick)
```

### zip 操作

拉链操作（zip）可以把多个值绑定在一起，进行统一处理。

```scala
scala> val names = Array("Tom", "Jack", "Bill")
names: Array[String] = Array(Tom, Jack, Bill)

scala> val scores = Array(80, 90, 70)
scores: Array[Int] = Array(80, 90, 70)

// zip 返回元组的数组
scala> val pairs = names.zip(scores)
pairs: Array[(String, Int)] = Array((Tom,80), (Jack,90), (Bill,70))
```

可以使用 for 循环迭代处理：

```scala
scala> for ((name, score)<-pairs)println(name + "'s score is " + score)
Tom's score is 80
Jack's score is 90
Bill's score is 70
```

如果可以确定元组中的第一个组元是唯一的，那么可以转换成映射：

```scala
scala> val map = names.zip(scores).toMap
map: scala.collection.immutable.Map[String,Int] = Map(Tom -> 80, Jack -> 90, Bill -> 70)
```

## 集合

使用 Set 类创建集合（没有重复元素），类似映射，集合分为可变和不可变两个版本，通过 "+" 向集合内添加新元素， 默认是不可变集合，+ 方法会创建并返回一个新的包含了新元素的集合。

```scala
scala> var results = Set("pass", "fail")
results: scala.collection.immutable.Set[String] = Set(pass, fail)

// 这里的 += 也是方法，等价于 results.+=("unknown")
scala> results += "unknown"

scala> results // 被重新赋值为新集合
res2: scala.collection.immutable.Set[String] = Set(pass, fail, unknown)
```

### 可变集合

需要明确导入 scala.collection.mutable 包，然后调用 mutable.Set：

```
scala> import scala.collection.mutable
import scala.collection.mutable

scala> val human = mutable.Set("male")
human: scala.collection.mutable.Set[String] = Set(male)

scala> human += "female"
res3: human.type = Set(male, female)
```

### 访问集合

类似映射，集合中的元素是无序的。换句话说，不能以索引的方式访问集合中的元素。  判断某一个元素在集合中比Seq类型的对象要快。

 ```scala
scala> human.foreach(println)
male
female

// 判定是否包含元素
scala> human.contains("male")
res29: Boolean = true
 ```

### 集合操作

由于 + 和 - 方法用于元素的增加和删除操作，所以要使用 ++ 和 -- 符号进行集合间操作。集合具有交并操作，和去除操作：

```scala
val a = Set(1,2,3)
val b = Set(1,4,5)

// 并集
scala> val c = a ++ b
c: scala.collection.immutable.Set[Int] = Set(5, 1, 2, 3, 4)

scala> val d = a | b
d: scala.collection.immutable.Set[Int] = Set(5, 1, 2, 3, 4)

// a 去除 b 中元素
scala> val e = a -- b
e: scala.collection.immutable.Set[Int] = Set(2, 3)

// 交集
scala> val f = a & b
f: scala.collection.immutable.Set[Int] = Set(1)
```

##函数式编程

### 副作用函数

副作用（side-effects ）的概念：一个带有副作用的函数不仅只是简单的返回一个值，还干了一些其他的事情（返回值是它的一个作用，而其他的作用就是副产品，所以称为副作用），比如: 

- 修改了外部变量

- 修改了数据结构
- 设置一个对象的成员
- 抛出一个异常或以一个错误终止
- 打印到终端或读取用户的输入
- 读取或写入一个文件
- 在屏幕上绘画

当函数没有副作用，那么我们就说这个函数符合函数式编程（FP），也即是纯函数。 函数式编程强调没有"副作用"，意味着函数要保持独立，所有功能就是依据参数值返回一个新的值，没有其他行为，不会破坏程序运行环境，不会影响自身和其他函数的运行。

以下函数是纯函数：

使用外部不可变变量（常量），也即可以使用外界常量值，不可修改外界变量值（例如入参或全局变量）：

```scala
val PI = 3.14
def area(radius :Double):Double={
    PI * radius * radius
}
```

内部使用不对外暴露的变量：

```scala
def addOne(i: Int) = {
  var s = i
  s = s + 1
  s
}
```

纯函数在给定的输入参数下，总是返回相同的结果，并且没有产生副作用的动作：例如对入参进行修改（修改外界的值），IO 操作等。

###函数式

指令式编程风格如下：

```scala
def printIntArray(array: Array[Int]): Unit ={
	for (i <- 0 until array.length)
		println(array(i))
}
```

函数式编程风格则要简便得多：

```scala
def printIntArray(array: Array[Int]): Unit ={
	array.foreach(println)
}
```

然而重构后的 printIntArray 并不是纯函数式函数，因为它有副作用（向标准输出流打印）。带有副作用的函数的标志性特征是结果类型为 Unit。

函数式编程的做法是返回一个已经格式化的字符串。

```scala
def formatIntArray(array: Array[Int]):String = {
	array.mkString("\n")
}
```

有副作用的函数完成与外界环境的交互，而无副作用的函数用于纯计算和处理。无副作用的程序更容易测试，例如 formatIntArray 只需要比对结果即可。

### 函数编程优点

函数式编程主要可以为当前面临的三大挑战提供解决方案。

1.  是并发的普遍需求。有了并发，我们可以对应用进行水平扩展，并提供其对抗服务器故障的能力。所以，如今并发编程已经是每个开发者的必备技能了。 
2. 是编写数据导向（如“大数据”）程序的要求。当然，从某种意义上说，每个程序都与数据密切相关，但如今大数据的发展趋势，使得有效处理海量数据的技术被提高到了更重要的位置。 
3. 是编写无 bug 的程序的要求。这个挑战与编程本身一样古老，但函数式编程从数学的角度为我们提供了新的工具，使我们向无 bug 的程序又迈进了一步。    

状态不可变这一特点解决了并发编程中最大的难题，即对共享的可变状态的访问问题。因 此，编写状态不可变的代码就称为编写健壮的并发程序的必备品，而拥抱函数式编程就是 写出状态不可变代码的最好途径。状态不可变，以及严密的函数式编程思想有其数学理论 为基础，还能减少程序中的逻辑错误。 

在函数式编程中，函数是第一等级的值，就像数据变量的值一样。你可以：

- 从函数中组合形成新函数（如 tan(x)=sin(x)/cos(x)）
- 可以将函数赋值给变量
- 可以将函数作为参数传递给其他函数
- 可以将函数作为其他函数的返回值。    

当一个函数采用其他函数作为变量或返回值时，它被称为**高阶函数**。在数学中，微积分中有两个高阶函数的例子— —微分与积分。我们将一个表达式作为函数传给“微分函数”，然后微分函数返回了一个新函数，即原函数的导数。

###递归

递归是函数式编程的一大特点，特别是尾递归（tail recursion），可以由编译器自动优化成非递归实现，从而防止栈溢出。

一个链表的求和操作可以如此实现：

```scala
def listSum(list: List[Int]): Int = {
	list match{
		case Nil => 0
		case _ => list.head + listSum(list.tail)
	}
}

// case 语句支持模式匹配，以下实现是等价的
def listSum(list: List[Int]): Int = {
	list match{
		case Nil => 0
		case head::tail => head + listSum(tail)
	}
}
```

match常常替代 if else 语句来匹配不同条件，实现递归算法的基础在于算法的出口（结束）和迭代部分。空链表 Nil 的和为 0，而非空链表的求和总是等于头部元素加上尾部元素的和。

上述写法不是尾递归的，当链表很大时可能导致 `StackOverflowError`  异常。在普通递归中，典型的模型是首先执行递归调用，然后获取递归调用的返回值并计算结果。以这种方式，在每次递归调用返回之前，均不会得到计算结果，所以程序要不停压栈，最终导致栈溢出。

在尾递归中，**递归调用的分支总是调用自身，而无需再等待自身返回并做额外的运算**，上例中由于每次都要等待 listSum(tail) 返回然后再加上 head，所以要不停压栈，而尾递归可以释放当前函数的栈帧。

```scala
// 尾递归实现链表求和
def tailSum(list: List[Int], total:Int = 0): Int = {
	list match{
		case Nil => total
		case head :: tail => tailSum(tail, total + head) // 无需返回，可以释放本次调用栈
	}
}

// 长链表不会导致栈溢出
println(tailSum((1 to 10000).toList))
```

将普通递归改写为尾递归的关键点在于添加一个记录中间结果的参数，这样进入递归分支就无需回退计算最终结果了。

直接实现尾递归有时候比较困难，首先实现普通递归，然后改写为尾递归更具可行性。另外为了不对外暴露 total 这一临时变量，可以在函数外部再次封装一层：

```scala
def tailSum(list: List[Int]): Int = {
	@tailrec //尾对作为内部函数
	def _tailSum(list: List[Int], total:Int): Int = {
		list match{
			case Nil => total
			case head :: tail => _tailSum(tail, total + head)
		}
	}
	
	_tailSum(list, 0)
}
```

可以直接为函数添加 @tailrec 修饰符，让编译器进行显式优化，如果函数不是尾递归的，则会报错，这一方法可以直接验证某函数是否尾递归的。



##集合常见操作

常见的集合类型——序列、列表、集合、数组、树及其他类似的类型，都支持基于只读遍历的通用操作。

特别是你实现的某个“容器” 类型也支持这些操作的情况。例如 Option 是包含零个 None 或一个 Some 元素的容器。    







## 类和对象

Scala中类用关键字 class 声明，而单例对象用关键字 object 声明。因此，使用 “实例”一词指代一般的类实例对象，“实例”和“对象”在大多数 OO 语言中通常是 同义的。    

### 类

Scala 中的类与 Java 中类定义类似，但是默认权限为 public ，且无需显式定义带参构造函数。

```scala
// class 定义类，默认访问权限为 public
scala> class PrintInt{}
defined class PrintInt

// new 实例化类，生成类的对象
scala> var pi = new PrintInt
pi: PrintInt = PrintInt@2520303a
```

####类成员

类成员（member）包括字段（field，也称为属性）和方法（method）。字段分为 var 定义的变量和 val 定义的常量为所有实例对象共享，var 字段每个实例独有，具有不同的内存空间。

```scala
// 如果不适用 var 声明，则默认参数为 val
scala> class PrintInt{var intVal = 0; val VAL = 100}
defined class PrintInt

scala> var PI = new PrintInt()
PI: PrintInt = PrintInt@4773998c

// 所有类的实例对象共享 val 字段
scala> PI.VAL
res60: Int = 100

// 不显式指定参数是变量还是常量，则默认定义为 private val 型，无法通过 p.name 直接访问成员
scala> class Person(name:String, age:Int) 
```

需要注意：

- Java 会自动初始化字段，Scala 必须给类的字段指定初始值。
- Java 中需要显式声明 public 字段或方法，Scala 默认访问权限就是 public 的。
- 如果没有任何修饰符，则该field是完全私有的（private val），既无getter也无setter，只在类的内部可读。 
- Scala 方法参数的均为 val 类型，不可在方法中对入参赋值。
- Scala 方法总是有返回值的，值为最后一条语句的结果，无需显式的 return语句。

```scala
class PrintInt {
    // 定义私有字段
	private var intVal = 0
	val VAL = 100
	
    // 参数总是 val 类型，不可在方法中对它赋值
	def addInt(num:Int = 1): Unit ={
		intVal += num
	}
}
```

方法的返回值应该显式地声明，否则代码很难被识读，返回值需要分析代码进行推断。

addInt 返回 Unit，其副作用修改了 intVal，带有副作用的方法被称作过程（procedure）。

#### 类构造器

构造器用于类的实例化。Scala 中没有样板构造器，只有构造参数，首先看一个示例：

```scala
// var 声明的构造参数称为类的可变参数，且是 public 的
class Person(var name: String, var age: Int)

object learnScala {
	def main(args: Array[String]): Unit = {
		println(new Person("Bill", 10).name)
	}
}
```

以上类的定义等价于 Java 如此冗长的定义：

```java
public class JPerson {
	private String name;
	private int age;
    
    // Java 的构造函数
	public JPerson(String name, int age) {
		this.name = name;
		this.age = age;
	}
    public void setName(String name) { this.name = name; }
    public String getName() { return this.name; }
    public void setAge(int age) { this.age = age; }
    public int getAge() { return this.age;}
}
```

一个实例可以使用 this 关键字指代它本身。尽管在 Java 代码中经常看到 this 的这种用 法， Scala 代码中却很少看到。    

在构造参数前加上 val，使得该参数成为类的一个不可变字段。case 关键字可以让所有参数成为类的不可变字段：

```scala
// case 定义不可变字段
case class ImmutablePerson(name: String, age: Int)

object learnScala {
	def main(args: Array[String]): Unit = {
		val IP = new ImmutablePerson("Bill", 10)
		println(IP.name)
	}
}
```

对于 case 类，编译器自动生成一个伴随对象。

#### 主构造器

主构造器（Primary Constructor），一个Scala 类可以有1个主构造器以及任意多个辅助构造器（Auxiliary）构造器。

```scala
class <class-name>(params-list){
    // Class Body，类体，整个大括号包含的部分
}
```

类定义（Class Definition）格式如上所示，主构造器就是类定义和类体中的所有可执行语句。可以把类定义看做构造函数。

```scala
// 定义了一个无参数的类主构造器 Person()，等价于 Java 中的默认构造器，可以省去空类体的大括号
class Person{
      
}

// 使用主构造器获取类实例，无参数构造器可省略小括号，以下两句等价
val man = Person()
val woman = Person
```

同理带参的类定义，等价于同时定义了一个带参的主构造器 Person(var name:String, var age:Int)：

```scala
class Person(var name: String, var age: Int)

val man = Person("Jack", 10)
```

为了深入理解，可以反编译 scala 的字节码文件：

```shell
> scalac Person.scala
> javap -p Person.class
Compiled from "Person.scala"
public class Person {
  # 自动定义了参数对应的成员变量
  private java.lang.String name;
  private int age;
  
  # getter 和 setter 方法
  public java.lang.String name();
  public void name_$eq(java.lang.String);
  public int age();
  public void age_$eq(int);
  
  # 重载带参构造函数
  public Person(java.lang.String, int);
}
```

 实际上 getter 方法和 setter 方法生成如下：

```scala
// Scala 中的 getter 和 setter 方法命名
public void name_$eq(String x$1)
{
    name = x$1;
}

public int age()
{
    return age;
}
```

这与 Java 中默认的 getter 和 setter 方法命名并不一致：

```java
// Java 中的 getter 和 setter 方法命名
public int getA() {
    return a;
}

public void setA(int a) {
    this.a = a;
}
```

当使用 "= " 为实例成员变量赋值时，Scala 编译器自动把 "=" 转换为 "_$eq"。例如：

```scala
// Person.scala
class Person(var name: String, var age: Int)
class testPerson {
  val man = new Person("Jack", 10)
  man.name = "Tom"
}
```

编译，然后反编译 testPerson.class 文件：

```java
public class testPerson
{
    public Person man()
    {
        return man;
    }

    public testPerson()
    {   
        // 将 man.name = "Tom" 转换为  man().name_$eq("Tom")
        man().name_$eq("Tom");
    }

    private final Person man = new Person("Jack", 10);
}
```

####辅助构造器

辅助构造器（Auxiliary Constructor）类似 Java 中定义的多个构造器，可以重载默认的构造器：

```scala
class Person(var name:String, var age:Int){
    // 只需要指定名字
	def this(name:String){
		this(name, 20)
	}
	// 只需要指定年龄
	def this(age:Int){
		this("Jack", age)
	}
}

object learnScala {
	def main(args: Array[String]): Unit = {
		println(new Person("Bill").age)
		println(new Person(10).age)
		println(new Person("Bill", 10).age)
	}
}
```

辅助构造器也叫次级构造器，可以带部分参数，然后通过 this 调用主构造器。

需要注意的是，辅助构造被命名为 this，它的**第一个表达式必须调用主构造器或其他辅助构造器**。编译器还要求被调用的构造器在代码中先于当前构造器出现。所以，我们在代码 中必须小心地排列构造器的顺序。 通过强制让所有构造器最终都调用主构造器，可以将代码冗余最小化，并确保新实例的初始化逻辑的一致性。    

**只有主构造器才能调用超类的方法**，辅助构造器必须首先调用主构造器或者其他构造器，这和 Java 不同。

#### 样例类

样例类（Case Class）也被称为案例类，除了自动生成主构造器外，还会自动生成伴生对象，并在伴生对象中实现 apply 函数，所以无需使用 new 实例化，默认类参数为 var 类型，无需声明。

一个普通类定义如下，默认参数为 private val 类型，不能直接访问：

```scala
// 普通类的参数默认为 private
scala> class Person(name:String, age:Int)
defined class Person

// 普通类需要 new 实例化
scala> val p = new Person("Jack", 20)
p: Person = Person@e276013

// 不可直接访问 private 类成员
scala> p.name
<console>:14: error: value name is not a member of Person
```

与普通类不同，样例类默认类参数（主构造器参数）为 public val 类型，可以直接实例化，并直接访问成员变量：

```scala
// 使用 case 关键字定义样例类，类参数默认为 public val 类型
scala> case class Person(name:String, age:Int)
defined class Person

// 样例类使用函数式方式实例化，等价于 val p = new Person("Jack", 20)
scala> val p = Person("Jack", 20)
p: Person = Person(Jack,20)

// 可以直接访问成员
scala> p.name
res154: String = Jack
```

编译器会为样例类自动生成伴随对象，其中包括 apply 方法和 unapply 方法，代码如下：

```scala
object Person { 
    def apply(name:String, age:Int) = { 
        new Person(name, age) 
    }
    
    def unapply(p:Person): Option[(String, age)] = {
        Some((p.name, p.age)) 
    }
}

// 调用伴随对象的 apply 函数
scala> val p = Person("Jack", 20)
p: Person = Person(Jack,20)

// 调用伴随对象的 unapply 函数
scala> Person.unapply(p)
res159: Option[(String, Int)] = Some((Jack,20))
```

- apply 方法它会在创建实例时调用。
- unapply方法会在case class的模式匹配时使用，这个方法主要是把一个case class 实例的进行“拆解”（unwrap)返回它的参数，类型为 Option 的子类 Some，以便进行模式匹配时使用。
- getter 方法，用于获取属性值。
- 此外还会自动生成 copy，equals 和hashCode方法，用于复制，比较对象。
- toString 方法用于代码调试。

使用 copy 方法更新对象属性是函数式编程的特征之一：

```scala
scala> val boyJack = Person("Jack", 10)
boyJack: Person = Person(Jack,10)

// 更新年龄，在 copy 方法内指明要更新的属性名和新值
scala> val oldJack = boyJack.copy(age=50)
oldJack: Person = Person(Jack,50)
```

可以说样例类是Scala 为模式匹配专门设计的类类型。也即当需要对某类实例进行模式匹配时，在类定义时添加 case 关键字。

###模式匹配

当模式匹配设计到类实例匹配的时候，就会调用类的 unapply 函数，所以模式匹配类对象时，它的类型应定义为样例类。

case 后的语句被称为匹配模式，=> 后的语句被称为执行语句 。

#### 通配模式

通配模式(_)会匹配任何对象：

```scala
val str = "ok"
str match {
    case "ok" => println("Good!")
    case _ => println(str) // 默认处理分支，不能直接打印 _，例如 println(_)
}
```

统配模式用于防止没有匹配的输入，过滤某个对象并不关心的局部。通配模式同样可以组合使用，来表示一个复杂的对象。

#### 常量模式

常量模式仅匹配自己。任何字面量都可以作为常量使用，例如 1,true,"hello"；任何 val 或单例对象也可以被当做常量使用，例如 Nil 匹配空列表。

```scala
import math.{E,Pi}
def describe(x:Any) = x match{
    case 5 => "five"
    case true => "truth"
    case "Hello" => "hi!"
    case Nil => "Empty list"
    case Pi => Pi // 可以匹配常量 math.Pi
    case _ => "something else"
}
```

如果新定义了一个名称引用常量，那么如果要使用它作为常量名匹配，则需要反引号包起来，编译器将把它解读为常量，而不是变量模式：

```scala
import math.{E,Pi}
val mypi = Pi
def describe(x:Any) = x match{
	case `mypi` => mypi // 反引号括起来
	case _ => "something else"
}
```

#### 变量模式

变量模式匹配任何对象，不过不同于通配模式，Scala 会将对应的变量绑定到匹配上的对象。这样就可以使用变量来访对象了。

```scala
def testMatch(x: Any): Any = x match {
	case 0 => "zero"
	case "true" => 1
	case other => println(other) // other 将匹配所有不是 "ok" 字符串的情况，可以引用它
}
println(testMatch(2))
```

#### 带类型模式

```scala
def generalSize(x: Any) = x match{
	case s:String => s.length // 匹配字符串类型
	case m: Map[_,_] => m.size
	case _ => -1
}
```

#### 序列模式

匹配以 0 开头的三个元素的列表：

```scala
expr match {
	case List(0, _, _) => println("found it")
	case _ =>
}
```

匹配以 0 开头的任意长度列表：

```scala
expr match {
    case List(0, _*) => println("found it")
    case _ =>
}
```

#### 元组模式

形如(a,b,c)这样的模式能匹配任意的三元组：

```scala
def tupleDemo(expr: Any) = {
    expr match {
        case (a, b, c) => println("matched " + a + b + c)
        case _ =>  // 返回 Unit 类
    }
}
```

#### 样例类模式

模式匹配会调用样例类的 unapply 方法将对象转换为 Some 类型，进行比较：

```scala
object Demo {
   case class Person(name: String, age: Int)
   def main(args: Array[String]) {
      val alice = new Person("Alice", 25)
      val bob = new Person("Bob", 32)
      val charlie = new Person("Charlie", 32)

      for (person <- List(alice, bob, charlie)) {
         person match {
            case Person("Alice", 25) => println("Hi Alice!")
            case Person("Bob", 32) => println("Hi Bob!")
            case Person(name, age) => println(
               "Age: " + age + " year, name: " + name + "?")
         }
      }
   }
}
```

###eta扩展

eta 扩展可以将类方法转化为函数，以用在 Map中。

```scala
scala> def sum(a: Int, b: Int) = a + b
sum: (a: Int, b: Int)Int

// _ 符号实现 eta 扩展
scala> sum _
res175: (Int, Int) => Int = $$Lambda$1947/845305433@6ec9aa1
```

方法可以作为参数传递给高阶函数，此时不需要显式转换。

###单例对象

Scala 类中不允许静态（static）成员，所以无法直接使用类名调用类方法。与此对应，Scala 提供单例对象（singleton object），使用 object 关键字定义。

单例对象在第一次被访问时才会被初始化，来自于scala自带的predef包。 单例对象不能接收参数，也不能用 new 实例化单例对象。

```scala
object SingletonObject{
	val value = 10 // 默认权限为 public
	def SaySingleObject(){
		println("Hello, This is Singleton Object" + value)
	}
}

object learnScala {
	def main(args: Array[String]): Unit = {
		println(SingletonObject.value)    // 访问公有属性
		SingletonObject.SaySingleObject() // 调用公有方法
	}
}
```

###伴生对象

当在同一源码文件中定义类，并且定义同名的单例对象时，它被称为这个类的伴生对象（companion object）。与此同时，类被称为这个单例对象的伴生类（companion class）。类和它的伴生对象可以互相访问对方的私有成员。

```scala
class MyClass {
    def printHiddenValue() = {
        println(SomeClass.HIDDEN_VALUE) // 访问伴生对象中的私有字段
    }
}

object MyClass {
    private val HIDDEN_VALUE = 10
}
```

在伴生对象中定义 apply 函数，可以不再通过 new 关键字实例化类对象：

```scala
class Person {
    private var name = ""
}

object Person {
    def apply(name: String): Person = {
        val p = new Person // 这里创建的是伴生类的实例，而不是伴生对象的实例
        p.name = name
        p
    }
}

// 等价于 Person.apply("Jack")
val man = Person("Jack")
```

伴生对象作为 "static" 对象的容器，可以定义一些常量，或者共享变量：

```scala
class Person(var name:String, val sex:String){
    // 私有成员 ID，只可通过 public 接口访问
	private var id:Int = 0
	
	this.id = Person.createId()
	
	def show(): Unit ={
		println(this.id + ":\t" + this.name + "\t" + this.sex)
	}
}

object Person{
	// 常量
	val SEX_TYPE_MALE = "male"
	val SEX_TYPE_FEMALE = "female"
	
	// 变量，自动生成唯一的 ID
	private var lastId:Int = 0
	
	def createId():Int={
		val id = lastId
		lastId += 1
		id
	}
}

object learnScala {
	def main(args: Array[String]): Unit = {
		val p1 = new Person("Bill", Person.SEX_TYPE_MALE)
		val p2 = new Person("Lily", Person.SEX_TYPE_FEMALE)
		
		p1.show()
		p2.show()
	}
}

// 示例打印结果：
/*
0:	Bill	male
1:	Lily	female
*/
```



###孤立对象

没有同名的伴生类的单例对象称为孤立对象（standalone object）。孤立对象用途很广，例如入口程序，将工具方法归集在一起。

Scala的入口函数 main 就是作为类中单例对象存在的：

```scala
object learnScala {
	def main(args: Array[String]): Unit = {
		println("Hello world!")
	}
}
```

Scala 在每一个Scala 源码文件都隐式地引入了 java.lang和scala 包的成员，以及名为 Predef 的单例对象的所有成员。Perdef 中定义了很多有用的方法，例如 println和assert。

```scala
import Predef._ // 默认导入预定义单例中的所有方法

// Predef.scala 中对 println 的定义：
  def println(x: Any) = Console.println(x)
```





