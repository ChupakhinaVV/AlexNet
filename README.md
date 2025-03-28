# LR1
# Теоретическая база
Глубокое обучение является передовой областью исследований машинного обучения (machine learning — ML). Deep learning (глубокое обучение) — это вид машинного обучения с использованием многослойных нейронных сетей, которые самообучаются на большом наборе данных.

Концепция глубокого обучения (Deep Learning — DL) впервые появилась в 2006 году как новая область исследований в машинном обучении. Вначале оно было известно как иерархическое обучение в [1], и как правило оно включало в себя множество областей исследований, связанных с распознаванием образов. Глубокое обучение в основном принимает в расчет два ключевых фактора: нелинейная обработка в нескольких слоях или стадиях и обучение под наблюдением или без него [2]. Нелинейная обработка в нескольких слоях относится к алгоритму, в котором текущий слой принимает в качестве входных данных выходные данные предыдущего слоя. Иерархия устанавливается между слоями, чтобы упорядочить важность данных, полезность которых следует установить. С другой стороны, контролируемое и неконтролируемое обучение связано с меткой классов целей: ее присутствие подразумевает контролируемую систему, а отсутствие — неконтролируемую.

Функционально нейросети делят на слои — структуры нейронов с общей задачей (рисунок 1) [3].

![image](https://github.com/user-attachments/assets/7b0e1da1-afeb-44d4-a87e-0a4013ec6e1e)

<p align="center">  
Рисунок 1 - Слои
</p>

Входной слой получает набор данных. В простейшем случае каждый нейрон отвечает за один параметр. Например, в нейросетях для прогнозирования цен номеров в отеле это будут название отеля, категория номера и дата заезда. Информацию по этим параметрам входной слой отдает на скрытые слои.

Скрытые слои производят вычисления на основе входящих параметров. В глубоком обучении у нейронных сетей несколько скрытых слоев. Это позволяет нейросети находить больше взаимосвязей во входных данных. Связи между нейронами имеют свой вес — долю значимости параметра среди всехданных. Например, в подсчете цен номеров большой вес будет иметь дата заезда, поскольку отели меняют цены в зависимости от спроса в конкретный день.

Выходной слой выводит результат вычислений, например, цены номеров в отелях.

В глубоком обучении используется больше одного скрытого слоя. Такие модели называют глубокими нейронными сетями (deep neural network). Например, в компьютерном зрении используют сверточные нейросети. В архитектуре таких нейросетей используют множество слоев, подбирая их количество под каждую задачу. Чем дальше информация со входного изображения продвигается по нейросети, тем более абстрактные детали находит нейросеть. Например, на первых слоях модель находит палочки и круги, из которых состоит любое изображение, а в конце сеть уже может найти хвосты и уши для распознавания животных на фотографиях.

Для разных задач используют свои модели глубокого обучения: 

●	Свёрточные нейронные сети (CNN) используют для обработки изображений. Такие сети содержат слои, которые выделяют ключевые признаки из входных данных с помощью математической операции — свёртки. Отсюда и такое название. Сеть учится различать объекты и паттерны в изображениях, что полезно для классификации, обнаружения объектов и сегментации изображений. Например, в изображении кошки свёрточные слои могут выделить морду, уши и полосатый окрас. 

●	Рекуррентные нейронные сети (RNN) могут запоминать предыдущие состояния, что делает их подходящими для задач, где важен контекст и зависимость от предыдущих шагов. По такому же принципу человек читает каждое слово в предложении, запоминает и поэтому понимает смысл. RNN используются в машинном переводе, анализе текстов, генерации речи и других областях, где важна последовательная структура данных.

●	Генеративно-состязательные сети (GAN) состоят из двух частей: генератора и дискриминатора. Генератор создаёт новые данные, например изображения или текст, а дискриминатор сравнивает эти данные с реальными. Обучение такой нейросети — это постоянная борьба художника и критика. Генератор стремится создавать всё более реалистичные данные, а дискриминатор — отличать их от настоящих. GAN используют для генерации реалистичных изображений, улучшения качества данных, создания текста и других задач, где нужен контент, близкий к реальности. 

# Описание разработанной системы (алгоритмы, принципы работы, архитектура)

AlexNet — свёрточная нейронная сеть для классификации изображений. AlexNet — сверточная нейронная сеть, которая оказала большое влияние на развитие машинного обучения, в особенности — на алгоритмы компьютерного зрения. Сеть с большим отрывом выиграла конкурс по распознаванию изображений ImageNet LSVRC-2012 в 2012 году (с количеством ошибок 15,3% против 26,2% у второго места).

Архитектура AlexNet схожа с созданной Yann LeCum сетью LeNet. Однако у AlexNet больше фильтров на слое и вложенных сверточных слоев. Сеть включает в себя свертки, максимальное объединение, дропаут, аугментацию данных, функции активаций ReLU и стохастический градиентный спуск.

Особенности AlexNet:

● Как функция активации используется Relu вместо арктангенса для добавления в модель нелинейности. За счет этого при одинаковой точности метода скорость становится в 6 раз быстрее.

● Использование дропаута вместо регуляризации решает проблему переобучения. Однако время обучения удваивается с показателем дропаута 0,5.

● Производится перекрытие объединений для уменьшения размера сети. За счет этого уровень ошибок первого и пятого уровней снижаются до 0,4% и 0,3%, соответственно.

**Архитектура Alexnet** (рисунок 2) [4]:

![image](https://github.com/user-attachments/assets/e293532a-7b2f-42df-b593-31ead45d42e6)
<p align="center"> 
Рисунок 2 - Архитектура AlexNet
</p>  

AlexNet содержит восемь слоев с весовыми коэффициентами. Первые пять из них сверточные, а остальные три — полносвязные. Выходные данные пропускаются через функцию потерь softmax, которая формирует распределение 1000 меток классов. Сеть максимизирует многолинейную логистическую регрессию, что эквивалентно максимизации среднего по всем обучающим случаям логарифма вероятности правильной маркировки по распределению ожидания. Ядра второго, четвертого и пятого сверточных слоев связаны только с теми картами ядра в предыдущем слое, которые находятся на одном и том же графическом процессоре. Ядра третьего сверточного слоя связаны со всеми картами ядер второго слоя. Нейроны в полносвязных слоях связаны со всеми нейронами предыдущего слоя.

Таким образом, AlexNet содержит 5 сверточных слоев и 3 полносвязных слоя. Relu применяется после каждого сверточного и полносвязного слоя. Дропаут применяется перед первым и вторым полносвязными слоями. Сеть содержит 62,3 миллиона параметров и затрачивает 1,1 миллиарда вычислений при прямом проходе.  Сверточные слои, на которые приходится 6% всех параметров, производят 95% вычислений.

Оптимизаторы — важный компонент архитектуры нейронных сетей. Они играют важную роль в процессе тренировки нейронных сетей, помогая им делать всё более точные прогнозы. 

Оптимизаторы определяют оптимальный набор параметров модели, таких как вес и смещение, чтобы при решении конкретной задачи модель выдавала наилучшие результаты.

Самой распространённой техникой оптимизации, используемой большинством нейронных сетей, является алгоритм градиентного спуска.

Большинство популярных библиотек глубокого обучения, например PyTorch и Keras, имеют множество встроенных оптимизаторов, базирующихся на использовании алгоритма градиентного спуска, например SGD, Adadelta, Adagrad, RMSProp, Adam и пр.

**AdaSmooth** — это метод стохастической оптимизации, разработанный для адаптивного изменения скорости обучения в процессе обучения моделей машинного обучения. Основная цель AdaSmooth — устранить необходимость тщательной настройки гиперпараметров, таких как скорость обучения, которые часто требуют значительных усилий и опыта [5].

В отличие от других методов, чувствительных к выбору гиперпараметров, AdaSmooth автоматически подстраивает скорость обучения для каждого параметра модели, делая процесс обучения более стабильным и эффективным. Это достигается за счет использования информации о предыдущих шагах обновления и градиентов, что позволяет динамически регулировать шаги оптимизации.

Экспериментальные результаты показывают, что AdaSmooth эффективно работает на практике и сравним или превосходит другие стохастические методы оптимизации в задачах обучения нейронных сетей. Кроме того, метод обладает низкими требованиями к памяти и прост в реализации, что делает его привлекательным для широкого спектра приложений в области машинного обучения.

# Результаты работы и тестирования системы

В результате выполнения лабораторной работы была реализована нейронная сеть AlexNet с оптимизатором AdaSmooth. Архитектура реализована с использованием Torch.
Результат обучения нейронной сети AlexNet с оптимизаторами Adam и AdaSmooth приведен на рисунках 3-4.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4ba3aa85-baaa-4aa4-b9f8-5ef2c4318054">
</p>

<p align="center">
Рисунок 3 - Результат обучения нейронной сети с оптимизатором Adam
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/371ce5c7-6410-482f-ab39-98b242d31438">
    <img src="https://github.com/user-attachments/assets/dd507221-0981-4cd2-85a6-38aa0d1a47a9">
</p>

<p align="center">
Рисунок 4 - Результат обучения нейронной сети с оптимизатором AdaSmooth
</p>

Графики потерь и точности приведены на рисунках 5-6.


![image](https://github.com/user-attachments/assets/e08eb3d0-3834-4373-a065-664e93cbf0d4)


<p align="center">
Рисунок 5 - График потерь
</p>

![image](https://github.com/user-attachments/assets/81543e30-cd9f-42bc-b377-cb3d0359f61e)


<p align="center">
Рисунок 6 - График точности
</p>


# Выводы по работе

В результате выполнения лабораторной работы была реализована нейронная сеть AlexNet с оптимизатором AdaSmooth. Архитектура реализована с использованием Torch.

Точность модели с оптимизаторам **Adam** составила 12.63%.

Точность модели с оптимизатором **AdaSmooth** составила 66.14%.

Таким образом, **AdaSmooth** оказался более устойчивым к плато и лучше адаптировался к сложной многоклассовой задаче. Это позволило добиться гораздо большей обучаемости модели.
**Adam** не смог эффективно обучить модель, что может быть связано с большим количеством классов, начальной инициализацией и архитектурой без предобученных весов.

Графики подтверждают, что AdaSmooth продолжал улучшаться на протяжении всех 40 эпох, тогда как Adam остановился на 32.



# Список использованных источников
1. Mosavi A., Varkonyi-Koczy A. R.: Integration of Machine Learning and Optimization for Robot Learning. Advances in Intelligent Systems and Computing 519, 349-355 (2017).
2. Bengio, Y.: Learning deep architectures for AI. Foundations and trends in Machine Learning 2, 1-127 (2009).
3. Skillfactory : сайт - URL: https://blog.skillfactory.ru/glossary/deep-learning/ (дата обращения: 10.02.2025).
4. Neurohive : сайт - URL: https://neurohive.io/ru/vidy-nejrosetej/alexnet-svjortochnaja-nejronnaja-set-dlja-raspoznavanija-izobrazhenij/ (дата обращения: 10.02.2025).
5. Arxiv : сайт - URL: https://arxiv.org/abs/2204.00825v1 (дата обращения: 18.02.2025).
