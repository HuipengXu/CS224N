### 1 Character-based convolutional encoder for NMT 

1. 应该是因为字符集合比词汇集合总数少很多，而我们在表示一个字符时用到的维度一般都会比集合总数要小，所以 `character-level embedding` 要比 `word embedding` 的维度小

2. in terms of $e_{char}$, $k$, $e_{word}$, $V_{word}$ (the size of the word-vocabulary in the lookup embedding model) and $V_{char}$ (the size of the character-vocabulary in the character-based embedding model).Given that in our code, k = 5, $V_{word} \approx 50000$ and $V_{char} = 96$, state which model has more parameters, and by what factor (e.g. twice as many? a thousand times as many?). 

   * `character-based embedding`: $V_{char}*e_{char}+e_{word}*e_{char}*k+e_{word}+2*e_{word}^2+2*e_{word} \approx 200640$
   * `word embedding`: $V_{word}*e_{word} \approx 12800000$
   * 约等于 1e2

3. `RNN` 仅使用最后一个 `hidden state` 作为词向量，能够捕获的信息有限，而 `convnet` 可以通过改变 `filter` 的数量自由地控制想捕获的模式或者说信息

4.  (1) `max pooling`

   * advantages: 获得了数据中最强的模式/信号
   * disadvantages： 丢弃了数据中的大部分稍弱的信息

   (2) `avg pooling`

   * advantages: 保存了大部分信息
   * disadvantages: 如果信息的强弱两极分化太严重，那么会将最有用的信息大大稀释，得到的信息将变得无用







> 参考：https://github.com/ZacBi/CS224n-2019-solutions/blob/master/Assignments/written%20part/a5_solution.md

