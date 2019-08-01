### 1 Neural Machine Translation with RNNs

* (g) `enc_masks` 的 作用在于将 `pad tokens` 在 `attention` 的计算中的得分置为负无穷，这样在进行 `softmax` 得到的概率分布中 `pad tokens` 的概率将会为 0，不会得到任何关注，这正是我们想要的
* (j) 
  * `dot product` : 
    * advantages: 无需多余的参数，计算效率高，节省内存
    * disadvantages: 过于简单捕捉不了复杂的模式
  * `additive`:
    * advantages: 能够学习到比较复杂的模式
    * disadvantages: 多了超参数需要调节，计算量加大，耗内存
  * `multiplicative` : 居中 :laughing:

### 2 Analyzing NMT Systems

1. i.specific linguistic construct , ii.specific linguistic construct , iii.specific model limitations , iv.specific model limitations, v.specific model limitations, vi.specific model limitations

2. TODO

3. about BLEU Score

   1. 计算 BLEU：![参考翻译和 `NMT` 产生的结果](https://raw.githubusercontent.com/Brycexxx/Images/master/20190801174140.jpg)

      1. for *c1*, $p_1=\frac{3}{5}=0.6$, $p_2=\frac{2}{4}=0.5$, $c=5$,  $r^*=6$, $BP=\exp(1-\frac{6}{5})=0.81873$, $BLEU=0.81873*\exp(\sum_{n=1}^2\lambda_n\log(p_n))=0.4484$

      2. for *c2*, $p_1=\frac{4}{5}=0.8$, $p_2=\frac{2}{4}=0.5$, $c=5$, $r^*=4$, $BP=1$, $BLEU=1 * \exp(\sum_{n=1}^2\lambda_n\log(p_n))=0.6325$

      3. 根据 *BLEU*， *c2* 更好；我也认同

      4. $$
         p_n=\frac{\sum_{ngram \in c}\min(max_{i=1,...k}Count_{r_i}(ngram), Count_c(ngram))}{\sum_{ngram \in c}Count_c(ngram)}
         $$

      5. 对以上 $p_n$ 的计算讲一下自己的理解。比如当 $n=2$ 的时候，首先看分母，计算候选翻译中总的 `2-gram` 数量，比如 *c2* 共有 4 个；再看分子，首先 *c2* 中第一个 `2-gram` 是 `love can` (`<start> love` 不算的话)，那么分别在两个参考翻译中查找是否含有，含有多少个， *r1* 中有一个，*r2* 中没有，取 $\max(0, 1)$ ，然后再和候选翻译中的个数一起取最小值；依此类推，可计算出 $p_n$  

   2. only *r1*

      1. for *c1* ， $p_1=\frac{3}{5}$, $p_2=\frac{2}{4}$, $c=5$, $r^*=6$, $BP=0.81873$, $BLEU=0.81873*\exp(\sum_{n=1}^2\lambda_n\log(p_n))=0.4484$
      2. for *c2*, $p_1=\frac{2}{5}$, $p_2=\frac{1}{4}$, $c=5$, $r^*=6$, $BP=0.81873$,  $BLEU=0.81873*\exp(\sum_{n=1}^2\lambda_n\log(p_n))=0.2589$
      3. 第一个 `NMT` 的得分更高，我不认为它的翻译更好

   3. 因为一个句子在同一种语言中可能有多种不同的表达方式，只有一个参考翻译时，难免会使得具有相同意思另一种说法的翻译结果得到不好的分数

   4. Advantages:

      - It's convenient and fast, a good substitue of human evaluation.
      - In the corpus level, the score of BLEU is close to human's score.

      Disadvantages:

      - It only works well on the corpus level because any zeros in precision scores will zero the entire BLEU score.
      - BLEU score as presented suffers for only comparing a candidate translation against a single reference, which is surely a noisy representation of the relevant n-grams that need to be matched.

> 参考：https://github.com/ZacBi/CS224n-2019-solutions/blob/master/Assignments/written%20part/a4_solution.md