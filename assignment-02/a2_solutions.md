### 1 Written: Understanding word2vec (23 points) 

1. $-\sum_{w\in Vocal}y_w\log(\hat{y}_w)=-(y_1\log(\hat{y}_1)+y_2\log(\hat{y}_2+...+y_k\log(\hat{y}_k)+...+y_m\log(\hat{y}_m))=-y_k\log(\hat{y}_k)=-\log(\hat{y}_o) \qquad \{y_i=0 |i \neq k\} ,y_k=1$

2. * $$
     \begin{align} \frac{\partial J_{naive-softmax}}{\partial v_c}&=-\frac{\sum_{w \in Vocab}\exp(u_w^Tv_c)}{\exp(u_o^Tv_c)}*\frac{\exp(u_o^Tv_c)u_o\sum_{w \in Vocab}\exp(u_w^Tv_c)-exp(u_o^Tv_c)\sum_{x \in Vocab}exp(u_x^Tv_c)u_x}{(\sum_{w \in Vocab}\exp(u_w^Tv_c))^2}\\ &=-(u_o-\frac{\sum_{x \in Vocab}exp(u_x^Tv_c)u_x}{\sum_{w \in Vocab}\exp(u_w^Tv_c)})\\ &=-(u_o-\sum_{x \in Vocab}\frac{\exp(u_x^Tv_c)}{\sum_{w \in Vocab}\exp(u_w^Tv_c)}u_x)\\ &=-(u_o-\sum_{x=1}^Vp(x|c)u_x)\\ &=-(Uy-U\hat{y})\\ &=U(\hat{y}-y)\\ &U \in R^{n*m},\qquad y \in R^{m*1} \end{align}
     $$

3. * when $w=o$,
     $$
     \begin{align} 
     \frac{\partial J_{naive-softmax}}{\partial u_w}&=\frac{1}{\sum_{w \in Vocab}\exp(u_w^Tv_c)}\sum_{x \in Vocab}\exp(u_x^Tv_c)v_c^T-v_c^T\\
     &=\sum_{x \in Vocab}\frac{\exp(u_x^Tv_c)}{\sum_{w \in Vocab}\exp(u_w^Tv_c)}v_c^T-v_c^T\\
     &=p(x|c)v_c^T-v_c^T\\
     &=(\hat{y}-1)v_c^T
     \end{align}
     $$

   * when $w\neq o$,
     $$
     \begin{align}  
     \frac{\partial J_{naive-softmax}}{\partial u_w}&=\frac{1}{\sum_{w \in Vocab}\exp(u_w^Tv_c)}\sum_{x \in Vocab}\exp(u_x^Tv_c)v_c^T\\
     &=\sum_{x \in Vocab}\frac{\exp(u_x^Tv_c)}{\sum_{w \in Vocab}\exp(u_w^Tv_c)}v_c^T\\
     &=p(x|c)v_c^T\\ 
     &=\hat{y}v_c^T
     \end{align}
     $$

   * merge,
     $$
     \begin{align}   
     \frac{\partial J_{naive-softmax}}{\partial u_w}&=(\hat{y}-y)v_c^T
     \end{align}
     $$

   * 解释：当 $w \neq o$ 时，真实向量 *y* 中对应的值为 0，只有当 *w* 属于 *outside word* 的时候，*y* 中对应的值才等于 1，故最后可以合并

4. * $$
     \begin{align}
     \frac{\partial\sigma}{\partial x}&=\frac{e^x(e^x+1)-e^xe^x}{(e^x+1)^2}=\sigma(x)-\sigma^2(x)=\sigma(x)(1-\sigma(x))
     \end{align}
     $$

5. * to $v_c$:
     $$
     \begin{align}\frac{\partial J_{neg-sample}}{\partial v_c}&=-\frac{\sigma(u_o^Tv_c)(1-\sigma(u_o^Tv_c))u_o}{\sigma(u_o^Tv_c)}-\sum_{k=1}^K\frac{\sigma(-u_k^Tv_c)(1-\sigma(-u_k^Tv_c))}{\sigma(-u_k^Tv_c)}(-u_k)\\&=\sum_{k=1}^K(1-\sigma(-u_k^Tv_c))u_k-(1-\sigma(u_o^Tv_c))u_o\\&=\sum_{k=1}^K\sigma(u_k^Tv_c)u_k-\sigma(-u_o^Tv_c)u_o\end{align}
     $$

   * to $u_o$:
     $$
     \begin{align}
     \frac{\partial J_{neg-sample}}{\partial u_o}&=-\frac{\sigma(u_o^Tv_c)(1-\sigma(u_o^Tv_c))v_c^T}{\sigma(u_o^Tv_c)}\\
     &=-\sigma(-u_o^Tv_c)v_c^T
     \end{align}
     $$

   * to $u_k$:
     $$
     \begin{align}
     \frac{\partial J_{neg-sample}}{\partial u_k}&=-\frac{\sigma(-u_k^Tv_c)(1-\sigma(-u_k^Tv_c))(-v_c^T)}{\sigma(-u_k^Tv_c)}\\
     &=\sigma(u_k^Tv_c)v_c^T
     \end{align}
     $$

   * 解释：和 *naive-softmax loss* 比起来，不需要对整个词库计算向量点积并求和，效率更高

6. * to *U*:
     $$
     \begin{align}
     \frac{\partial J(_{skip-gram}(v_c,w_{t-m},...w_{t+m}))}{\partial U}&=\sum_{-m \le j \le m \\ \quad j \neq 0}\frac{\partial J(v_c,w_{t+j},U)}{\partial U}
     \end{align}
     $$

   * to $v_c$:
     $$
     \begin{align}
     \frac{\partial J(_{skip-gram}(v_c,w_{t-m},...w_{t+m}))}{\partial v_c}&=\sum_{-m \le j \le m \\ \quad j \neq 0}\frac{\partial J(v_c,w_{t+j},U)}{\partial v_c}
     \end{align}
     $$

   * to $v_w$:
     $$
     \begin{align} 
     \frac{\partial J(_{skip-gram}(v_c,w_{t-m},...w_{t+m}))}{\partial v_w}&=o \qquad (w \neq c)
     \end{align}
     $$
     