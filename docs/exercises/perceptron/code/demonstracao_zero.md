#### Inicialização em zero e invariância à taxa

Considere duas execuções do Perceptron, ambas iniciadas com

$$
\mathbf{w}_0 = 0
\qquad\text{e}\qquad
b_0 = 0,
$$

mas com taxas de aprendizado diferentes, $\eta_1$ e $\eta_2$.

Defina

$$
c = \frac{\eta_2}{\eta_1}.
$$

Queremos mostrar que, ao longo de todo o treinamento,

$$
\mathbf{w}^{(2)} = c\,\mathbf{w}^{(1)}
$$

e

$$
b^{(2)} = c\,b^{(1)}.
$$

No início, isso é verdadeiro, pois

$$
\mathbf{w}^{(1)}_0 = \mathbf{w}^{(2)}_0 = 0
$$

e

$$
b^{(1)}_0 = b^{(2)}_0 = 0.
$$

Agora suponha que, antes de uma atualização, valha

$$
\mathbf{w}^{(2)} = c\,\mathbf{w}^{(1)}
$$

e

$$
b^{(2)} = c\,b^{(1)}.
$$

Para a primeira execução, a ativação é

$$
z_1 = \mathbf{w}^{(1)}\cdot\mathbf{x} + b^{(1)}.
$$

Para a segunda execução,

$$
z_2 = \mathbf{w}^{(2)}\cdot\mathbf{x} + b^{(2)}.
$$

Substituindo as relações anteriores,

$$
z_2
=
c\,\mathbf{w}^{(1)}\cdot\mathbf{x}
+
c\,b^{(1)}.
$$

Logo,

$$
z_2
=
c\left(
\mathbf{w}^{(1)}\cdot\mathbf{x}
+
b^{(1)}
\right),
$$

portanto,

$$
z_2 = c\,z_1.
$$

Como

$$
c = \frac{\eta_2}{\eta_1} > 0,
$$

multiplicar a ativação por $c$ não altera seu sinal. Assim,

$$
z_1 \ge 0
\iff
z_2 \ge 0.
$$

Portanto, as duas execuções fazem sempre a mesma predição:

$$
\hat{y}_1 = \hat{y}_2.
$$

Consequentemente, o erro

$$
e = y - \hat{y}
$$

também é o mesmo nas duas execuções.

A regra de atualização da primeira execução é

$$
\mathbf{w}^{(1)}_{\text{novo}}
=
\mathbf{w}^{(1)}
+
\eta_1 e\mathbf{x}.
$$

Para a segunda execução,

$$
\mathbf{w}^{(2)}_{\text{novo}}
=
\mathbf{w}^{(2)}
+
\eta_2 e\mathbf{x}.
$$

Como

$$
\mathbf{w}^{(2)} = c\,\mathbf{w}^{(1)}
$$

e

$$
\eta_2 = c\,\eta_1,
$$

temos

$$
\mathbf{w}^{(2)}_{\text{novo}}
=
c\,\mathbf{w}^{(1)}
+
c\,\eta_1 e\mathbf{x}.
$$

Colocando $c$ em evidência,

$$
\mathbf{w}^{(2)}_{\text{novo}}
=
c\left(
\mathbf{w}^{(1)}
+
\eta_1 e\mathbf{x}
\right).
$$

Assim,

$$
\boxed{
\mathbf{w}^{(2)}_{\text{novo}}
=
c\,\mathbf{w}^{(1)}_{\text{novo}}
}
$$

e, de forma análoga para o bias,

$$
\boxed{
b^{(2)}_{\text{novo}}
=
c\,b^{(1)}_{\text{novo}}
}.
$$

Portanto, essa relação permanece válida durante todo o treinamento. Ao final,

$$
\boxed{
\mathbf{w}_{\eta_2}
=
\frac{\eta_2}{\eta_1}\mathbf{w}_{\eta_1}
}
$$

e

$$
\boxed{
b_{\eta_2}
=
\frac{\eta_2}{\eta_1}b_{\eta_1}
}.
$$

A fronteira de decisão é dada por

$$
\mathbf{w}\cdot\mathbf{x} + b = 0.
$$

Multiplicar todos os parâmetros por uma constante positiva $c$ produz

$$
c\,\mathbf{w}\cdot\mathbf{x} + c\,b = 0,
$$

ou seja,

$$
c\left(
\mathbf{w}\cdot\mathbf{x} + b
\right)=0.
$$

Como $c \neq 0$,

$$
c\left(
\mathbf{w}\cdot\mathbf{x} + b
\right)=0
\iff
\mathbf{w}\cdot\mathbf{x} + b=0.
$$

Logo, as duas execuções produzem exatamente a mesma fronteira de decisão.

Além disso, como fazem as mesmas previsões em cada etapa, elas cometem erros nas mesmas amostras, realizam atualizações nos mesmos pontos e convergem no mesmo número de épocas.

Assim, se o treinamento começasse com

$$
\mathbf{w}=0,
\qquad
b=0,
$$

a taxa de aprendizado $\eta$ apenas reescalaria os valores de $\mathbf{w}$ e $b$, sem alterar a fronteira de decisão nem a sequência de previsões. É por isso que o exercício utiliza pesos iniciais aleatórios diferentes de zero.