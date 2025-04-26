import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat

import json

df = pd.read_json("enem_2023.json")

df.shape

df.head(10)

df.info()

df.isna().sum()

plt.figure(figsize=(10, 6))
df.isnull().sum().plot(kind='bar')
plt.title('Contagem de Valores Nulos por Coluna')
plt.xlabel('Colunas')
plt.ylabel('Contagem de Nulos')
plt.show()

df_numerico = df.select_dtypes(include='number')
coluna_maior_amplitude = (df_numerico.max() - df_numerico.min()).idxmax()
coluna_maior_amplitude

df_sem_nulos = df.dropna()

df_sem_nulos.isna().sum()

df_sem_nulos.describe()

df_numerico = df.select_dtypes(include='number')
df_sem_nulos = df_numerico.agg(['mean', 'median'])
df_sem_nulos


df_sem_nulos['nota_ponderada'] = (df['Linguagens'] * 2) + \
                       (df['Ciências humanas'] * 4) + \
                       (df['Ciências da natureza'] * 2) + \
                       (df['Matemática'] * 1) + \
                       (df['Redação'] * 1)

df_ordenado = df.sort_values('nota_ponderada', ascending=False)

top_students = df_ordenado.head(500)


top_students

4. Se todos esses estudantes aplicassem para ciência da computação e
existem apenas 40 vagas, qual seria a variância e média da nota dos
estudantes que entraram no curso de ciência da computação?



top_40 = df_ordenado.head(40)

media_top_40 = top_40['nota_ponderada'].mean()
variancia_top_40 = top_40['nota_ponderada'].var()
print(f"Média: {media_top_40}")
print(f"Variancia: {variancia_top_40}")

5. Qual o valor do teto do terceiro quartil para as disciplinas de matemática
e linguagens?

q3_matematica = df_sem_nulos['Matemática'].quantile(0.75)
q3_linguagens = df_sem_nulos['Linguagens'].quantile(0.75)

print(f"Teto Q3 Matemática: {(q3_matematica)}")
print(f"Teto Q3 Linguagens: {(q3_linguagens)}")


6. Faça o histograma de Redação e Linguagens, de 20 em 20 pontos.
Podemos dizer que são histogramas simétricos, justifique e classifique se
não assimétricas?



# Histograma de Linguagens
plt.figure(figsize=(4, 4))
sns.histplot(df['Linguagens'].dropna(), bins=range(0, 1001, 20), kde=True, color='gray')
plt.title('Histograma de Notas de Linguagens')
plt.xlabel('Nota de Linguagens')
plt.ylabel('Frequência')
plt.show()

# Histograma de Redação
plt.figure(figsize=(4, 4))
sns.histplot(df['Redação'].dropna(), bins=range(0, 1001, 20), kde=True, color='gray')
plt.title('Histograma de Notas de Redação')
plt.xlabel('Nota de Redação')
plt.ylabel('Frequência')
plt.show()



assimetria_linguagens = df['Linguagens'].skew()
assimetria_redacao = df['Redação'].skew()

print(f"Assimetria de Linguagens: {assimetria_linguagens:.2f} - simétrico")
print(f"Assimetria de Redação: {assimetria_redacao:.2f} - simétrico")

7. Agora coloque um range fixo de 0 até 1000, você ainda tem a mesma
opinião quanto a simetria?

# Histograma de Linguagens com range fixo de 0 a 1000
plt.figure(figsize=(8, 6))
sns.histplot(df['Linguagens'].dropna(), bins=range(0, 1000), kde=True, color='skyblue')
plt.title('Histograma de Notas de Linguagens (0-1000)')
plt.xlabel('Nota de Linguagens')
plt.ylabel('Frequência')
plt.xlim(0, 1000)
plt.show()

# Histograma de Redação com range fixo de 0 a 1000
plt.figure(figsize=(8, 6))
sns.histplot(df['Redação'].dropna(), bins=range(0, 1000), kde=True, color='salmon')
plt.title('Histograma de Notas de Redação (0-1000)')
plt.xlabel('Nota de Redação')
plt.ylabel('Frequência')
plt.xlim(0, 1000)
plt.show()


8. Faça um boxplot para as notas de Ciências da Natureza e Redação,
analisando os quartis e identificando possíveis outliers. Utilize o método
IQR (Intervalo Interquartílico) para essa análise.

plt.figure(figsize=(9, 6))

plt.subplot(1, 2, 1)
sns.boxplot(y=df['Ciências da natureza'], color='lightgreen')
plt.title('Boxplot de Ciências da Natureza')
plt.ylabel('Nota')



plt.subplot(1, 2, 2)
sns.boxplot(y=df['Redação'], color='skyblue')
plt.title('Boxplot de Redação')
plt.ylabel('Nota')

plt.tight_layout()
plt.show()


9. Remova todos os outliers e verifique se eles são passíveis de alterar a
média nacional significativamente? (considere significativamente um valor
acima de 5%)

plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['nota_ponderada'], label='Estudantes')

plt.axhline(y=media_nacional, color='green', linestyle='--', label=f'Média Nacional ({media_nacional:.2f})')

plt.xlabel('Estudantes Index')
plt.ylabel('Nota Ponderada')
plt.title('Outliers - Média Nacional')
plt.legend()
plt.show()




def count_outliers_zscore(df, threshold=3):

  outlier_counts = {}
  for col in df.select_dtypes(include=['number']).columns:
    z_scores = np.abs(stat.zscore(df[col].dropna()))
    outlier_counts[col] = (z_scores > threshold).sum()
  return outlier_counts

outliers = count_outliers_zscore(df)
outliers
