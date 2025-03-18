""" Resumo:
O KNeighborsClassifier é um algoritmo de aprendizado de máquina supervisionado usado para classificação.
Ele faz parte das ferramentas mais populares para ciência de dados e aprendizado de máquina.
A ideia básica do KNN (K-Nearest Neighbors), é classificar novos pontos de dados com base na maioria das
classes presentes entre seus "vizinhos" mais próximos.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd # Opcional, mas útil para manipular os dados

# Exemplo de dados (substitua pelos seus dados reais)
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [2, 4, 1, 3, 5, 7, 6, 8, 10, 9],
    'label': [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  # 0 e 1 são as classes
}
df = pd.DataFrame(data)

# Separando os dados em features (X) e rótulos (y)
X = df[['feature1', 'feature2']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o classificador KNN com 3 vizinhos (você pode ajustar este valor)
knn = KNeighborsClassifier(n_neighbors=3)

# Treinando o modelo
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Acurácia: Porcentagem de previsões corretas
print("Acurácia:", accuracy_score(y_test, y_pred))

# Relatório de Classificação: Precisão, recall, F1-score para cada classe
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de Confusão: Mostra os verdadeiros positivos, falsos positivos, etc.
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))

# Exemplo de ajuste de parâmetros
for k in [1, 3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # Ou 'manhattan', 'cosine', etc.
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"Acurácia para k={k}:", accuracy_score(y_test, y_pred))