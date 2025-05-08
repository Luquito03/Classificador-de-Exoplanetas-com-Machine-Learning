import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay



# Carregar os dados
df = pd.read_csv("cumulative.csv")

# Remover espaços e filtrar apenas "CONFIRMED" e "FALSE POSITIVE"
df['koi_disposition'] = df['koi_disposition'].str.strip()
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]

# Criar a variável de saída binária
df['exoplanet_confirmed'] = df['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)

# Selecionar apenas colunas numéricas relevantes para X
features = [
    'koi_period',         # período orbital
    'koi_duration',       # duração do trânsito
    'koi_depth',          # profundidade do trânsito
    'koi_prad',           # raio do planeta
    'koi_srad',           # raio da estrela
    'koi_steff'           # temperatura efetiva da estrela
]

df = df[features + ['exoplanet_confirmed']].dropna()  # remove só as linhas com dados ausentes nessas colunas

# Separar X e y
X = df[features]
y = df['exoplanet_confirmed']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Criando e treinando o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")



print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


sns.countplot(x=y)
plt.title("Distribuição das classes (0 = Falso Positivo, 1 = Confirmado)")
plt.xlabel("Exoplaneta Confirmado")
plt.ylabel("Número de Casos")
plt.show()

importances = model.feature_importances_
feature_names = X.columns


ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Falso Positivo", "Confirmado"], cmap="Blues")
plt.title("Matriz de Confusão")
plt.show()



