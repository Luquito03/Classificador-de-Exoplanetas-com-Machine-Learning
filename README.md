
# 🔭 Classificador de Exoplanetas com Machine Learning

Este projeto usa dados da missão Kepler (NASA) para treinar um modelo de aprendizado de máquina capaz de identificar se um objeto observado é um **exoplaneta confirmado** ou um **falso positivo**.

## 🚀 Sobre o Projeto

A missão Kepler observou mais de 150.000 estrelas para detectar possíveis exoplanetas por meio do método de trânsito. Com esses dados, treinamos um modelo de **Random Forest** para classificar automaticamente objetos como:

- **CONFIRMED** (Exoplaneta confirmado)
- **FALSE POSITIVE** (Erro ou detecção inválida)

## 🧠 Técnicas e Ferramentas Usadas

- Python 3
- Pandas e NumPy
- Scikit-learn (Random Forest)
- Matplotlib e Seaborn (visualizações)
- Kaggle API (dados)

## 📊 Dataset

Usamos o conjunto de dados **Kepler Exoplanet Search Results**, disponível em:  
🔗 [Kaggle – NASA Kepler Exoplanet Data](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)

O arquivo principal utilizado é `cumulative.csv`.

## ⚙️ Como Rodar o Projeto

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/nome-do-repo.git
cd nome-do-repo
```

2. Instale as dependências:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

4. Execute o script principal:

```bash
python main.py
```

## 📈 Resultados

O modelo obteve uma acurácia de aproximadamente **89%**, com bom desempenho na distinção entre candidatos reais e falsos positivos.  
Incluímos também visualizações de:

- Distribuição de classes
- Importância das variáveis
- Matriz de confusão

## 💡 Melhorias Futuras

- Testar outros algoritmos (XGBoost, SVM, redes neurais)
- Interface web com Streamlit
- Detecção multiclasses (incluindo candidatos ainda não confirmados)

## 👨‍🚀 Autor

- Nome: **Lucas**
- GitHub: [@Luquito03](https://github.com/Luquito03/)
- LinkedIn: [Lucas Bernadino](https://www.linkedin.com/in/lucas-bernadino-a56b9b240/)


