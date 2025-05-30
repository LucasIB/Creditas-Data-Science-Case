# Creditas-Data-Science-Case
Creditas-Case-Challenge

# Projeto: Classificação de Clientes para Análise de Crédito
Este projeto visa construir um modelo de machine learning para priorizar clientes que têm maior probabilidade de serem enviados para análise de crédito em uma empresa de empréstimos com garantia de veículo.
## Contexto do Problema
Após serem pré-aprovados automaticamente, os clientes preenchem um formulário adicional e podem ser enviados à análise de crédito por um consultor. Nosso objetivo é construir um modelo preditivo para classificar e priorizar os clientes mais propensos a serem enviados para essa análise.
## Execução
### Requisitos
Python >= 3.8

Instale os pacotes com:

`pip install -r requirements.txt`
### Como rodar o projeto
Certifique-se de que o arquivo dataset.csv está na mesma pasta.

Rode o notebook modelo_classificacao_credito.ipynb ou o script .py com:

`python Creditas - Credit Analysis.py`
### Principais Etapas
> - Análise exploratória (EDA)
> 
> - Tratamento de dados (valores ausentes, encoding)
> - Balanceamento com SMOTE
> - Modelos: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM
> - Ajuste de threshold personalizado (ex: 0.35)
>
> - Avaliação com métricas e curvas (ROC, Precision-Recall)
### Métricas de Avaliação
> - Accuracy: acertos gerais
>
> - Recall (classe 1): quantos bons clientes foram identificados corretamente
>
> - Precision (classe 1): dos classificados como bons, quantos realmente eram
>
> - F1-Score: equilíbrio entre precision e recall
>
> - AUC-ROC: capacidade geral do modelo distinguir entre as classes
### Interpretação com SHAP
> - shap.plots.bar: importaância global das variáveis
>
> - shap.plots.beeswarm: direção e impacto dos valores
>
> - shap.plots.waterfall: explicação individual por cliente
### Monitoramento recomendado em produção
Acompanhar recall, precision e AUC-ROC da classe 1
>
> - Verificar drift em features importantes
>
> - Atualizar modelo trimestralmente com novos dados
>
> - Criar painel com alertas de degradação
### Melhorias futuras
> - Usar CatBoost para lidar melhor com variáveis categóricas
>
> - Incluir variáveis de comportamento (tempo de preenchimento, etc)
>
> - Enriquecimento com fontes externas de risco de crédito
>
> - Criar pipeline com MLFlow/DVC para controle de versão
### Requisitos de Dependência
> - Salve o seguinte como requirements.txt:
~~~
scikit-learn
pandas
numpy
xgboost
lightgbm
shap
imblearn
matplotlib
~~~
