#-----------------------------------------------------------------------------------------------#
#                                  ___ ___ ___ ___ ___ _____ _   ___                            #
#                                 / __| _ \ __|   \_ _|_   _/_\ / __|                           #
#                                | (__|   / _|| |) | |  | |/ _ \\__ \                           #
#                                 \___|_|_\___|___/___| |_/_/ \_\___/                           #
#                                                                                               #
#                                     @uthor: Lucas Igor Balthazar                              #
#                                      Cientista de Dados, Físico                               #
#                                  mailto: lucasibalthazar@gmail.com                            #
#                                                2025                                           #
# Este notebook é um estudo de caso para a empresa [Creditas](https://www.creditas.com/)        #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#

# Imports e Setup
import os                                           #all built-in operating system dependent modules
import sys                                          #provides access to some variables used or maintained by the interpreter
import pandas as pd                                 #data processing csv, excel, files
import numpy as np                                  #scientific computing
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns                               #data visualization library based on matplotlib
import matplotlib.pyplot as plt                     #graphical plots analysis
#Machine Learning Libs
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc

#Funções auxiliares úteis

def print_dtypes(df: pd.DataFrame)-> pd.DataFrame:
    '''Printing columns dtypes from database. 
    args.: df = pandas DataFrame'''
    
    return df.dtypes

def cardinality_check(df: pd.DataFrame)-> pd.DataFrame:
    '''Checking cardinality / distinct count for all columns dataframe. 
    args.: df = pandas DataFrame'''
    
    return df.apply(pd.Series.nunique)

def checking_NaN(df: pd.DataFrame)-> pd.DataFrame:
    '''Checking empty values'''
    empt = df.isnull().values.any()
    if empt:
        val = df.isnull().sum()
    else:
        val = 0
    return print(f'Dados faltantes: {empt}\nValores:\n{val}')

def proportion_missing(df):
    '''Missing data proportion (%)'''
    perc ={}
    for item in df.columns:
        if df[item].isnull().values.any():
            perc[item] = round((df[item].isnull().sum()/df.shape[0])*100,3)
    return perc

def categoric_array(df: pd.DataFrame)-> pd.DataFrame:
    '''Getting the non-numeric (categorical) values in df's'''
    array_not_num = [item for item in df.columns if df[item].dtypes == 'object']
    return array_not_num

def numeric_array(df: pd.DataFrame)-> pd.DataFrame:
    '''Getting the numeric (numeric) values in df's'''
    array_num = [item for item in df.columns if df[item].dtypes != 'object']
    return array_num

def verify_category_uniform(df):
    '''To analyse the uniform type in the categorical variables'''
    array_not_num = [item for item in df.columns if df[item].dtypes == 'object']
    cat_type = {}
    for item in array_not_num: cat_type[item] = df[item].unique()
    return cat_type

def plot_dataframe_structure(df: pd.DataFrame, df_name)-> pd.DataFrame:
    '''To plot DataFrame Structure Proportions
    args.: df = pandas DataFrame'''
    plt.figure()
    df.dtypes.value_counts().plot.pie(ylabel='')
    plt.title('Data types '+str(df_name))
    plt.show()

def summary_data(df):
    print('#'*30+' SHAPE '+'#'*30)
    print(df.shape)
    print('#'*30+' HEAD '+'#'*30)
    print(df.head(3))
    print('#'*30+' TAIL '+'#'*30)
    print(df.tail(3))
    print('#'*30+' DTYPES '+'#'*30)
    print(df.dtypes)
    print('#'*30+' DESCRIBE '+'#'*30)
    print(df.describe())
    print('#'*30+' INFO '+'#'*30)
    print(df.info(3))
    print('#'*30+' QUANTILES '+'#'*30)
    print(df.quantile([0.25,0.5,0.75]))

# Importando dados a partir do caminho do arquivo
def filepath(file_name:'csv') -> 'str':
    '''Add the file path'''
    if not (file_name.find('.csv') != -1): file_name = file_name+'.csv'
    script_dir = os.getcwd()
    rel_path = file_name       #File name
    abs_file_path = os.path.join(script_dir, rel_path)
    
    return abs_file_path

#Criando os DataFrame dataset:
#df = pd.read_csv(filepath('dataset'))
df = pd.read_csv(r'C:\Users\lucas\Documents\Creditas Case\desafio-ds\desafio-ds\Entrega-case-Lucas_Balthazar\dataset.csv')

#Criando os DataFrame descrição dos Dados
#df_description = pd.read_csv(filepath('description'))
df_description = pd.read_csv(r'C:\Users\lucas\Documents\Creditas Case\desafio-ds\desafio-ds\Entrega-case-Lucas_Balthazar\description.csv')

#Checando casos de NaN
print('[INFO] Chegando os NaN do df bruto:')
checking_NaN(df)

#Proporção de valores ausentes em relção a base
proportion_missing(df)
null_prop = pd.DataFrame(proportion_missing(df), index=np.arange(1)).T
null_prop = null_prop.rename(columns={0:'%'})
print('[INFO] Proporção de valores ausentes de df: ')
null_prop

#Alterando tipo da variável id
df['id'] = df['id'].astype(object)

## FILTRANDO A BASE DE DADOS PARA CLIENTES PRÉ-APORVADOS
df_pre_approved = df[df['pre_approved'] == 1].copy()

# Remover colunas com muitos valores ausentes ou irrelevantes para o modelo
cols_to_drop = ['loan_term', 'marital_status', 'utm_term']
df_pre_approved = df_pre_approved.drop(columns=cols_to_drop)

#Checando os valores filtando apenas os clientes pré-aprovados
print(f'Tamanho do DataFrame apenas de clientes pré-aprovados: {df_pre_approved.shape}')

# Verificando a distribuição da variável-alvo
var_resposta = df_pre_approved["sent_to_analysis"].value_counts(normalize=True)
print("Distribuição da variável-alvo (em %):\n", var_resposta * 100)

#Separando as colunas numéricas
num_cols = numeric_array(df_pre_approved)
print('[INFO] Variáveis numéricas separadas')

#Separando as colunas categóricas
cat_cols = categoric_array(df_pre_approved)
print('[INFO] Variáveis categóricas separadas')

## TRATAMENTO DO VALORES AUSENTES DAS VARIÁVEIS DE INDICADORES `FLAGS`
possible_flags = []
for col in num_cols:
    unique_vals = df_pre_approved[col].dropna().unique()
    if set(unique_vals).issubset({0, 1}):
        possible_flags.append(col)

print("Colunas identificadas como flags binárias:", possible_flags)
# Flags a preencher com 0
flags_zero = [
    'dishonored_checks',
    'expired_debts',
    'banking_debts',
    'commercial_debts',
    'protests',
    'informed_restriction'
]
for col in flags_zero:
    if df_pre_approved[col].isnull().sum() > 0:
        df_pre_approved[col].fillna(0, inplace=True)
        print(f"'{col}' preenchido com 0")

# nova flag: se houve consulta automatizada de restrição
df_pre_approved['restriction_checked'] = df_pre_approved['verified_restriction'].notna().astype(int)

#Identificando as colunas numérias contínuas (exceto flags):
lst = ['sent_to_analysis', 'restriction_checked']
continuous_cols = [col for col in num_cols if col not in possible_flags and col not in lst]

# Inserindo contínuas com mediana
for col in continuous_cols:
    if df_pre_approved[col].isnull().sum() > 0:
        mediana = df_pre_approved[col].median()
        df_pre_approved[col].fillna(mediana, inplace=True)
        print(f"Contínua: '{col}' preenchida com mediana: {mediana}")

## EDA - EXPLORATORY DATA ANALYSIS
variables = ['age', 'monthly_income', 'loan_amount', 'collateral_value', 'collateral_debt', 'monthly_payment']

fig, axes = plt.subplots(nrows=len(variables), ncols=2, figsize=(14, len(variables)*3))
fig.suptitle("Distribuições com e sem Outliers (Percentil 95)", fontsize=16, y=1.02)

for i, col in enumerate(variables):
    # Histograma completo
    sns.histplot(data=df_pre_approved, x=col, hue="sent_to_analysis", bins=50, kde=True, ax=axes[i, 0], color='steelblue')
    axes[i, 0].set_title(f"{col} - Distribuição Completa")
    axes[i, 0].set_xlabel("")
    axes[i, 0].set_ylabel("Frequência")

    # Histograma sem outliers
    limite = df_pre_approved[col].quantile(0.95)
    df_sem_outliers = df_pre_approved[df_pre_approved[col] < limite]
    sns.histplot(data=df_sem_outliers, x=col, hue="sent_to_analysis", bins=50, kde=True, ax=axes[i, 1], color='seagreen')
    axes[i, 1].set_title(f"{col} - Sem Outliers (até p95)")
    axes[i, 1].set_xlabel("")
    axes[i, 1].set_ylabel("Frequência")

plt.tight_layout()

#Gráficode barras para flags
flags = flags_zero + ['restriction_checked']

# Calcular proporção de envio para análise por flag
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, flag in enumerate(flags):
    prop = df_pre_approved.groupby(flag)['sent_to_analysis'].mean()
    sns.barplot(x=prop.index, y=prop.values, ax=axes[i])
    axes[i].set_title(f"Taxa de Envio por {flag}")
    axes[i].set_ylabel("Prob. de Envio para Análise")
    axes[i].set_xlabel(flag)

plt.tight_layout()
plt.show()

## PREPARAÇÃO PARA MODELAGEM
# Lista de variáveis com caudas longas (já detectadas)
log_features = ["monthly_income", "collateral_value", "loan_amount", "collateral_debt", "monthly_payment"]

# Aplicar transformação log1p (log(1+x)) — segura para zero
for col in log_features:
    df_pre_approved[f"{col}_log"] = np.log1p(df_pre_approved[col])

#Encoding de variáveis categóricas
#Selecionando as variáveis categóricas com baixa cardinalidade
cat_cols_model = ["state", "gender", "channel", "informed_purpose"]

# Preencher valores ausentes com "Desconhecido"
for col in cat_cols_model:
    df_pre_approved[col] = df_pre_approved[col].fillna("Desconhecido")

# Aplicar one-hot encoding
df_encoded = pd.get_dummies(df_pre_approved, columns=cat_cols_model, drop_first=True)

#Selecionar apenas variáveis úteis
# Lista das colunas finais (evita variáveis brutas e duplicadas)
feature_cols = [
    "age"
] + [f"{col}_log" for col in log_features] + flags + list(df_encoded.columns[df_encoded.columns.str.startswith(tuple(cat_cols_model))])

X = df_encoded[feature_cols]
y = df_encoded["sent_to_analysis"]

#Dividir a Base de Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

#Balaceamento de classe com SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

## TREINAMENTO DOS MODELOS MACHINE LEARNING
print('[INFO] Treinamento dos Modelos de Machine Learning')

# Modelo 1: Regressão Logística
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_resampled, y_resampled)
y_pred_log = logreg.predict(X_test)
y_prob_log = logreg.predict_proba(X_test)[:,1]
print('[INFO] Regressão Logística treinada')

# Modelo 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_resampled, y_resampled)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]
print('[INFO] Random Forest treinada')

# Modelo 3: XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_resampled, y_resampled)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]
print('[INFO] XGBoost treinada')

# Modelo 4: SVM
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_resampled[:5000], y_resampled[:5000])
y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:,1]
print('[INFO] SVM treinada')

# Avaliar todos juntos
modelos = {
    "Logistic Regression": (y_pred_log, y_prob_log),
    "Random Forest": (y_pred_rf, y_prob_rf),
    "XGBoost": (y_pred_xgb, y_prob_xgb),
    "SVM": (y_pred_svm, y_prob_svm)
}

resultados = []

for nome, (y_pred, y_prob) in modelos.items():
    resultados.append({
        "Modelo": nome,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob)
    })

df_resultados = pd.DataFrame(resultados).sort_values(by="AUC-ROC", ascending=False)
print(df_resultados)

# Função de avaliação
def avaliar_modelo(nome, y_true, y_pred, y_prob):
    print(f"\nModelo: {nome}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print("AUC-ROC:", roc_auc_score(y_true, y_prob))

# Avaliar ambos os modelos
avaliar_modelo("Logistic Regression", y_test, y_pred_log, y_prob_log)
avaliar_modelo("Random Forest", y_test, y_pred_rf, y_prob_rf)
avaliar_modelo("XGBoost", y_test, y_pred_xgb, y_prob_xgb)
avaliar_modelo("SVM", y_test,y_pred_svm, y_prob_svm)

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Converter para porcentagem
cm_percent = cm / cm.sum() * 100

# Criar labels com o valor em %
labels_cm = np.array([["{:.1f}%".format(v) for v in row] for row in cm_percent])

# Rótulos das classes
labels_x = ["Não Enviado", "Enviado"]
labels_y = ["Não Enviado", "Enviado"]

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm_percent, annot=labels_cm, fmt="", cmap="Blues", xticklabels=labels_x, yticklabels=labels_y)
plt.title("Matriz de Confusão - Valores em %")
plt.xlabel("Predição do Modelo")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()

# Previsões de probabilidade
y_scores = rf.predict_proba(X_test)[:, 1]

# Obter curvas
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

# Plotar
plt.figure(figsize=(10,6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.plot(thresholds, f1_scores[:-1], label='F1 Score', linestyle='--')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall e F1-Score vs Threshold")
plt.legend()
plt.grid(True)
plt.show()

#Alterando o Threshold = 0.35
threshold_custom = 0.35

#Gerar predições binárias com o novo threshold
y_pred_custom = (y_scores >= threshold_custom).astype(int)

# Avaliar
print("Avaliação com threshold =", threshold_custom)
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))
print("AUC-ROC:", roc_auc_score(y_test, y_scores))

#Testando vário thresholds
for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
    print(f"\n--- Threshold: {t} ---")
    y_pred = (y_scores >= t).astype(int)
    print(classification_report(y_test, y_pred, digits=3))

#Importância das Variáveis:
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(15).plot(kind='barh', figsize=(10, 6), title="Top 15 Features")
plt.show()

#Curva ROC
y_prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (Recall)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.grid()
plt.show()

## GERAR RANKING DE CLIENTES POR PORTABILIDADE
# Criar ranking no conjunto de teste
ranking_df = X_test.copy()
ranking_df["prob_sent_to_analysis"] = y_prob_rf
ranking_df["real_class"] = y_test.values

# Ordenar pelo score de maior para menor
ranking_df = ranking_df.sort_values(by="prob_sent_to_analysis", ascending=False)

# Visualizar top clientes
ranking_df.to_excel('ranking_clientes.xlsx')

print(ranking_df.head(10))

print('[INFO] Ranking de Clientes gerado com sucesso!')
print('[INFO] Pipeline de Análise de Crédito finalizado com sucesso!')