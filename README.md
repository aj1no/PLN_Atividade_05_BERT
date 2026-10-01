# Classificação Binária de Sentimento no IMDB com BERT - PLN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aj1no/PLN_Atividade_05_BERT/blob/main/05_BERT.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Este repositório contém a resolução da **Atividade 05: Classificação de Sentimento com BERT (Bidirectional Encoder Representations from Transformers)**, desenvolvida para a disciplina de **Processamento de Linguagem Natural (PLN)**.

---

## Informações Acadêmicas

* **Curso:** Ciência de Dados (5º Semestre)
* **Disciplina:** Processamento de Linguagem Natural (PLN)
* **Professor:** Leonardo de Lellis Rossi
* **Aluno:** Rodolfo Vinicius Cima Takemoto

---

## Estrutura do Notebook

O notebook [`05_BERT.ipynb`](05_BERT.ipynb) está dividido nas seguintes seções principais:

### 1. Configuração e Reprodutibilidade
* **Fixação de Sementes (`seed=123`):** Garantia de consistência estocástica nos geradores de números pseudo-aleatórios do Python (`random`), NumPy (`np.random`) e PyTorch (`torch.manual_seed` / `torch.cuda.manual_seed_all`).
* **Seleção de Dispositivo:** Detecção automática de acelerador gráfico CUDA/GPU com fallback para execução em CPU.

### 2. Preparação do Dataset IMDB (Large Movie Review Dataset)
* **Carregamento e Extração:** Download e leitura dos reviews categorizados em classes positivas (`True` / 1) e negativas (`False` / 0).
* **Divisão Amostral Estratificada:**
  * **Treinamento:** 20.000 avaliações balanceadas;
  * **Validação (Desenvolvimento):** 5.000 avaliações;
  * **Teste:** 25.000 avaliações para inferência cega final.

### 3. Tokenização e Representação com Hugging Face
* **Tokenizador BERT (`bert-base-uncased` e `bert-base-cased`):** Processamento de sentenças em sequências de índices (`input_ids`) e máscaras de atenção (`attention_mask`) com preenchimento (`padding`) e truncamento padronizado.
* **Classe `IMDBDataset`:** Estruturação personalizada de `torch.utils.data.Dataset` para empacotar vetores de entrada e rótulos de classificação.

### 4. Arquitetura da Rede Neural (`BertClassifier`)
* **Backbone:** Modelo pré-treinado BERT base (12 camadas Transformer, 768 dimensões ocultas, 12 cabeças de atenção e ~110M parâmetros).
* **Camada de Classificação:** Extração do embedding do token especial `[CLS]` (via cabeça de pooling do BERT), seguida de camada de regularização `Dropout(p=0.2)` e projeção linear (`Linear(768, 2)`).

### 5. Laço de Treinamento com Acumulação de Gradientes (`Gradient Accumulation`)
* **Otimização com Restrição de Memória:** Implementação explícita de passos de forward e backward distribuídos em $n$ etapas com normalização da função de perda (`loss / bs_acc_step`) antes do passo do otimizador (`optimizer.step()`).
* **Função de Custo e Otimizador:** `CrossEntropyLoss` e otimizador `AdamW` com decaimento de peso (`weight_decay=0.01`).

### 6. Verificação de Overfitting em Lote Reduzido
* **Validação de Capacidade:** Treinamento deliberado em subconjunto restrito (100 amostras) por 7 épocas, comprovando capacidade de memorização rápida (acurácia de 100% e perda tendendo a 0).

### 7. Treinamento Completo e Curvas de Aprendizado
* **Fine-Tuning Supervisionado:** Treinamento das épocas completas, com monitoramento simultâneo da perda e acurácia no treino e validação, salvando os pesos de melhor desempenho (`best_bert_model.pt`).
* **Curvas de Aprendizado:** Gráficos comparativos de perda (`Cross-Entropy Loss`) e acurácia ao longo das épocas.

### 8. Avaliação no Conjunto de Teste
* **Métricas Globais:** Acurácia final no conjunto de teste.
* **Relatório Detalhado:** Matriz de Confusão e Relatório de Classificação (`Precision`, `Recall`, `F1-Score` por classe de sentimento).

---

## Ferramentas Utilizadas

Em conformidade com as diretrizes de transparência acadêmica:

* **Ferramenta:** Google Colab / Antigravity IDE
  * **Utilização:** Ambiente de desenvolvimento, implementação dos modelos PyTorch, execução dos testes de treinamento e geração dos artefatos.
* **Ferramenta:** Gemini 3.7 Flash (High)
  * **Utilização:** Apoio na estruturação da arquitetura `BertClassifier`, laço de acumulação de gradientes e documentação técnica.
* **Ferramenta:** GitHub MCP Server
  * **Utilização:** Criação e sincronização automatizada do repositório no GitHub.

---

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/aj1no/PLN_Atividade_05_BERT.git
   cd PLN_Atividade_05_BERT
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # ou
   .\venv\Scripts\activate   # Windows
   ```

3. Instale as dependências necessárias:
   ```bash
   pip install torch torchvision transformers datasets scikit-learn matplotlib seaborn jupyter
   ```

4. Inicie o Jupyter Notebook:
   ```bash
   jupyter notebook 05_BERT.ipynb
   ```

---

## Licença

Este projeto está sob a licença [MIT](LICENSE).
