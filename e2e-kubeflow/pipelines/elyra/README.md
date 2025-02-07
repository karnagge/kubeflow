# Elyra Pipeline para Treinamento XGBoost

Este diretório contém uma implementação da pipeline de treinamento usando Elyra, que permite executar e testar a pipeline localmente antes de implantá-la no Kubeflow.

## Pré-requisitos

1. Instalar as dependências do projeto:
```bash
pip install -r ../../requirements.txt
```

Isso instalará todas as dependências necessárias, incluindo o Elyra com todos os seus componentes adicionais.

2. Instalar o Elyra no VSCode:
   - Abra o VSCode
   - Vá para a aba de extensões (Ctrl+Shift+X)
   - Procure por "Elyra"
   - Instale a extensão "Elyra" da IBM

## Estrutura da Pipeline

A pipeline consiste em 3 notebooks que devem ser executados em sequência:

1. `01_data_generation.ipynb`: Gera os dados sintéticos para treinamento
2. `02_preprocessing.ipynb`: Realiza o pré-processamento dos dados
3. `03_training.ipynb`: Treina o modelo XGBoost e salva os resultados

## Executando a Pipeline

1. Abra o VSCode no diretório do projeto

2. Abra o Elyra Pipeline Editor:
   - Clique com o botão direito em qualquer notebook
   - Selecione "Open With" > "Pipeline Editor"

3. Crie uma nova pipeline:
   - Arraste os notebooks da pasta `elyra` para o editor na ordem correta
   - Conecte os notebooks na sequência: data_generation -> preprocessing -> training
   - Configure os parâmetros de entrada/saída em cada notebook conforme necessário

4. Execute a pipeline:
   - Clique no botão "Run Pipeline" no editor
   - Selecione "Local Runtime" como ambiente de execução
   - Clique em "Run"

## Estrutura de Diretórios

```
elyra/
├── 01_data_generation.ipynb  # Geração de dados
├── 02_preprocessing.ipynb    # Pré-processamento
├── 03_training.ipynb        # Treinamento do modelo
└── README.md                # Este arquivo
```

## Resultados

Após a execução, você encontrará:

- Dados gerados em: `../data/`
- Dados processados em: `../data/processed/`
- Modelo treinado e métricas em: `../models/xgboost/`

## Verificação da Instalação

Para verificar se o Elyra foi instalado corretamente, você pode executar:
```bash
elyra-metadata list runtimes
```

Isso deve mostrar os ambientes de execução disponíveis, incluindo o runtime local.