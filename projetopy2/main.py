from fastapi import FastAPI
import pandas as pd
import os

app = FastAPI()
alunos = pd.DataFrame()
datasets = {}  

def load_dataset(name: str, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo {path} não encontrado.")
    df = pd.read_csv(path)
    datasets[name] = df
    return df.head()

# Carregar os dados dos alunos
@app.post("/datasets/load/alunos")
async def load_alunos():
    global alunos  # <-- importante
    alunos = pd.DataFrame([
        {"id_aluno": 1, "nome": "João Silva", "curso": "Engenharia de Software", "periodo": 5,
         "disciplinas_cursadas": "Algoritmos, Banco de Dados, IA", "areas_interesse": "ML, Programação, Cloud"},
        {"id_aluno": 2, "nome": "Maria Souza", "curso": "Ciência de Dados", "periodo": 3,
         "disciplinas_cursadas": "Estatística, Python, ML", "areas_interesse": "IA, Visualização, Big Data"},
        {"id_aluno": 3, "nome": "Carlos Oliveira", "curso": "Engenharia de Computação", "periodo": 7,
         "disciplinas_cursadas": "Redes, SO, Arquitetura", "areas_interesse": "IoT, Segurança, Hardware"},
        {"id_aluno": 4, "nome": "Ana Pereira", "curso": "SI", "periodo": 4,
         "disciplinas_cursadas": "Gestão, Redes, BD", "areas_interesse": "UX, PM, Analytics"},
        {"id_aluno": 5, "nome": "Lucas Santos", "curso": "Engenharia de Software", "periodo": 6,
         "disciplinas_cursadas": "BD, IA, Web", "areas_interesse": "Mobile, DevOps, QA"}
    ])
    return {
        "message": "✅ Dataset 'alunos' carregado com sucesso!",
        "preview": alunos.head().to_dict(orient="records")
    }


# Buscar os alunos por nome
@app.get("/datasets/alunos/buscar")
async def buscar_aluno(nome: str):
    if alunos.empty:
        return {"error": "Dataset de alunos ainda não foi carregado."}
    
    # Busca parcial (case insensitive)
    resultados = alunos[alunos["nome"].str.contains(nome, case=False, na=False)]
    
    if resultados.empty:
        return {"message": f"Nenhum aluno encontrado com o nome '{nome}'."}
    
    return resultados.to_dict(orient="records")