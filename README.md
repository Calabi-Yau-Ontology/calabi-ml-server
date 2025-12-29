# Calabi ML Server (NER & Suggestion)

FastAPI based Calabi NER / Term recommendation server

```
calabi-ml-server/
├── src
│   ├── nlp
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── dtos.py
│   │   ├── exceptions.py
│   │   ├── router.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   └── utils.py           
│   ├── config.py              
│   └── main.py
├── .env
├── Dockerfile
├── README.md
└── requirements.txt
```

## Run Locally
1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the server

```bash
uvicorn src.main:app --host 0.0.0.0
```