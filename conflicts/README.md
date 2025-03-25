conda create -n rag python=3.10
conda activate rag
conda install pytorch=2.1.0 torchvision=0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
python embed_and_index.py

uvicorn app:app --reload --port 8000
curl -X POST "http://127.0.0.1:8000/rag/" -H "Content-Type: application/json" -d "{\"question\": \"I am anxious recently, what should I do？\"}"


