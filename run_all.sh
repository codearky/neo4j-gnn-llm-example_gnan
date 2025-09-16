wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg
echo 'deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install -y neo4j=1:2025.07.0
cp /var/lib/neo4j/products/* /var/lib/neo4j/plugins/
echo 'dbms.security.procedures.allowlist=gds.*' >> /etc/neo4j/neo4j.conf
NEO4J_PASSWORD=${NEO4J_PASSWORD:-test12345}
neo4j stop || true
neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD" || true
neo4j start
curl -sS -u neo4j:"$NEO4J_PASSWORD" -H 'Content-Type: application/json' -d '{"statements":[{"statement":"RETURN 1 AS test"}]}' http://localhost:7474/db/neo4j/tx/commit
cd data-loading
python emb_download.py
python load_data.py
cd ..
NEO4J_URI=bolt://localhost:7687 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=test12345 python train.py --checkpointing --llama_version llama3.1-8b --retrieval_config_version 0 --g_retriever_config_version 0 --eval_batch_size 4 --num_gnn_layers 4 --algo_config_version 0 --num_gpus 4