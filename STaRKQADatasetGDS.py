import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from graphdatascience import GraphDataScience
from pandas import DataFrame
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from compute_metrics import compute_intermediate_metrics


def get_nodes_by_vector_search(query_embedding: np.ndarray, k_nodes: int, driver: Driver) -> list[int]:
    res = driver.execute_query("""
    CALL db.index.vector.queryNodes($index, $k, $query_embedding) YIELD node
    RETURN node.nodeId AS nodeId
    """,
                               parameters_={
                                   "index": "text_embeddings",
                                   "k": k_nodes,
                                   "query_embedding": query_embedding})
    return [rec.data()['nodeId'] for rec in res.records]

def cypher_retrieval(node_ids: list[int], driver: Driver):
    res = driver.execute_query("""
                    UNWIND $nodeIds AS nodeId
                    MATCH (m {nodeId:nodeId})-[r]->(n)
                    RETURN
                    m.nodeId as sourceNodeId, n.nodeId as targetNodeId, type(r) as relationshipType,
                    labels(m)[0] as sourceNodeType, labels(n)[0] as targetNodeType
                """,
                               parameters_={'nodeIds': node_ids})
    return pd.DataFrame([rec.data() for rec in res.records])

def get_textual_nodes(node_ids: list[int], driver: Driver) -> DataFrame:
    res = driver.execute_query("""
    UNWIND $nodeIds AS nodeId
    MATCH(node:_Entity_ {nodeId:nodeId})
    RETURN node.nodeId AS nodeId, node.name AS name, node.details AS description, node.textEmbedding AS textEmbedding
    """,
                               parameters_={"nodeIds": node_ids})
    return pd.DataFrame([rec.data() for rec in res.records])

def get_textual_edges(node_pairs: list[tuple[int, int]], driver: Driver) -> DataFrame:
    res = driver.execute_query("""
    UNWIND $node_pairs AS pair
    MATCH(src:_Entity_ {nodeId:pair[0]})-[e]->(tgt:_Entity_ {nodeId:pair[1]})
    RETURN src.nodeId AS src, type(e) AS edge_attr, tgt.nodeId AS dst
    """,
                               parameters_={"node_pairs": node_pairs})
    return pd.DataFrame([rec.data() for rec in res.records])

def textualize_graph(textual_nodes_df, textual_edges_df):
    textual_nodes_df.description.fillna("")
    textual_nodes_df['node_attr'] = textual_nodes_df.apply(
        lambda row: f"name: {row['name']}, description: {row['description']}", axis=1)
    textual_nodes_df.rename(columns={'nodeId': 'node_id'}, inplace=True)
    nodes_desc = textual_nodes_df.drop(['name', 'description', 'textEmbedding'], axis=1).to_csv(index=False)
    edges_desc = textual_edges_df.to_csv(index=False)
    return nodes_desc + '\n' + edges_desc

def assign_node_prizes(nodes_df, topn_nodes):
    nodes = nodes_df['nodeId'].tolist()
    node_prizes = {node: len(topn_nodes) - rank for rank, node in enumerate(topn_nodes)}
    node_prizes = [4 / len(topn_nodes) * node_prizes.get(node, 0) for node in nodes]
    nodes_df['nodePrize'] = node_prizes

def assign_edge_costs(relationships_df, topn_edges=None):
    edge_costs = .5 - np.zeros(len(relationships_df))
    relationships_df['edgeCost'] = edge_costs # No edge prizes for now (recall drops 3pts, f1 is about the same)

def convert_pcst_output(pcst_output) -> (np.array, np.array):
    pcst_src = pcst_output['nodeId'].values
    pcst_tgt = pcst_output['parentId'].values
    pcst_nodes = np.unique(np.concatenate((pcst_src, pcst_tgt)))
    pcst_edges = np.stack((pcst_src, pcst_tgt), axis=1)
    return pcst_nodes, pcst_edges

def convert_non_pcst_output(relationships_df, query_embedding: np.ndarray, driver: Driver, max_nodes: int = 1000) -> (np.array, np.array):
    """
    Filter the graph to top k nodes by vector similarity and keep edges between them.
    
    Args:
        relationships_df: DataFrame with sourceNodeId and targetNodeId columns
        query_embedding: Question embedding to compute similarity
        driver: Neo4j driver for fetching node embeddings
        max_nodes: Maximum number of nodes to keep (default: 1000)
    
    Returns:
        Tuple of (filtered_nodes, filtered_edges)
    """
    src = relationships_df['sourceNodeId'].values
    tgt = relationships_df['targetNodeId'].values
    unique_nodes = np.unique(np.concatenate([src, tgt]))
    
    # Get node embeddings and compute similarities
    textual_nodes_df = get_textual_nodes(unique_nodes.tolist(), driver)
    textual_nodes_df['vector_similarity'] = textual_nodes_df.apply(
        lambda row: row['textEmbedding'] @ query_embedding, axis=1
    )
    
    # Sort by similarity and take top k
    textual_nodes_df = textual_nodes_df.sort_values(by=['vector_similarity'], ascending=False)
    top_k_nodes = set(textual_nodes_df.head(max_nodes)['nodeId'].tolist())
    
    # Filter edges to only include edges between top k nodes
    filtered_edges = []
    for s, t in zip(src, tgt):
        if s in top_k_nodes and t in top_k_nodes:
            filtered_edges.append([s, t])
    
    if len(filtered_edges) > 0:
        filtered_edges = np.array(filtered_edges)
    else:
        filtered_edges = np.empty((0, 2), dtype=int)
    
    return np.array(list(top_k_nodes)), filtered_edges

class STaRKQADataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        raw_dataset: Dataset,
        retrieval_config_version: int,
        algo_config_version: int,
        split: str = "train",
        force_reload: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        self.split = split
        self.raw_dataset = raw_dataset
        self.retrieval_config_version = retrieval_config_version
        self.algo_config_version = algo_config_version
        self.query_embedding_dict = torch.load(os.path.join(os.path.dirname(__file__), 'data-loading/emb/prime/text-embedding-ada-002/query/query_emb_dict.pt')) # load from parent directory of this file

        super().__init__(root, force_reload=force_reload, transform=transform)

        path = self.processed_paths[0]
        self.load(path)

    @property
    def processed_file_names(self) -> list[str]:
        return [self.split + '_data.pt']

    def process(self) -> None:
        if not os.path.isfile('db.env'):
            print(f"Expected file 'db.env' was not found.")

        load_dotenv('db.env', override=True)
        NEO4J_URI = os.getenv('NEO4J_URI')
        NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

        retrieval_data = []

        dataframe = self.raw_dataset.data.loc[self.raw_dataset.indices]
        answer_ids = {index : eval(qa_row[2]) for index, qa_row in dataframe.iterrows()}

        # Cypher query retrieval
        with open(f"configs/retrieval_config_v{self.retrieval_config_version}.yaml", "r") as f:
            cypher_config = yaml.safe_load(f)

        base_subgraph_folder = os.path.join(os.path.dirname(__file__), f'base_subgraphs/v{self.retrieval_config_version}/')
        base_subgraph_file = f"{base_subgraph_folder}{self.split}_data_base_subgraph.pt"

        if os.path.exists(base_subgraph_file):
            print(f"Load precomputed base subgraphs from disk...")
            base_subgraph = torch.load(base_subgraph_file, weights_only=False)
        else:
            print("Retrieve base subgraphs for each question...")
            base_subgraph = {}
            for index, (question_id, _, _) in tqdm(dataframe.iterrows()):
                query_emb = self.query_embedding_dict[question_id].numpy()[0]
                with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                    topk_node_ids = get_nodes_by_vector_search(query_emb, 25*cypher_config['k_nodes'], driver)[:cypher_config['k_nodes']]
                    relationships_df = cypher_retrieval(topk_node_ids, driver) # Variations of cypher queries are supported here

                nodes = np.unique(np.concatenate((relationships_df['sourceNodeId'].values, relationships_df['targetNodeId'].values)))
                nodes_df = pd.DataFrame({'nodeId': nodes})

                base_subgraph[index] = (nodes_df, relationships_df)

            os.makedirs(base_subgraph_folder, exist_ok=True)
            torch.save(base_subgraph, base_subgraph_file)

            compute_intermediate_metrics(answer_ids, {k: v[0]['nodeId'].tolist() for k,v in base_subgraph.items()})

        # PCST subgraph pruning
        print(f"Compute PCST graphs...")
        print (f"has {len(dataframe)} rows to process")

        with open(f"configs/algo_config_v{self.algo_config_version}.yaml", "r") as f:
            pcst_config = yaml.safe_load(f)
        mode = pcst_config.get("mode", "pcst")  # one of: 'similarity', 'pcst', 'merge'
        textualize_similarity_graph = pcst_config.get("textualize_similarity_graph", False)
        textualize_pcst_graph = pcst_config.get("textualize_pcst_graph", True)
        max_nodes_no_pcst = pcst_config.get("max_nodes_no_pcst", 1000)
        assert mode in ('similarity', 'pcst', 'merge'), f"Invalid mode: {mode}. Expected one of 'similarity', 'pcst', or 'merge'."
        
        all_pcst_nodes = {} # for metrics only
        
        # Checkpoint setup
        checkpoint_file = os.path.join(self.processed_dir, f'{self.split}_checkpoint.pt')
        if os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, weights_only=False)
            retrieval_data = checkpoint['retrieval_data']
            all_pcst_nodes = checkpoint['all_pcst_nodes']
            start_idx = 6000 # checkpoint['last_index'] + 1
            print(f"Resuming from index {start_idx}")
        else:
            start_idx = 0
        
        dataframe_items = list(dataframe.iterrows())
        for i, (index, (question_id, prompt, _)) in enumerate(tqdm(dataframe_items[start_idx:], initial=start_idx, total=len(dataframe))):
            query_emb = self.query_embedding_dict[question_id].numpy()[0]
            nodes_df, relationships_df = base_subgraph[index]
            # initialize per-iteration holders
            sim_nodes, sim_edges = None, None
            pcst_nodes, pcst_edges = None, None
            final_nodes, final_edges = None, None

            if mode == 'similarity':
                with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                    sim_nodes, sim_edges = convert_non_pcst_output(relationships_df, query_emb, driver, max_nodes_no_pcst)
                final_nodes = sim_nodes
                final_edges = sim_edges
            elif mode in ('pcst', 'merge'):
                with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                    topn_nodes = get_nodes_by_vector_search(query_emb, pcst_config["prized_nodes"], driver)
                    topk_nodes = get_nodes_by_vector_search(query_emb, pcst_config["topk_nodes"], driver) # for union

                assign_node_prizes(nodes_df, topn_nodes) #adds column 'nodePrizes'
                assign_edge_costs(relationships_df) #adds column 'edgeCosts'

                # Run the pcst algorithm
                gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
                with gds.graph.construct(graph_name='pcst-graph', nodes=nodes_df, relationships=relationships_df.drop(['sourceNodeType','targetNodeType'], axis=1), undirected_relationship_types=['*']) as G:
                    pcst_output = gds.prizeSteinerTree.stream(G, prizeProperty='nodePrize', relationshipWeightProperty='edgeCost')
                pcst_nodes, pcst_edges = convert_pcst_output(pcst_output)

                # Take union with top25
                pcst_nodes = np.unique(np.concatenate((pcst_nodes, topk_nodes)))

                # Determine final graph (PCST only or merged with similarity)
                if mode == 'merge':
                    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                        sim_nodes, sim_edges = convert_non_pcst_output(relationships_df, query_emb, driver, max_nodes_no_pcst)
                    union_nodes = np.unique(np.concatenate((pcst_nodes, sim_nodes)))
                    # Build induced edges among union nodes from base relationships
                    src_all = relationships_df['sourceNodeId'].values
                    tgt_all = relationships_df['targetNodeId'].values
                    union_set = set(union_nodes.tolist())
                    induced_edges = []
                    for s, t in zip(src_all, tgt_all):
                        if s in union_set and t in union_set:
                            induced_edges.append([s, t])
                    final_nodes = union_nodes
                    if len(induced_edges) > 0:
                        final_edges = np.array(induced_edges)
                    else:
                        final_edges = np.empty((0, 2), dtype=int)
                else:
                    final_nodes = pcst_nodes
                    final_edges = pcst_edges
            else:
                raise ValueError(f"Unsupported mode: {mode}. Expected one of 'similarity', 'pcst', or 'merge'.")

            # Retrieve node embedding, label and textual graph description
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                textual_nodes_df_all = get_textual_nodes(final_nodes, driver)
                textual_edges_df_final = get_textual_edges(final_edges, driver)
                # Prepare PCST and similarity textual edges if applicable
                textual_edges_df_pcst = get_textual_edges(pcst_edges, driver) if (pcst_edges is not None) else None
                textual_edges_df_sim = get_textual_edges(sim_edges, driver) if (sim_edges is not None) else None
                answers = get_textual_nodes(answer_ids[index], driver)['name'].tolist()

            # Order nodes by similarity to question
            textual_nodes_df_all['vector_similarity'] = textual_nodes_df_all.apply(lambda row: row['textEmbedding'] @ query_emb, axis=1)
            textual_nodes_df_all = textual_nodes_df_all.sort_values(by=['vector_similarity'], ascending=False)
            # Keep node_id for mapping without mutating this DataFrame via textualize_graph
            textual_nodes_df_all = textual_nodes_df_all.copy()
            textual_nodes_df_all['node_id'] = textual_nodes_df_all['nodeId']
            # Metrics storage: store final nodes
            all_pcst_nodes[index] = textual_nodes_df_all['nodeId'].tolist()

            # Generate textualized graph
            desc = ""
            pcst_desc = ""
            # If PCST is in use, keep previous behavior for desc (PCST textualization), and also set pcst_desc if enabled
            if pcst_nodes is not None:
                tn_pcst = textual_nodes_df_all[textual_nodes_df_all['nodeId'].isin(pcst_nodes)].copy()
                if textualize_pcst_graph:
                    pcst_desc = textualize_graph(tn_pcst, textual_edges_df_pcst if textual_edges_df_pcst is not None else pd.DataFrame(columns=['src','edge_attr','dst']))
                # Maintain existing behavior for desc with PCST
                if mode in ('pcst', 'merge'):
                    desc = pcst_desc
            # Similarity textualization only when requested or when PCST is skipped
            if mode == 'similarity' and textualize_similarity_graph:
                tn_sim = textual_nodes_df_all.copy()  # final graph equals similarity graph here
                desc = textualize_graph(tn_sim, textual_edges_df_final)
            elif mode == 'merge' and textualize_similarity_graph and (sim_nodes is not None):
                tn_sim = textual_nodes_df_all[textual_nodes_df_all['nodeId'].isin(sim_nodes)].copy()
                sim_edges_df = textual_edges_df_sim if textual_edges_df_sim is not None else pd.DataFrame(columns=['src','edge_attr','dst'])
                desc = textualize_graph(tn_sim, sim_edges_df)

            node_embedding = torch.tensor(textual_nodes_df_all['textEmbedding'].tolist())
            consecutive_map = {id : i for i, id in enumerate(textual_nodes_df_all['node_id'].values)}
            edge_index = torch.tensor([(consecutive_map[src], consecutive_map[tgt]) for src, tgt in final_edges], dtype=torch.int32).T #when dtype is not specified, it becomes a float tensor when unserialized, weird.
            enriched_data = Data(
                x=node_embedding,
                edge_index=edge_index,
                edge_attr=None,
                question=f"Question: {prompt}\nAnswer: ",
                label=('|').join(answers).lower(),
                desc=desc,
            )
            # Attach PCST node list and pcst_desc if PCST was used
            if pcst_nodes is not None:
                enriched_data.pcst_nodes = list(map(int, pcst_nodes.tolist() if isinstance(pcst_nodes, np.ndarray) else pcst_nodes))
                enriched_data.pcst_desc = pcst_desc if textualize_pcst_graph else ""
            retrieval_data.append(enriched_data)
            
            # Save checkpoint every 500 iterations
            if (start_idx + i + 1) % 500 == 0:
                torch.save({
                    'retrieval_data': retrieval_data,
                    'all_pcst_nodes': all_pcst_nodes,
                    'last_index': i
                }, checkpoint_file)
                print(f"\nCheckpoint saved at iteration {start_idx + i + 1}")
        
        compute_intermediate_metrics(answer_ids, all_pcst_nodes)
        
        # Dataset graph statistics
        if len(retrieval_data) > 0:
            node_counts = [int(d.x.size(0)) for d in retrieval_data]
            edge_counts = [int(d.edge_index.size(1)) for d in retrieval_data]
            avg_nodes = float(np.mean(node_counts))
            med_nodes = float(np.median(node_counts))
            avg_edges = float(np.mean(edge_counts))
            med_edges = float(np.median(edge_counts))
            print(f"{self.split} dataset graph stats -> avg_nodes: {avg_nodes:.2f}, med_nodes: {med_nodes:.0f}, avg_edges: {avg_edges:.2f}, med_edges: {med_edges:.0f}")
        else:
            print(f"{self.split} dataset graph stats -> no graphs constructed")
        
        self.save(retrieval_data, self.processed_paths[0])
        
        # Clean up checkpoint file after successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"Checkpoint file removed after successful completion")