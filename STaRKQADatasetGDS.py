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
        if os.path.exists(path):
            self.load(path)
        else:
            print(f"Processed dataset file not found at '{path}'. Chunked processing may be in progress; skipping load.")

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

        # Prepare dataframe and answer ids
        dataframe = self.raw_dataset.data.loc[self.raw_dataset.indices]
        answer_ids = {index : eval(qa_row[2]) for index, qa_row in dataframe.iterrows()}
        indices = list(dataframe.index)

        # Cypher query retrieval
        with open(f"configs/retrieval_config_v{self.retrieval_config_version}.yaml", "r") as f:
            cypher_config = yaml.safe_load(f)

        base_subgraph_folder = os.path.join(os.path.dirname(__file__), f'base_subgraphs/v{self.retrieval_config_version}/')
        base_subgraph_file = f"{base_subgraph_folder}{self.split}_data_base_subgraph.pt"
        base_subgraph_chunks_folder = os.path.join(base_subgraph_folder, 'chunks', self.split)

        # Chunk size (env var overrides config, default 100)
        default_chunk_size = 100
        chunk_size_env = os.getenv('CHUNK_SIZE')
        try:
            chunk_size_from_env = int(chunk_size_env) if chunk_size_env is not None else None
        except ValueError:
            chunk_size_from_env = None
        chunk_size = chunk_size_from_env or cypher_config.get('chunk_size', default_chunk_size)

        # Base subgraph: load or build in chunks
        if os.path.exists(base_subgraph_file):
            print(f"Load precomputed base subgraphs from disk...")
            base_subgraph = torch.load(base_subgraph_file, weights_only=False)
        else:
            print("Retrieve base subgraphs for each question in chunks...")
            os.makedirs(base_subgraph_folder, exist_ok=True)
            os.makedirs(base_subgraph_chunks_folder, exist_ok=True)

            num_items = len(indices)
            num_chunks = (num_items + chunk_size - 1) // chunk_size

            existing_chunk_files = [f for f in os.listdir(base_subgraph_chunks_folder) if f.startswith('base_chunk_') and f.endswith('.pt')]
            processed_chunk_ids = set(int(f.split('.')[0].split('_')[-1]) for f in existing_chunk_files)

            for chunk_id in range(num_chunks):
                if chunk_id in processed_chunk_ids:
                    continue
                start = chunk_id * chunk_size
                end = min(start + chunk_size, num_items)
                chunk_indices = indices[start:end]

                chunk_base_subgraph = {}
                for index in tqdm(chunk_indices, desc=f"BaseSubgraph chunk {chunk_id+1}/{num_chunks}"):
                    question_id, _, _ = dataframe.loc[index]
                    query_emb = self.query_embedding_dict[question_id].numpy()[0]
                    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                        topk_node_ids = get_nodes_by_vector_search(query_emb, 25*cypher_config['k_nodes'], driver)[:cypher_config['k_nodes']]
                        relationships_df = cypher_retrieval(topk_node_ids, driver)
                    nodes = np.unique(np.concatenate((relationships_df['sourceNodeId'].values, relationships_df['targetNodeId'].values)))
                    nodes_df = pd.DataFrame({'nodeId': nodes})
                    chunk_base_subgraph[index] = (nodes_df, relationships_df)

                torch.save(chunk_base_subgraph, os.path.join(base_subgraph_chunks_folder, f"base_chunk_{chunk_id:06d}.pt"))

            # Merge base subgraph chunks
            base_subgraph = {}
            merge_chunk_files = [f for f in os.listdir(base_subgraph_chunks_folder) if f.startswith('base_chunk_') and f.endswith('.pt')]
            if len(merge_chunk_files) != num_chunks:
                print("Base subgraph chunks incomplete. Exiting early to resume later.")
                return
            for fpath in sorted(merge_chunk_files):
                part = torch.load(os.path.join(base_subgraph_chunks_folder, fpath), weights_only=False)
                base_subgraph.update(part)

            torch.save(base_subgraph, base_subgraph_file)
            compute_intermediate_metrics(answer_ids, {k: v[0]['nodeId'].tolist() for k,v in base_subgraph.items()})

        # PCST subgraph pruning (chunked)
        print(f"Compute PCST graphs in chunks...")

        with open(f"configs/algo_config_v{self.algo_config_version}.yaml", "r") as f:
            pcst_config = yaml.safe_load(f)
        skip_pcst = pcst_config.get('skip_pcst', False)
        if skip_pcst:
            print("Skipping PCST...")

        pcst_chunk_size = (chunk_size_from_env or pcst_config.get('chunk_size', chunk_size))
        pcst_chunks_folder = os.path.join(os.path.dirname(__file__), f'processed_chunks/v{self.retrieval_config_version}_algo_v{self.algo_config_version}/{self.split}/')
        os.makedirs(pcst_chunks_folder, exist_ok=True)

        num_items = len(indices)
        num_chunks = (num_items + pcst_chunk_size - 1) // pcst_chunk_size

        existing_pcst_chunk_files = [f for f in os.listdir(pcst_chunks_folder) if f.startswith('chunk_') and f.endswith('.pt')]
        processed_pcst_chunk_ids = set(int(f.split('.')[0].split('_')[-1]) for f in existing_pcst_chunk_files)

        for chunk_id in range(num_chunks):
            if chunk_id in processed_pcst_chunk_ids:
                continue
            start = chunk_id * pcst_chunk_size
            end = min(start + pcst_chunk_size, num_items)
            chunk_indices = indices[start:end]

            chunk_data_by_index = {}
            chunk_pcst_nodes_by_index = {}

            for index in tqdm(chunk_indices, desc=f"PCST chunk {chunk_id+1}/{num_chunks}"):
                question_id, prompt, _ = dataframe.loc[index]
                query_emb = self.query_embedding_dict[question_id].numpy()[0]
                nodes_df, relationships_df = base_subgraph[index]

                if not skip_pcst:
                    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                        topn_nodes = get_nodes_by_vector_search(query_emb, pcst_config["prized_nodes"], driver)
                        topk_nodes = get_nodes_by_vector_search(query_emb, pcst_config["topk_nodes"], driver)

                    assign_node_prizes(nodes_df, topn_nodes)
                    assign_edge_costs(relationships_df)

                    gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
                    with gds.graph.construct(graph_name='pcst-graph', nodes=nodes_df, relationships=relationships_df.drop(['sourceNodeType','targetNodeType'], axis=1), undirected_relationship_types=['*']) as G:
                        pcst_output = gds.prizeSteinerTree.stream(G, prizeProperty='nodePrize', relationshipWeightProperty='edgeCost')
                    pcst_nodes, pcst_edges = convert_pcst_output(pcst_output)
                    pcst_nodes = np.unique(np.concatenate((pcst_nodes, topk_nodes)))
                else:
                    pcst_nodes = nodes_df['nodeId'].values
                    pcst_edges = np.stack((relationships_df['sourceNodeId'].values, relationships_df['targetNodeId'].values), axis=1) if len(relationships_df) > 0 else np.empty((0, 2), dtype=int)

                with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                    textual_nodes_df = get_textual_nodes(pcst_nodes, driver)
                    textual_edges_df = get_textual_edges(pcst_edges, driver)
                    answers = get_textual_nodes(answer_ids[index], driver)['name'].tolist()

                textual_nodes_df['vector_similarity'] = textual_nodes_df.apply(lambda row: row['textEmbedding'] @ query_emb, axis=1)
                textual_nodes_df = textual_nodes_df.sort_values(by=['vector_similarity'], ascending=False)
                chunk_pcst_nodes_by_index[index] = textual_nodes_df['nodeId'].tolist()

                desc = textualize_graph(textual_nodes_df, textual_edges_df)
                node_embedding = torch.tensor(textual_nodes_df['textEmbedding'].tolist())
                consecutive_map = {id : i for i, id in enumerate(textual_nodes_df['node_id'].values)}
                edge_index = torch.tensor([(consecutive_map[src], consecutive_map[tgt]) for src, tgt in pcst_edges], dtype=torch.int32).T if len(pcst_edges) > 0 else torch.empty((2,0), dtype=torch.int32)
                enriched_data = Data(
                    x=node_embedding,
                    edge_index=edge_index,
                    edge_attr=None,
                    question=f"Question: {prompt}\nAnswer: ",
                    label=('|').join(answers).lower(),
                    desc=desc,
                )
                chunk_data_by_index[index] = enriched_data

            torch.save({'data_by_index': chunk_data_by_index, 'pcst_nodes_by_index': chunk_pcst_nodes_by_index}, os.path.join(pcst_chunks_folder, f"chunk_{chunk_id:06d}.pt"))

        # Merge all PCST chunks to final processed output
        merge_pcst_chunk_files = [f for f in os.listdir(pcst_chunks_folder) if f.startswith('chunk_') and f.endswith('.pt')]
        if len(merge_pcst_chunk_files) != num_chunks:
            print("PCST chunks incomplete. Exiting early to resume later.")
            return

        merged_data_by_index = {}
        all_pcst_nodes = {}
        for fpath in sorted(merge_pcst_chunk_files):
            part = torch.load(os.path.join(pcst_chunks_folder, fpath), weights_only=False)
            merged_data_by_index.update(part['data_by_index'])
            all_pcst_nodes.update(part['pcst_nodes_by_index'])

        retrieval_data = [merged_data_by_index[idx] for idx in indices]
        compute_intermediate_metrics(answer_ids, all_pcst_nodes)
        self.save(retrieval_data, self.processed_paths[0])