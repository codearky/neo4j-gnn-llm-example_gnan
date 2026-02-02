# Graph Construction Algorithms for Knowledge Graph Question Answering

## Overview

To apply GNAN to knowledge graph question answering, we develop two graph construction algorithms (algo3 and algo4) that retrieve relevant subgraphs from a large knowledge graph database. These algorithms balance two competing objectives: (1) retrieving sufficient information to answer multi-hop questions, and (2) maintaining manageable subgraph sizes for efficient processing by the GNN+LLM model.

## Base Retrieval Strategy

Both algorithms begin with a common retrieval strategy that leverages semantic similarity between the question and entities in the knowledge graph:

1. **Vector Search Initialization**: Given a question, we compute its embedding using a sentence transformer model (text-embedding-ada-002). We then perform vector search over node embeddings in the knowledge graph to retrieve the top-$k$ most semantically similar nodes as initial seeds.

2. **Configuration**: In our experiments, we retrieve the top 4 nodes from an initial candidate set of 100 nodes ranked by cosine similarity to the question embedding.

## Algorithm 3: PCST with Merge Strategy

Algorithm 3 constructs subgraphs using a hybrid approach that combines two complementary graph extraction methods:

### Step 1: Base Subgraph Retrieval

From the initial seed nodes, we perform a 1-hop expansion to collect all neighboring nodes and edges, creating a base candidate subgraph $G_{base}$.

### Step 2: Prize-Collecting Steiner Tree (PCST)

We apply the Prize-Collecting Steiner Tree algorithm to identify the most relevant subgraph structure:

1. **Node Prize Assignment**: We query the knowledge graph for the top 100 nodes most similar to the question embedding. These nodes are assigned prizes proportional to their ranking:
   $$\text{prize}(v) = \frac{4}{\text{num\_prized\_nodes}} \times (100 - \text{rank}(v))$$

2. **Edge Cost Assignment**: All edges in the base subgraph are assigned a uniform cost of 0.5.

3. **PCST Optimization**: The Prize-Collecting Steiner Tree algorithm finds a tree $T_{PCST}$ that maximizes:
   $$\max_{T \subseteq G_{base}} \left( \sum_{v \in T} \text{prize}(v) - \sum_{e \in T} \text{cost}(e) \right)$$

   This optimization balances collecting high-prize nodes (semantically relevant to the question) while minimizing tree size (edge costs).

4. **Top-k Union**: We take the union of $T_{PCST}$ with the top-25 most similar nodes to ensure critical entities are always included:
   $$V_{PCST} = V(T_{PCST}) \cup \text{Top25}(G_{base}, q)$$

### Step 3: Similarity-Based Filtering

In parallel, we construct a similarity-based subgraph:

1. Compute vector similarity between all nodes in $G_{base}$ and the question embedding
2. Select the top-200 most similar nodes: $V_{sim}$
3. Construct the induced subgraph containing only edges between nodes in $V_{sim}$

### Step 4: Merge Strategy

The final subgraph is created by merging both approaches:

$$V_{final} = V_{PCST} \cup V_{sim}$$

$$E_{final} = \{(u,v) \in E(G_{base}) : u, v \in V_{final}\}$$

This merge strategy ensures we capture both:
- **Structural importance** via PCST (nodes that connect relevant entities)
- **Semantic relevance** via similarity filtering (nodes most similar to the question)

**Configuration (algo_config_v3.yaml):**
```yaml
mode: merge
prized_nodes: 100
topk_nodes: 25
max_nodes_no_pcst: 200
```

## Algorithm 4: Multi-Hop Expansion with PCST

Algorithm 4 extends Algorithm 3 with an adaptive multi-hop expansion strategy that explores deeper neighborhoods while controlling graph size through similarity-based pruning.

### Key Difference: Hop Expansion with Per-Seed Pruning

Before applying PCST, Algorithm 4 expands the initial base subgraph through multiple hops:

**Expansion Procedure** (`expand_graph_by_hops`):

Starting from the initial seed nodes $S_0$, we iteratively expand for $h$ hops:

For each hop $i = 1, \ldots, h$:
1. For each node $u$ in the current frontier $S_{i-1}$:
   - Retrieve all neighbors $N(u)$ in the knowledge graph
   - Compute similarity scores: $\text{sim}(v) = \text{emb}(v) \cdot \text{emb}(q)$ for each $v \in N(u)$
   - Select top-$k_{seed}$ neighbors by similarity (per-seed pruning)

2. Update frontier: $S_i = \bigcup_{u \in S_{i-1}} \text{TopK}(N(u), k_{seed})$

3. Collect edges: Add all selected edges to the base subgraph

This expansion strategy enables:
- **Multi-hop reasoning**: Reaching entities that are 2-3 hops away from initial seeds, supporting complex multi-hop questions
- **Controlled growth**: Limiting expansion to the most relevant neighbors at each hop prevents exponential graph explosion
- **Adaptive pruning**: Per-seed selection ensures diverse exploration while maintaining focus on question-relevant paths

After expansion, Algorithm 4 applies the same PCST + merge strategy as Algorithm 3 (Steps 2-4 above).

**Configuration (algo_config_v4.yaml):**
```yaml
mode: merge
prized_nodes: 100
topk_nodes: 25
max_nodes_no_pcst: 200
expand_hops: 3              # NEW: multi-hop expansion
expand_topk_per_seed: 5     # NEW: per-seed neighbor limit
```

## Comparative Analysis

| Aspect | Algorithm 3 | Algorithm 4 |
|--------|-------------|-------------|
| Initial Retrieval | 1-hop from top-4 seeds | 1-hop from top-4 seeds |
| Graph Expansion | None | 3-hop with top-5 per seed |
| PCST Application | On 1-hop subgraph | On expanded subgraph |
| Typical Subgraph Size | 200-400 nodes | 300-600 nodes |
| Multi-hop Capability | Limited (1-hop only) | Enhanced (up to 3-hop paths) |
| Best For | Single-hop questions | Complex multi-hop questions |

## Implementation Details

### Vector Search with Neo4j

We use Neo4j's native vector index for efficient similarity search:

```python
def get_nodes_by_vector_search(query_embedding, k_nodes, driver):
    res = driver.execute_query("""
    CALL db.index.vector.queryNodes($index, $k, $query_embedding)
    YIELD node
    RETURN node.nodeId AS nodeId
    """, parameters_={"index": "text_embeddings", "k": k_nodes,
                      "query_embedding": query_embedding})
    return [rec.data()['nodeId'] for rec in res.records]
```

### PCST with Neo4j Graph Data Science

We leverage the Neo4j Graph Data Science library's optimized PCST implementation:

```python
from graphdatascience import GraphDataScience

gds = GraphDataScience(NEO4J_URI, auth=(username, password))
with gds.graph.construct('pcst-graph', nodes=nodes_df,
                         relationships=edges_df,
                         undirected_relationship_types=['*']) as G:
    pcst_output = gds.prizeSteinerTree.stream(
        G,
        prizeProperty='nodePrize',
        relationshipWeightProperty='edgeCost'
    )
```

### Node Embeddings

All nodes in the knowledge graph are pre-embedded using the same sentence transformer model (text-embedding-ada-002) that embeds the questions. This ensures semantic consistency between question and node representations, enabling effective vector search and similarity-based filtering.

## Rationale and Design Choices

### Why PCST?

The Prize-Collecting Steiner Tree algorithm is particularly well-suited for knowledge graph question answering:

1. **Balances relevance and connectivity**: PCST naturally trades off including semantically relevant nodes (high prizes) with maintaining a connected structure (edge costs)

2. **Handles multi-hop reasoning**: By finding optimal connecting paths between relevant entities, PCST can capture multi-hop relationships without exhaustive expansion

3. **Theoretically grounded**: Unlike heuristic methods, PCST provides an optimization framework with provable approximation guarantees

### Why Merge Strategy?

Combining PCST and similarity filtering addresses complementary failure modes:

- **PCST alone** may miss highly relevant but structurally isolated nodes
- **Similarity filtering alone** may include disconnected entities without capturing relationships
- **Merged approach** ensures both semantic relevance and structural coherence

### Why Multi-Hop Expansion (Algo4)?

The 3-hop expansion in Algorithm 4 is motivated by analysis of question complexity:

- Many questions in STaRK-QA require 2-3 hop reasoning paths
- Simple 1-hop retrieval (Algo3) may miss critical intermediate entities
- Controlled per-seed expansion prevents graph explosion while reaching distant relevant nodes

## Integration with GNAN

These graph construction algorithms provide the input subgraphs for the GNAN model. Each constructed subgraph becomes a graph instance in the training/evaluation dataset:

```python
gnn = TensorGNAN(
    in_channels=1536,      # node embedding dimension
    hidden_channels=1536,
    out_channels=1536,
    n_layers=4,
    normalize_rho=True,
    feature_groups=[list(range(1536))],
)
```

The GNAN model then learns to:
1. Weight nodes by their distance from each other (via $\rho$ function)
2. Transform node embeddings (via feature group shape functions)
3. Aggregate information for answer prediction

The interpretability of GNAN is particularly valuable here, as it can reveal which nodes in the constructed subgraph were most important for answering each question, providing insights into whether the graph construction algorithms successfully retrieved the right information.

---

## Code References

- Graph construction: [STaRKQADatasetGDS.py:59-117](STaRKQADatasetGDS.py#L59-L117) (expand_graph_by_hops)
- PCST implementation: [STaRKQADatasetGDS.py:307-324](STaRKQADatasetGDS.py#L307-L324)
- Merge strategy: [STaRKQADatasetGDS.py:325-346](STaRKQADatasetGDS.py#L325-L346)
- Algorithm configs: [configs/algo_config_v3.yaml](configs/algo_config_v3.yaml), [configs/algo_config_v4.yaml](configs/algo_config_v4.yaml)
