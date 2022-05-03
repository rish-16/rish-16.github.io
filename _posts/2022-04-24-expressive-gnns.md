---
title: '[WIP] Expressive GNNs and Where To Find Them'
date: 2022-04-24
permalink: /posts/expressive-gnns/
tags:
    - graph deep learning
    - expressiveness
    - routing problems
---

> This blog post consists of research I'm currently engaged with at NUS. My findings span the past half-year's worth of reading the literature and I'm excited to explain these concepts to you from scratch!

---

{: class="table-of-content"}
* TOC
{:toc}

## Foreword

At NUS, I'm currently looking into Graph Neural Network expressiveness and I'm really excited about it. I've spoken to a a few seniors in this area and had learned so much over the past few months. It's an exciting line of research within Graph DL and I'm sure the pace will pick up soon. It's more theoretical than practical currently but there's definitely room for expansion. In fact, I'm currently looking into the practical aspects of expressiveness at NUS, which seems to be an underexplored niche.

Before we get into the meat of the topic, let's get some preliminaries off the list first!

## Graph Neural Networks

### Graphs 

A graph is a data structure consists of nodes/vertices $$V$$ and edges $$E$$. These nodes represent entities or objects and the edges between them denote some relationship. These relationships are either unidirectional or bidirection. Let's assume we're working with undirected graphs for the rest of this post. 

### Graph Neural Networks

A graph $$\mathcal{G}(V, E)$$ consists of vertices (or nodes) $$v \in V$$ and edges $$e_{ij} \in E \subseteq {V \times V}$$ joining two nodes $$i$$ and $$j$$. $$e_{i,j} = 1$$ if there's a connection between nodes $$i$$ and $$j$$, 0 otherwise. The neighbourhood of a node $$i$$, namely $$\mathcal{N}_i$$, is a defined as the set of all nodes with outgoing edges to and incoming edges from $$i$$; formally, $$\mathcal{N}_i = \{j : e_{ij} \in E\}$$. 

Each node $$i$$ has an associated representation $$h_i^t \in \mathbb{R}^n$$ and (discrete or continuous) label $$y$$, for each GNN layer $$t \in \{1, \dots, T\}$$. Each node $$i$$ starts off with $$h_i^1 = x_i$$, where $$x_i \in \mathbb{R}^n$$ is the input features for the node. Edges $$e_{ij}$$ can also have an associated representation $$a_{ij}^t \in \mathbb{R}^m$$ depending on context. Each GNN layer $$t$$ performs a single step of _Message Passing_. This involves combining the target node representation $$h^{t}_i$$ with the node representations $$h_j^{t}$$ from the neighbourhood $$\mathcal{N}_i$$ (Equation \ref{mp}). Intuitively, at a layer $$t$$, the GNN looks at the $$t$$-hop neighbourhood of $$i$$, represented by a subtree rooted at node $$i$$.

$$
\begin{equation} \label{mp}
    h^{t+1}_i = \sigma(\psi(h^{t}_i,~\square(\{h^t_j : j \in \mathcal{N}_i\})))
\end{equation}
$$

Here, $$\psi$$ is any affine transformation function (like a MLP) and $$\sigma$$ is a non-linear, element-wise activation function (like _Sigmoid_, _ReLU_, or _Softmax_. $$\square$$ is a permutation-invariant aggregation function that combines neighbouring node features; choices include "sum", "max", "min", and "mean". This aggregation function can be thought of as a hashing function that operates on multisets (sets with legal repetition) of node features. The same can be done to edge representation $$a_{ij}^t$$ (Equation \ref{edge_mp}):

$$
\begin{equation} \label{edge_mp}
    a_{ij}^{t+1} = \phi(a_{ij}^t, h^t_i, h^t_j)
\end{equation}
$$

$$\phi : \mathbb{R}^m \rightarrow \mathbb{R}^m$$, parameterised by $$\theta$$, takes in current edge feature $$a_{ij}^t$$, and respective node features $$h^t_i$$ and $$h^t_j$$, to output the new edge representation $$a_{ij}^{t+1}$$ for the next layer.

## Understanding Expressiveness

__Expressiveness__ refers to the level at which a graph neural network can discriminate between two dissimiliar graphs. This brings us to the concept of _graph isomorphism_.

### Graph Isomorphism
Formally, two graphs are isomorphic if there exists a bijection (1:1 mapping) between its edges. This means the connectivities of the graphs should be alike. Trivially,

### Weisfeiler-Leman GI Test

Two graphs are isomorphic if there exists a bijection between the vertex sets of both graphs. As such,  the most notable algorithm for graph isomorphism is the __Weisfeiler-Leman__ (WL) test. All nodes are assigned an initial _colour_ (node-wise discrete label) and through iterations of naive vertex refinement, the colours of nodes are updated by incorporating it with the colours of neighbouring nodes. This is done using a hash function that takes in a multiset of neighbouring node colours that outputs a unique label for the next round of refinement. The test determines two graphs are non-isomorphic if the distribution of new colours differ at some iteration. 


However, the WL test is necessary but insufficient to show graph isomorphism as there exist pairs of non-isomorphic graphs that are indistinguishable using the method __ADDFIG__.

### Higher-Order Structures

### Weisfeiler-Leman Hierarchy

Expressiveness refers to the ability of a GNN to discriminate two graphs. The inability to learn structural information from graphs results in _over-smoothing_ – when two different nodes are assigned the same embedding representation in latent space, thereby being classified as the same. Therefore, structural awareness is important as it imbues inductive biases such as invariance to positions of nodes into the GNN, thereby allowing it to tell apart graphs.

The vanilla WL test examines individual nodes and looks at their immediate 1-hop neighbourhood. GNNs capable of discerning graphs using this 1-hop neighbourhood are called 1-WL GNNs. More formally, we say the GNN is as _powerful_ as 1-WL. We can generalise this to the $$k$$-hop neighbourhood where $$k \in \{2, 3, \dots\}$$. This wider neighbourhood can be viewed as a larger multiset of neighbours and _their_ neighbours, forming higher-order structures. When a GNN is able to discern these higher-order structures, we call it a $$k$$-WL GNN, and claim the GNN is as powerful as $$k$$-WL. Expressiveness is measured using these different "levels" of $$k$$-WL, altogether forming the _WL Hierarchy_. A $$k$$-WL GNN is strictly weaker than a $(k+1)$-WL GNN in that there exists a graph that the latter can discriminate and the former cannot but the converse is not true.

In fact, regular Message Passing Neural Networks fail 1-WL because aggregation functions like "mean" and "max" cannot tell apart two non-identical graphs __ADDFIG__. To avoid such scenarios, we introduce injectivity to the aggregation – multiset hashing – function; the "sum" aggregator is one such example.

## Notable Works

## Conclusion