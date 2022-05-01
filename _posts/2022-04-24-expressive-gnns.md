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

Each node $$i$$ has an associated representation $$h_i^t \in \mathbb{R}^n$$ and (discrete or continuous) label $$y$$, for each GNN layer $$t \in \{1, \dots, T\}$$. Each node $$i$$ starts off with $$h_i^1 = x_i$$, where $$x_i \in \mathbb{R}^n$$ is the input features for the node. Edges $$e_{ij}$$ can also have an associated representation $$a_{ij}^t \in \mathbb{R}^m$$ depending on context. Each GNN layer $$t$$ performs a single step of \textit{Message Passing}. This involves combining the target node representation $$h^{t}_i$$ with the node representations $$h_j^{t}$$ from the neighbourhood $$\mathcal{N}_i$$ (Equation \ref{mp}). Intuitively, at a layer $$t$$, the GNN looks at the $$t$$-hop neighbourhood of $$i$$, represented by a subtree rooted at node $$i$$.

$$
\begin{equation} \label{mp}
    h^{t+1}_i = \sigma(\psi(h^{t}_i,~\square(\{h^t_j : j \in \mathcal{N}_i\})))
\end{equation}
$$

Here, $$\psi$$ is any affine transformation function (like a MLP) and $$\sigma$$ is a non-linear, element-wise activation function (like \textit{Sigmoid}, \textit{ReLU}, or \textit{Softmax}). $$\square$$ is a permutation-invariant aggregation function that combines neighbouring node features; choices include "sum", "max", "min", and "mean". This aggregation function can be thought of as a hashing function that operates on multisets (sets with legal repetition) of node features. The same can be done to edge representation $$a_{ij}^t$$ (Equation \ref{edge_mp}):

$$
\begin{equation} \label{edge_mp}
    a_{ij}^{t+1} = \phi(a_{ij}^t, h^t_i, h^t_j)
\end{equation}
$$

$$\phi : \mathbb{R}^m \rightarrow \mathbb{R}^m$$, parameterised by $$\theta$$, takes in current edge feature $$a_{ij}^t$$, and respective node features $$h^t_i$$ and $$h^t_j$$, to output the new edge representation $$a_{ij}^{t+1}$$ for the next layer.

## Understanding Expressiveness

### Graph Isomorphism

### Weisfeiler-Leman GI Test

### Higher-Order Structures

### Weisfeiler-Leman Hierarchy

## Notable Works

## Conclusion