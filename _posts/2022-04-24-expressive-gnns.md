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

<img src="/images/2022-04-24-expressive-gnns/mp.png" width="100%">

__Figure 1:__ A node $$i$$ has a feature vector $$x_i \in \mathbb{R}^n$$ (coloured envelope) and has a neighbourhood $$\mathcal{N}_i$$ (left). A single round of Message Passing involves aggregating (collecting) representations from a target node's neighbourhood and incorporating them into its own representation, for all nodes in the graph in parallel (right).

<img src="/images/2022-04-24-expressive-gnns/khop.png" width="100%">

__Figure 2:__<b>(a)</b> is__Figure 1:__ the original graph. <b>(b)</b> is the rooted subtree of target node (green) at layer $$t=1$$. <b>(c)</b> is the rooted subtree of target node (green) at layer $$t=2$$. These rooted subtrees are multisets of node features.

## Understanding Expressiveness

__Expressiveness__ refers to the level at which a graph neural network can discriminate between two dissimiliar graphs. This brings us to the concept of _graph isomorphism_.

### Graph Isomorphism
Formally, two graphs are isomorphic if there exists a bijection (1:1 mapping) between its edges. This means the connectivities of the graphs should be alike. Trivially, if they are different, the two graphs are non-isomorphic.

### Weisfeiler-Leman GI Test

Two graphs are isomorphic if there exists a bijection between the vertex sets of both graphs. As such,  the most notable algorithm for graph isomorphism is the __Weisfeiler-Leman__ (WL) test. All nodes are assigned an initial _colour_ (node-wise discrete label) and through iterations of naive vertex refinement, the colours of nodes are updated by incorporating it with the colours of neighbouring nodes. This is done using a hash function that takes in a multiset of neighbouring node colours that outputs a unique label for the next round of refinement. The test determines two graphs are non-isomorphic if the distribution of new colours differ at some iteration. 

The key to working with node labels is the multiset hashing function. A multi

__Figure 3:__ The WL test performed on two graphs $$A$$ and $$B$$ that are isomorphic. 

However, the WL test is necessary but insufficient to show graph isomorphism as there exist pairs of non-isomorphic graphs that are indistinguishable using the method.

<img src="/images/2022-04-24-expressive-gnns/wlfail.png" width="100%">

__Figure 4:__ Examples of two graphs indistinguishable by the WL test. They produce similar distributions through the iterations of colour refinement. It's catastrophic if datasets have graphs that exhibit similar properties and can't be told apart.

### Higher-Order Structures

To understand

### Weisfeiler-Leman Hierarchy

Expressiveness refers to the ability of a GNN to discriminate two graphs. The inability to learn structural information from graphs results in _over-smoothing_ – when two different nodes are assigned the same embedding representation in latent space, thereby being classified as the same. Therefore, structural awareness is important as it imbues inductive biases such as invariance to positions of nodes into the GNN, thereby allowing it to tell apart graphs.

The vanilla WL test examines individual nodes and looks at their immediate 1-hop neighbourhood. GNNs capable of discerning graphs using this 1-hop neighbourhood are called 1-WL GNNs. More formally, we say the GNN is as _powerful_ as 1-WL. We can generalise this to the $$k$$-hop neighbourhood where $$k \in \{2, 3, \dots\}$$. This wider neighbourhood can be viewed as a larger multiset of neighbours and _their_ neighbours, forming higher-order structures. When a GNN is able to discern these higher-order structures, we call it a $$k$$-WL GNN, and claim the GNN is as powerful as $$k$$-WL. Expressiveness is measured using these different "levels" of $$k$$-WL, altogether forming the _WL Hierarchy_. A $$k$$-WL GNN is strictly weaker than a $(k+1)$-WL GNN in that there exists a graph that the latter can discriminate and the former cannot but the converse is not true.

In fact, regular Message Passing Neural Networks fail 1-WL because aggregation functions like "mean" and "max" cannot tell apart two non-identical graphs. To avoid such scenarios, we introduce injectivity to the aggregation (multiset hashing) function; the "sum" aggregator is one such example.

<img src="/images/2022-04-24-expressive-gnns/aggrfails.png" width="100%">

__Figure 5:__ The graphs on the left cannot be discriminated using the "mean" aggregation. The graphs on the right cannot be discriminated using the "max" and "mean" aggregator. 

## Notable Works

## Conclusion

The research community has more or less moved away from standard GNNs towards expressive GNNs like those mentioned above. The obvious benefits include better structural awareness, which is paramount for real-life problems like protein studies, molecule interaction modelling, and social media analysis. I hope this blogpost shares some exciting insights about this new family of expressive GNNs. In terms of what's to come, I believe we need more benchmarking efforts to really compare these expressive GNNs with one another. This means coming up with new, dedicated datasets, both real-life and synthetic.

> I look forward to sharing more with you in time! Lots of exciting work to be done and lots of learnings and takeaways in store. Stay tuned!
>
> Till then, I'll see you in the next post :D