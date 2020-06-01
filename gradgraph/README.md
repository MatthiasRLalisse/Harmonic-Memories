Gradient Graphs
======================

Gradient Graphs (GGs) are a framework for Knowledge Base Completion based on the idea that semantic objects like knowledge base triplets `(e1, r, e2)` can be "cleaned up" by subjecting them to learned semantic well-formedness conditions encoded in the weights of a neural network. A Gradient Graph network is made up of a *compositional layer* **x** and a *hidden layer* **h**. The compositional layer is the result of applying a simple *composition function* to a triplet of two entity vectors and one relation vector. The result is a compositional representation of the graph triplet, obtained as a simple function of the triplet's parts.

![Schematic of Gradient Graph](/home/immanuel/Documents/Hopkins/papers/GradientGraphs/code__fork/0MASTER/GG_master/doc/gg_fig.png)

The hidden layer representation, on the other hand, don't depend just on the embeddings, but also on the network' s parameters: a weight matrix **W** that densely connects all the units in the hidden layer, along with a bias vector **b**, that define a measure of how well-formed a given configuration of the network is. We refer to the well-formedness of a particular state of the network as its **Harmony**, which is a function of both the compositional embedding and the hidden state of the network. The Harmony is a sum of a <span style="color:#656cf2">**Core Harmony**</span> term measuring the intrinsic well-formedness of the state vector **h**, and a <span style="color:#ef7070">**Faithfulness**</span> term that penalizes the internal state for getting two far from the compositional embedding. 

![Definition of Harmony](./doc/harmony_function.png)

The network' s hidden state can generally drift away from the compositional embedding, and for a given **x**, there is a *best* hidden state ![h_hat](./doc/h_hat.png) that offers the best balance between faithfulness to the input embeddings, and the network's knowledge about what semantically well-formed structures look like *in general*. The optimal **h** can thus be interpreted as the *best* embedding of the triplet, once it has been cleaned up by the network.

![h_hat def](./doc/h_hat_def.png)

 The resulting representations are context-sensitive, reflecting the ways that the meaning of a word or concept can change when it occurs in a complex structure. 

In Graph Completion tasks, we have shown that this kind "supracompositional" optimization of triplet embeddings leads to sizable improvements in the network's generalization of its knowledge to new triplets. The mechanism is essentially the following: the network uses its knowledge about what features of entities are significant to a particular inference about the entity. For instance, probing the representations of a trained ![HHolE](./doc/HHolE.png) (*Harmonic Holographic Embeddings*) model, we found that the neighborhoods of optimized triplet embeddings were sensitive to the relation in question. For instance, the compositionally-obtained embedding of the triplet `(barack_obama, political_office, us_president)`has neighbors that include non-presidents like `hillary_clinton` and `al_gore`. When the triplet is optimized, though, the "being-a-president" components of the embedding become more prominent, putting the optimized embedding in a neighborhood that only includes triplets with actual presidents in the first position. On the other hand, the compositional embedding of `(al_gore, political_office, us_president)` is closest to triplets with non-presidents in the first position--a region of the semantic space occupied by triplets with overall low values of Harmony. This region of the space therefore contains mainly negative examples of the `(?, political_office, us_president)` query (positive instances are displayed in **bold**). ![Type vs. Token Neighborhoods](./doc/gore-obama.png)



### The `gradgraph` module

To assist the `gradgraph` provides the basic code for training a gradient Knowledge Graph Completion (KBC) model, as well as examining the representations it has learned, allowing you to replicate the experiments in our paper. **@@@LINK TO PAPER@@@**

The demo script `HHolE_demo.py` illustrates the basic functionality of the `gradgraph` package. Start by importing `gradgraph` and initializing a task setting and model, setting the batch size and negative sampling rate (number of negative samples per positive example). A model is initialized with the number of entity and relation embeddings, the dimensionality of the entity embeddings, and the `lambda_` parameter controlling the Faithfulness penalty. If no `lambda_` value is passed, the model will treat `lambda_` as infinite, which simply means that the contextually optimal embedding is just the compositional embedding. Some models, like `HTPR`, also take a `relation_dim` parameter since relations and entities can have different dimensionality. 

```python
import gradgraph as gg

my_task = gg.KBETaskSetting(dataName='freebase', negsamples=250, batch_size=200)
my_model = gg.models.HHolE(n_entity=my_task.n_entity, n_relation=my_task.n_relation, entity_dim=50, lambda_=1., dataName='freebase')
```
```python
>>> print(my_model.name)
'HHolE50D1.0L.freebase'
```

You can also pass the task as an argument to the model initializer, e.g. `model = gg.models.HDistMult(task=my_task, entity_dim=50)`, which will automatically retrieve the required size of the entity and  relation embedding matrices (`n_entity` and `n_relation`). 

The `KBCTaskSetting` class loads in the training, test, and validation data, and has methods for training (with minibatching and negative sampling), as well as evaluating on validation and test sets. `gradgraph` is packaged with FB15K (`freebase`), WN18 (`wordnet`), and FB15K-237 (`fb-237`). If `dataName` is one of these preset values, loading will be automatic. Otherwise, point the task initializer to a`dataDirectory` containing properly formatted data:

```python
my_task = gg.KBETaskSetting(dataDirectory='path/to/my/data')
```

 New datasets can be freely added, but `KBCTaskSetting` expects files to have a standard format (see `data/README.md` for details). 

To train`my_model` for one epoch, save it, and then evaluate on the validation set, do:
```python
#train for one epoch
my_task.trainEpoch(my_model, interactive=True)
my_model.save()

#evaluate on validation set
results_valid = my_task.eval(my_model, interactive=True) 	
```
Setting `interactive=True` makes the model print incremental progress to `stdout`. By default, `my_task.eval` evaluates the model on the validation set. To evaluate on the test set, do:
```python
#evaluate on test set
results_test = my_task.eval(my_model, test_set=True, interactive=True) 	
```

Once you have a trained model, you can retrieve the learned parameters by evaluating the model's `params` attribute:

```python
params = my_model.params
W = params['weights']
b = params['bias']
e_embeddings = params['entity_embeddings']
r_embeddings = params['relation_embeddings']
```

## Pre-Trained Models

We provide two trained `HHolE` models trained on Freebase (dim = 512, lambda = 1.0)  and WordNet (dim = 256, lambda = 2.0). These will be loaded in automatically by initializing an `HHolE` object with the corresponding parameter settings: 

```python
#pre-trained freebase model
>>> fb_model = gg.models.HHolE(name='HHolE-freebase-gold')
>>> print(fb_model.hyperparams)
{'entity_dim': 512, 'relation_dim': 512, 'h_dim': 512, 'lambda': 1.0, 'n_entity': 14951, 'n_relation': 1345}
```

```python
#pre-trained wordnet model
>>> wn_model = gg.models.HHolE(name='HHolE-wordnet-gold')
>>> print(wn_model.hyperparams)
{'entity_dim': 256, 'relation_dim': 256, 'h_dim': 256, 'lambda': 2.0, 'n_entity': 40943, 'n_relation': 18}
```

To replicate the results from our paper, run e.g.:

```python
#evaluate gold-standard FB15K model
>>> fb_task = gg.KBETaskSetting('freebase')
>>> fb_results = fb_task.eval(fb_model, test_set=True)
>>> for key in ['MR', 'MRR', 'Hits1', 'Hits3', 'Hits10']: 
    					print('\t%s: %.3f' % (key, fb_results[key]))
	MR: 21.420
	MRR: 0.795
	Hits1: 0.727
	Hits3: 0.848
	Hits10: 0.901
```

```python
#evaluate gold-standard WN18 model
>>> wn_task = gg.KBETaskSetting('wordnet')
>>> wn_results = wn_task.eval(wn_model, test_set=True)
>>> for key in ['MR', 'MRR', 'Hits1', 'Hits3', 'Hits10']: 
    					print('\t%s: %.3f' % (key, wn_results[key]))
	MR: 189.276
	MRR: 0.939
	Hits1: 0.931
	Hits3: 0.945
	Hits10: 0.951
```

Analyzing the representations
----------------------------

`gradgraph` also allows you to analyze the representations learned by the network. To use this functionality, load in a trained model and initialize a `gg.Analyzer` object with a trained model:

```python
analyzer = gg.Analyzer(my_model)
```
The analyzer will read in the database and attempt to load human-readable names for entities. We provide the mapping from machine IDs to human-readable names for both WN18 and FB15K. The human-readable version of FB15K is based on a mapping between Freebase and Wikipedia article names built by [Ling & Weld 2012](https://github.com/xiaoling/figer). 

### Navigating the database

To search the database for entities or relations related to some query of interest, do:

```python
>>> analyzer.search('Miley Cyrus', type='e')	#entities matching 'Miley Cyrus'
['Miley Cyrus', 'Miley cyrus', 'MileyRayCyrus', 'Destiny Hope Cyrus (Miley)', 'Miley Ray Cyrus', 'Smiley cyrus', 'Miley Cyrus tours', 'Mileycyrus', 'List of Miley Cyrus concert tours']
>>> analyzer.search('music album', type='r')	#relations related to music albums
['/music/album/genre', '/music/artist/album', '/music/genre/albums', '/music/album/artist', '/music/album_release_type/albums']
```
This process involves a simple string match search through entities or relations in the database. To find triplets about `Miley Cyrus` that are actually in the database (i.e. true facts about `Miley Cyrus`), use the `tripletsAbout` method:
```python
>>> analyzer.tripletsAbout('Miley Cyrus')
{('Miley Cyrus', '/award/award_nominee/award_nominations./award/award_nomination/award', 'Golden Globe Award for Best Original Song'), ('Miley Cyrus', '/music/group_member/instruments_played', 'Acoustic guitar') ... }
```

### Examining triplet embeddings

Gradient Graph triplets are optimized according to learned well-formedness conditions encoded in the weight matrix and bias vector. The input layer **x** is obtained as a compositional function of the entity and relation embeddings. The hidden layer representations **h**, on the other hand, aren't simple functions of the entity and relation embeddings. The **h** vectors also depend on the well-formedness of the triplet representation, which depends on the weight matrix **W** and bias vector **b**. 

`gg` also allows you to probe the network representations of triplets to retrieve token representations. For entities corresponding to `Selena Gomez` and `Miley Cyrus` queried for whether they are `friends`, you can predict the compositional and supracompositional triplet embeddings using `model.predict`:

```python
e1_idx = analyzer.name2idx['Selena Gomez']
r_idx = analyzer.relation2idx['/celebrities/celebrity/celebrity_friends.'\
                              					'/celebrities/friendship/friend'] 
e2_idx = analyzer.name2idx['Miley Cyrus']
x_compositional, h_optimized = model.predict(e1_idx, r_idx, e2_idx)
```

It is instructive to compare the compositional triplet neighborhoods with the neighborhoods of the optimized triplet embeddings. This functionality is packed into the `neighbors` method. For a given triplet `t = (e1, r, e2)`, `neighbors` finds the closest triplets to `t` among those obtained by replacing either `e1` and `e2` with another entity. Both neighbors of the compositional embeddings (`TOP X`), and neighbors of the optimized triplet embedding (`TOP H`) are displayed, <sup>[1](#myfootnote1)</sup> along with whether the triplet embedding actually occurs in the database (`X IN DB` and `H IN DB`, displaying `X` if the corresponding triplet occurs in the database), and its rank (`X RANK` and `H RANK`) among candidate entities according to the Harmony scoring function.

```python
>>> r_name = '/government/politician/government_positions_held.'\
				'/government/government_position_held/office_position_or_title'
>>> analyzer.neighbors('John McCain', r_name, 'US President', direction='l', top_n=5)
Querying e1: ('John McCain', '/government/politician/government_positions_held./government/government_position_held/office_position_or_title', 'US President')
  n  TOP X                   X RANK    X IN DB    TOP H         H RANK    H IN DB
---  ----------------------  --------  ---------  ------------  --------  ---------
  0  John McCain             TypeErr              John McCain   TypeErr
  1  John Kerry              39                   John Kerry    39
  2  Hillary Rodham Clinton  35                   Colin Powell  41
  3  Colin Powell            41                   Nancy Pelosi  43
  4  Richard Nixon           31        X          Joe Biden     45
  5  Herbert Hoover          37        X          Dick Cheney   40

```

If the queried entity does not satisfy the relation's type constraints, its rank is displayed as `TypeErr`. This behavior can be overridden by querying the entity's neighborhood among *all* entities using the `type_constrain=False` keyword:

```python
>>> analyzer.neighbors(q_name, r_name, 'US President', direction='l', type_constrain=False)
Querying e1: ('John McCain', '/government/politician/government_positions_held./government/government_position_held/office_position_or_title', 'US President')
  n  TOP X                     X RANK  X IN DB    TOP H                H RANK  H IN DB
---  ----------------------  --------  ---------  -----------------  --------  --------
  0  John McCain                   58             John McCain              58
  1  John Conyers                  45             John Conyers             45
  2  John Kerry                    42             John Kerry               42
  3  Charles B. Rangel             98             Ron Paul                 60
  4  Hillary Rodham Clinton        35             Charles B. Rangel        98
  5  Ron Paul                      60             Robert Byrd              73
```

### Getting entity tokens

For certain models, you can also obtain approximate *token embeddings* of a given entity by inverting the triplet composition function. Except for extremely simple composition models (e.g. concatenation of entity and relation vectors), there isn't a unique factorization of triplet representations into a composition of entity and relation tokens that might have generated them. For instance, ![HDistMult](./doc/HDistMult.png) compositional embeddings are obtained via elementwise multiplication of the entity and relation vectors:

![HDistMult composition function](./doc/HDistMult_composition_fn.png)

For any given value of `x_HDM`, there are infinitely many vectors `e1`, `r`, and `e2` satisfying the above equality. But by fixing `r` and `e2` to their token embeddings, we can solve for the contextually optimal embedding of `e1` by multiplying ![h_hat](./doc/h_hat_.png) by the elementwise reciprocals of `r` and `e` :

![e_hat](./doc/e_hat_1.png)

To retrieve these contextually optimal entity embeddings, use `model.predict_token`, which computes contextually optimal entity tokens using the type embeddings for the two remaining triplet components. 

```python
e1_embedding, e1_token_embedding = model.predict_token(e1_idx, r_idx, e2_idx, direction='l')
e2_embedding, e2_token_embedding = model.predict_token(e1_idx, r_idx, e2_idx)
															#direction defaults to 'r'
```

Since composition functions vary between models, the inverses of these composition functions also vary, and need to be defined for each model separately (see the files in `gradgraph/models` for examples).

The `neighbors ` method of the `gg.Analyzer` class also allows comparison of semantic neighborhoods using *entity* types and tokens rather than compositional *triplet* types and tokens  by setting `entity_tokens` to `True` in an `analyzer.neighbors` call. The following compares the entity embedding of `'John McCain'` to other entity embeddings, and the token embedding of `John McCain` in the context of the query to token embeddings of other entities in the context of the same query.

```python
analyzer.neighbors('John McCain', r_name, 'US President', direction='l', entity_tokens=True)	 
  n  TOP X             X RANK    X IN DB    TOP H                     H RANK    H IN DB
---  ----------------  --------  ---------  ------------------------  --------  -------
  0  John McCain       TypeErr              John McCain               TypeErr
  1  Richard Nixon     31        X          Joe Biden                 45
  2  John Kerry        39                   William Pitt the Younger  46
  3  Ulysses S. Grant  10        X          Nancy Pelosi              43
  4  William McKinley  29        X          Colin Powell              41
  5  Dick Cheney       40                   DeWitt Clinton            49

```

<a name="myfootnote1">1</a>: More precisely,  for`analyzer.neighbors` compares two different sets of triplet embeddings: (1) compositional triplet embeddings, and (2) the optimized triplet embeddings. When computing the neighborhood of the compositional triplet ``t = (e1, r, e2)``, we compare `t` with other compositional triplet embeddings. On the other hand, when computing the semantic neighborhood for the optimized embedding `hat_t`, we compare it with other optimized triplet embeddings.