# Documents chains

## Stuff

**The simplest. It takes a list of documents, inserts them all into a prompt and passes that prompt to an LLM.**

![Alt text](../assets/document-stuff.png)

## Refine

**Looping over the input documents and iteratively updating its answer**

![Alt text](../assets/document-refine.png)

Advantage:
it is well-suited for tasks that require analyzing more documents than can fit in the model's context

Negative:
the Refine chain can perform poorly when documents frequently cross-reference one another or when a task requires detailed information from many documents.

## Map reduce

**Applies an LLM chain to each document individually (the Map step), treating the chain output as a new document.**

![Alt text](../assets/document-map-reduce.png)

## Map re-rank

The map re-rank documents chain runs an initial prompt on each document, that not only tries to complete a task but also gives a score for how certain it is in its answer. The highest scoring response is returned.

![Alt text](../assets/document-map-rerank.png)
