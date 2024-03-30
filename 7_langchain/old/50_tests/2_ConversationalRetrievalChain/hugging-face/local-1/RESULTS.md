# google/flan-t5-large

```
Question: What is this app about?

Answer: Helpful


Question: What Assets are used in the app?

Token indices sequence length is longer than the specified maximum sequence length for this model (863 > 512). Running this sequence through the model will result in indexing errors
Answer: Asset


Question: Is in the code use async/await?

Answer: yes


Question: What is the most efficient class in the code?

Answer: ContractCondition


Question: What is the class hierarchy?

Answer: def __str__(self): return "BidPastLow: %d, AskPastHigh: %d" % (self.bidPastLow, self.askPastHigh) class TickAttribLast(Object): def __init__(self): self.pastLimit = False self.unreported = False def __str__(self): return "CanAutoExecute: %d, PastLimit: %d, PreOpen: %d" % (self.pastLimit, self.preOpen) class TickAttribBidAsk(Object): def __init__(self): self.bidPastLow = False self.askPastHigh = False def __str__(self): return "BidPastLow: %d, AskPastHigh: %d" % (self.bidPastLow, self.askPastHigh) aaa = super(OptionsTreeModel, self).setHeaderData(section, orientation, value, role) # print(aaa) return aaa def headerData(self, section, orientation, role=Qt.DisplayRole) -> object: # log.debug("Running...") # log.debug(locals()) if role == Qt.DisplayRole: aaa = self.header_data[section] # print(aaa) return aaa else: aaa = super(OptionsTreeModel, self).headerData(section, orientation, role) # print(aaa) return aaa # # ---------------------------------------------------------------------------------------------------


Question: What classes are derived from the DBObject class?

Answer: IBContractDetails(ContractDetails, DBObject)


NN takes: 182.74473428726196 sec.
```
