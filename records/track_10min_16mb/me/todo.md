* total tokens, steps to see all tokens

embeddings
* init std
* lr (tied vs untied)


rope base (tune for val seq len)?
logit softcap???
qk gain???

why casted linear (float32)?
* why so many optimizers?


# Ideas
* tie -> lora 
* low rank compression (enforce low rank?)

* some matrix params should not be muon-ed (see speedrun)
