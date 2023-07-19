class Embedding_Queues():
    def __init__(self, num_classes,max_length=4096):
        self.num_classes=num_classes
        self.max_length=max_length
        self.queues = [[] for j in range(self.num_classes) ]
    
    def enqueue(self,new_embeddings):
        for i in range(self.num_classes):
            if new_embeddings[i] is not None:
                self.queues[i] = self.queues[i] + [new_embeddings[i][j,:] for j in range(len(new_embeddings[i]))]
                if len(self.queues[i])>self.max_length:
                    self.queues[i] = self.queues[i][len(self.queues[i])-self.max_length:]
